"""Isolated Lorenz bifurcation comparison: two ICs, extrema and stationary states.

Imports the existing checkpoint/scaling loaders without modifying them. The
true Lorenz ODE is defined here; no external plotting backend is required.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
ALRNN_DIR = HERE.parent
if str(ALRNN_DIR) not in sys.path:
    sys.path.insert(0, str(ALRNN_DIR))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import torch

from alrnn_model import load_saved_model
from compare_model_bifurcation_diagrams import load_training_coordinates
from bifurcation_training_markers import alrnn_training_parameters, TrainingParameterRange, add_training_range
from run_term_ablations import Step, schedule, validate_gpu_ids


VERSION = 1
INITIAL_STATES = np.array([[1., 1., 1.], [-1., -1., 1.]])
DEFAULT_RUN = ALRNN_DIR / "runs/term_ablations/lorenz_rho/combined_lorenz_rho_balanced/20261103_165123_417235_497351"
DATASETS = ("lorenz_rho_5_to_23", "lorenz_rho_1_to_30")
LABELS = ("rho in [5, 23]", "rho in [1, 30]")
VARIANTS = {"cphi": "Cphi", "cphi_shifted_relu": "Cphi + shifted ReLU"}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--training-data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--variant", choices=tuple(VARIANTS), help="Internal worker selection; omitted runs both variants.")
    parser.add_argument("--num-parameters", type=int, default=290)
    parser.add_argument("--parameter-min", type=float, default=1.)
    parser.add_argument("--parameter-max", type=float, default=30.)
    parser.add_argument("--dt", type=float, help="Defaults to the training data's dt; model step size must match it.")
    parser.add_argument("--warmup-time", type=float, default=200.)
    parser.add_argument("--observation-time", type=float, default=50.)
    parser.add_argument("--stationary-atol", type=float, default=1e-5)
    parser.add_argument("--stationary-rtol", type=float, default=1e-5)
    parser.add_argument("--stationary-rate", type=float, default=1e-3, help="Maximum final rate in standardized observations per time unit.")
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for field in ("run_root", "training_data_root", "output_root"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    if args.output_root is None:
        args.output_root = HERE / "results" / args.run_root.name
    if args.training_data_root is None:
        plan = args.run_root / "plan.json"
        saved = json.loads(plan.read_text(encoding="utf-8-sig")) if plan.is_file() else {}
        args.training_data_root = Path(saved.get("args", {}).get("training_data_root",
            "/export/home/klenkeit/training_data/mixed_parameter_trajectories/lorenz_rho/combined_lorenz_rho")).expanduser().resolve()
    validate_gpu_ids(args.gpu_ids, max_gpus=6)
    if args.num_parameters < 2 or args.cpu_threads < 1 or args.batch_size < 1 or args.dpi < 40:
        parser.error("Use >=2 parameter values and positive CPU threads/batch size; dpi must be >=40.")
    for field in ("warmup_time", "observation_time", "stationary_atol", "stationary_rate", "rtol", "atol"):
        if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            parser.error(f"{field} must be finite and positive")
    if not math.isfinite(args.stationary_rtol) or args.stationary_rtol < 0:
        parser.error("stationary_rtol must be finite and non-negative")
    if not np.isfinite([args.parameter_min, args.parameter_max]).all() or not 1 <= args.parameter_min < args.parameter_max <= 30:
        parser.error("Use a parameter subrange inside [1, 30].")
    if args.plot_only and args.recompute:
        parser.error("--plot-only cannot be combined with --recompute")
    return args


def preflight(args):
    scales = []
    steps = []
    for name in DATASETS:
        folder = args.training_data_root / name
        if (folder / ".compact_in_progress.json").exists():
            raise ValueError(f"Dataset conversion is incomplete: {folder}")
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
        if metadata.get("config", {}).get("regime") != "lorenz_rho_1_to_30":
            raise ValueError(f"Expected Lorenz-rho training data: {folder}")
        scaling = load_training_coordinates(folder)
        if scaling.mean.shape != (3,):
            raise ValueError(f"Expected three observed dimensions: {folder}")
        if not np.allclose([scaling.phi_boundary, scaling.phi_scale], [24.74, 23.74], atol=1e-7):
            raise ValueError(f"Unexpected Lorenz phi reference: {folder}")
        scales.append(scaling)
        steps.append(float(metadata["config"]["dt"]))
    if not np.allclose(steps, steps[0], atol=0., rtol=1e-12) or not np.isfinite(steps).all() or min(steps) <= 0:
        raise ValueError("Training datasets need the same finite positive dt.")
    if args.dt is None:
        args.dt = steps[0]
    if not math.isfinite(args.dt) or not math.isclose(args.dt, steps[0], rel_tol=1e-10):
        raise ValueError("Evaluation dt must equal the trained model's sampling interval.")
    for field in ("warmup_time", "observation_time"):
        count = getattr(args, field) / args.dt
        if not math.isclose(count, round(count), abs_tol=1e-7):
            raise ValueError(f"{field} must be a multiple of dt.")
    if round(args.observation_time / args.dt) < 3:
        raise ValueError("Observation window needs at least four states.")
    return scales


def settings(args, reference):
    return dict(version=VERSION, analysis_code_sha256=digest(__file__),
                checkpoint_loader_sha256=digest(ALRNN_DIR / "alrnn_model.py"),
                sigma=10., beta=8./3., initial_states=INITIAL_STATES.tolist(),
                parameter_values=np.linspace(args.parameter_min, args.parameter_max, args.num_parameters).tolist(),
                dt=args.dt, warmup_time=args.warmup_time, observation_time=args.observation_time,
                stationary_atol=args.stationary_atol, stationary_rtol=args.stationary_rtol,
                stationary_rate=args.stationary_rate, rtol=args.rtol, atol=args.atol,
                reference_coordinates=reference.metadata())


def lorenz_rhs(time, state, rho):
    x, y, z = state
    return np.array([10. * (y-x), rho*x-y-x*z, x*y-(8./3.)*z])


def extract_points(tail, rate, config):
    """Minima/maxima, or one stationary point, with an all-observations test."""
    tail = np.asarray(tail, dtype=np.float64)
    if tail.ndim != 2 or tail.shape[1] != 3 or len(tail) < 4:
        raise ValueError("Expected at least four three-dimensional observations.")
    if not np.isfinite(tail).all() or not np.isfinite(rate).all():
        return np.empty(0), np.empty(0, dtype=np.int8), dict(status="non_finite")
    span = np.ptp(tail, axis=0)
    tolerance = config["stationary_atol"] + config["stationary_rtol"] * np.maximum(1., np.max(np.abs(tail), axis=0))
    stationary = bool(np.all(span <= tolerance) and np.max(np.abs(rate)) <= config["stationary_rate"])
    diagnostic = dict(status="stationary" if stationary else "oscillatory_or_transient",
                      span=span.tolist(), final_rate=np.asarray(rate).tolist())
    if stationary:
        return np.array([np.mean(tail[:, 0])]), np.array([0], dtype=np.int8), diagnostic
    maxima, _ = find_peaks(tail[:, 0])
    minima, _ = find_peaks(-tail[:, 0])
    indices = np.r_[minima, maxima]
    kinds = np.r_[np.full(len(minima), -1, dtype=np.int8), np.ones(len(maxima), dtype=np.int8)]
    if not len(indices):
        diagnostic["status"] = "no_extrema_not_stationary"
    return tail[indices, 0], kinds, diagnostic


def point_arrays(records):
    return dict(rho=np.asarray([r[0] for r in records], dtype=np.float64),
                value=np.asarray([r[1] for r in records], dtype=np.float64),
                kind=np.asarray([r[2] for r in records], dtype=np.int8),
                initial_condition=np.asarray([r[3] for r in records], dtype=np.int8))


def add_records(records, values, kinds, rho, initial):
    records.extend((float(rho), float(value), int(kind), initial) for value, kind in zip(values, kinds))


def save_cache(path, arrays, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays, metadata_json=json.dumps(metadata, allow_nan=False))
    temporary.replace(path)


def read_cache(path, expected):
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        if metadata.get("signature") != identity(expected):
            raise ValueError(f"Cache settings or model identity differ: {path}. Choose another output root or use --recompute.")
        return {key: source[key] for key in source.files if key != "metadata_json"}, metadata


def ground_truth(args, reference, config):
    path = args.output_root / "ground_truth.npz"
    expected = dict(kind="lorenz_two_initial_states_truth", settings=config)
    if path.is_file() and not args.recompute:
        print(f"Reusing Lorenz ground truth: {path}", flush=True)
        return read_cache(path, expected)
    if args.plot_only or args.variant is not None:
        raise FileNotFoundError(f"Required ground-truth cache: {path}")
    steps = int(round(args.observation_time / args.dt))
    times = args.warmup_time + np.arange(steps + 1) * args.dt
    records, diagnostics = [], []
    for index, rho in enumerate(config["parameter_values"]):
        for initial, x0 in enumerate(INITIAL_STATES):
            solution = solve_ivp(lambda t, state: lorenz_rhs(t, state, rho), (0., times[-1]), x0,
                                 t_eval=times, method="DOP853", rtol=args.rtol, atol=args.atol,
                                 max_step=5.*args.dt)
            if not solution.success or solution.y.shape != (3, len(times)):
                raise RuntimeError(f"Lorenz integration failed at rho={rho}, IC={initial}: {solution.message}")
            tail = (solution.y.T - reference.mean) / reference.std
            rate = lorenz_rhs(times[-1], solution.y[:, -1], rho) / reference.std
            values, kinds, diagnostic = extract_points(tail, rate, config)
            if diagnostic["status"] == "non_finite":
                raise RuntimeError(f"Non-finite ground truth at rho={rho}, IC={initial}")
            add_records(records, values, kinds, rho, initial)
            diagnostics.append(dict(rho=rho, initial_condition=initial, **diagnostic))
        if (index + 1) % 10 == 0 or index + 1 == args.num_parameters:
            print(f"Ground truth: {index + 1}/{args.num_parameters} rho values, both ICs", flush=True)
    metadata = dict(signature=identity(expected), **expected, diagnostics=diagnostics)
    arrays = point_arrays(records)
    save_cache(path, arrays, metadata)
    return arrays, metadata


@torch.inference_mode()
def model_tails(model, initial, phi, warmup_steps, observation_steps, device, batch_size, progress=False):
    """Seed at t=0, free-run the MODEL's own warm-up, retain only its tail."""
    total = warmup_steps + observation_steps
    for start in range(0, len(initial), batch_size):
        stop = min(start + batch_size, len(initial))
        count = stop-start
        x0 = torch.tensor(initial[start:stop], dtype=torch.float32, device=device)
        latent = x0 @ model.B
        latent[:, :3] = x0
        inputs = torch.tensor(phi[start:stop], dtype=torch.float32, device=device).reshape(-1, 1)
        tail = torch.full((count, observation_steps+1, 3), float("nan"), device=device)
        failed = torch.full((count,), -1, dtype=torch.int64, device=device)
        alive = torch.ones(count, dtype=torch.bool, device=device)
        for step in range(1, total+1):
            updated = model(latent, inputs if model.phi_dim else None)
            finite = torch.isfinite(updated).all(dim=1)
            failed = torch.where(alive & ~finite, step, failed)
            alive = alive & finite
            latent = torch.where(alive[:, None], updated, 0.)
            if step >= warmup_steps:
                tail[:, step-warmup_steps] = torch.where(alive[:, None], latent[:, :3], float("nan"))
            if progress and (step % max(1, total//5) == 0 or step == total):
                print(f"Rollout batch {start + 1}-{stop}: {step}/{total} steps", flush=True)
        yield start, stop, tail.cpu().numpy(), failed.cpu().numpy()


def model_source(run, scaling):
    metadata = json.loads((run / "metrics.json").read_text(encoding="utf-8-sig"))
    return dict(run_dir=str(run), checkpoint_sha256=digest(run / "model.pt"),
                metrics_sha256=digest(run / "metrics.json"), model_config=metadata.get("model_config"),
                training_coordinates=scaling.metadata())


def model_points(args, model, scaling, reference, config):
    parameters = np.repeat(np.asarray(config["parameter_values"]), len(INITIAL_STATES))
    physical_initial = np.tile(INITIAL_STATES, (args.num_parameters, 1))
    initial = ((physical_initial-scaling.mean)/scaling.std).astype(np.float32)
    phi = (parameters-scaling.phi_boundary)/scaling.phi_scale
    warmup = int(round(args.warmup_time/args.dt))
    observed = int(round(args.observation_time/args.dt))
    records, diagnostics = [], []
    device = f"cuda:{args.gpu_ids[0]}" if args.device == "cuda" else "cpu"
    for start, stop, tails, failures in model_tails(model, initial, phi, warmup, observed, device, args.batch_size, progress=True):
        common = (tails.astype(np.float64)*scaling.std+scaling.mean-reference.mean)/reference.std
        for offset, tail in enumerate(common):
            index = start+offset
            rate = (tail[-1]-tail[-2])/args.dt
            values, kinds, diagnostic = extract_points(tail, rate, config)
            failure = int(failures[offset])
            if failure >= 0:
                values, kinds = np.empty(0), np.empty(0, dtype=np.int8)
                diagnostic = dict(status="diverged", failed_step=failure)
            add_records(records, values, kinds, parameters[index], index % 2)
            diagnostics.append(dict(rho=float(parameters[index]), initial_condition=index % 2, **diagnostic))
        print(f"Model: {stop}/{len(parameters)} trajectories (including own warm-up)", flush=True)
    return point_arrays(records), diagnostics


def training_ranges(args, names):
    regime = type("LorenzRegime", (), {"name": "lorenz_rho_1_to_30"})()
    bounds = []
    for name in names:
        folder = args.training_data_root / name
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
        values = alrnn_training_parameters(folder, metadata, regime, {"multi_trajectory": True})
        if values is None:
            raise ValueError(f"Could not read training parameter coverage: {folder}")
        bounds.append(values)
    return bounds


def plot_results(folder, truth, models, labels, bounds, config, dpi):
    import matplotlib.pyplot as plt

    data = [truth, *models]
    titles = ["Ground Truth", *labels]
    parameter = np.asarray(config["parameter_values"])
    paths = []
    for individual in (None, *range(len(data))):
        indices = list(range(len(data))) if individual is None else [individual]
        figure, axes = plt.subplots(len(indices), 1, figsize=(14, 3.8*len(indices)),
                                    sharex=True, sharey=True, squeeze=False, layout="constrained")
        for axis, index in zip(axes[:, 0], indices):
            points = data[index]
            stationary = points["kind"] == 0
            for mask, size, alpha in ((~stationary, 1.4, .55), (stationary, 5., .85)):
                axis.scatter(points["rho"][mask], points["value"][mask], color="black",
                             s=size, alpha=alpha, linewidths=0)
            axis.set_title(titles[index])
            axis.set_ylabel("x extrema / stationary x, standardized")
            axis.set_xlim(config["parameter_values"][0], config["parameter_values"][-1])
            axis.grid(alpha=.12)
            if index:
                values = bounds[index-1]
                if isinstance(values, TrainingParameterRange):
                    add_training_range(axis, values, parameter)
                else:
                    for value in values:
                        axis.axvline(float(value), color="tab:blue", alpha=.35, linewidth=.8)
        axes[-1, 0].set_xlabel("rho")
        filename = "bifurcation_x_compare.png" if individual is None else (
            "bifurcation_x_ground_truth.png" if individual == 0 else f"bifurcation_x_model_{individual:02d}.png")
        path = folder / filename
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        paths.append(path.name)
    return paths


def worker(args, scales, truth, config):
    folder = args.output_root / args.variant
    folder.mkdir(parents=True, exist_ok=True)
    sources = [model_source(args.run_root / args.variant / name, scale) for name, scale in zip(DATASETS, scales)]
    expected = dict(kind="lorenz_two_initial_states_models", settings=config, models=sources,
                    ground_truth_sha256=digest(args.output_root/"ground_truth.npz"))
    path = folder / "points.npz"
    if path.is_file() and not args.recompute:
        arrays, metadata = read_cache(path, expected)
        print(f"Reusing model points: {path}", flush=True)
    else:
        if args.plot_only:
            raise FileNotFoundError(f"--plot-only requires {path}")
        arrays, model_diagnostics = {}, []
        device = f"cuda:{args.gpu_ids[0]}" if args.device == "cuda" else "cpu"
        for index, (name, scaling) in enumerate(zip(DATASETS, scales)):
            run = args.run_root / args.variant / name
            model, _, _ = load_saved_model(run, device)
            if model.N != 3 or model.phi_dim != 1:
                raise ValueError(f"Expected a three-observation, one-parameter model: {run}")
            points, diagnostics = model_points(args, model, scaling, scales[0], config)
            arrays.update({f"model_{index}_{key}": value for key, value in points.items()})
            model_diagnostics.append(diagnostics)
            del model
        metadata = dict(signature=identity(expected), **expected, diagnostics=model_diagnostics)
        save_cache(path, arrays, metadata)
    models = [{key: arrays[f"model_{index}_{key}"] for key in ("rho", "value", "kind", "initial_condition")} for index in range(len(DATASETS))]
    labels = [f"{VARIANTS[args.variant]}: {label}" for label in LABELS]
    bounds = training_ranges(args, DATASETS)
    paths = plot_results(folder, truth, models, labels, bounds, config, args.dpi)
    provenance = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
                      analysis=metadata, plots=paths, ground_truth_file=str(args.output_root/"ground_truth.npz"),
                      script_sha256=digest(__file__), checkpoint_loader_sha256=digest(ALRNN_DIR/"alrnn_model.py"),
                      kind_codes={"-1": "local minimum", "0": "stationary observation", "1": "local maximum"},
                      method="Both exact physical initial states, own model warm-up, all-3-observations stationarity test; no teacher forcing or truth burn-in initialization. One stationary x per converged IC; otherwise all local x minima/maxima. Non-finite rollouts excluded and recorded.",
                      paper="https://arxiv.org/html/2507.19036v2#A4")
    (folder / "bifurcation_sources.json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    failures = sum(d["status"] == "diverged" for diagnostics in metadata["diagnostics"] for d in diagnostics)
    print(f"{args.variant}: {failures} divergent trajectories recorded. Plots: {folder}", flush=True)


def worker_command(args, variant, gpu):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--variant", variant,
               "--run-root", str(args.run_root), "--training-data-root", str(args.training_data_root),
               "--output-root", str(args.output_root), "--gpu-ids", str(gpu), "--device", args.device]
    for field in ("num_parameters", "parameter_min", "parameter_max", "dt", "warmup_time", "observation_time",
                  "stationary_atol", "stationary_rtol", "stationary_rate", "rtol", "atol", "cpu_threads", "batch_size", "dpi"):
        if getattr(args, field) is not None:
            command.extend(["--"+field.replace("_", "-"), str(getattr(args, field))])
    # The parent has already refreshed the shared truth cache when requested.
    if args.recompute:
        command.append("--recompute")
    return command


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(run_root=str(args.run_root), training_data_root=str(args.training_data_root),
                              output_root=str(args.output_root), variants=list(VARIANTS), gpu_ids=args.gpu_ids,
                              initial_states=INITIAL_STATES.tolist(), warmup_time=args.warmup_time,
                              observation_time=args.observation_time, num_parameters=args.num_parameters,
                              coordinate_system="fixed training standardization", no_training=True), indent=2))
        return 0
    torch.set_num_threads(args.cpu_threads)
    scales = preflight(args)
    config = settings(args, scales[0])
    if not args.plot_only and args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu for a small local check.")
        validate_gpu_ids(args.gpu_ids, device_count=torch.cuda.device_count(), max_gpus=6)
    chosen = [args.variant] if args.variant else list(VARIANTS)
    for variant in chosen:
        for name in DATASETS:
            for filename in ("model.pt", "metrics.json"):
                path = args.run_root / variant / name / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing trained model: {path}")
    # Workers read the parent's truth cache even under --recompute.
    truth_args = argparse.Namespace(**vars(args))
    if args.variant is not None:
        truth_args.recompute = False
    truth, _ = ground_truth(truth_args, scales[0], config)
    if args.variant is not None:
        worker(args, scales, truth, config)
        return 0
    if args.plot_only or args.device == "cpu":
        for variant in VARIANTS:
            args.variant = variant
            worker(args, scales, truth, config)
        return 0
    log_dir = args.output_root / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir.mkdir(parents=True)
    gpu_file = log_dir / "gpu_ids.json"
    gpu_file.write_text(json.dumps(args.gpu_ids)+"\n")
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    for field in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[field] = str(args.cpu_threads)
    print(f"Variant logs: {log_dir}\nAll results: {args.output_root}", flush=True)
    def steps(variant, gpu):
        return [Step(f"Lorenz minima/maxima and stationary states: {variant}", worker_command(args, variant, gpu), HERE)]
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        results = schedule(gpu_file, args.gpu_ids, torch.cuda.device_count(), log_dir, steps, env,
                           job_names=list(VARIANTS), max_gpus=6)
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
    (log_dir/"summary.json").write_text(json.dumps(results, indent=2)+"\n")
    return int(any(code != 0 for code in results.values()))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
