"""Paired ground-truth / saved ALRNN ensembles under one common parameter drift.

Only this directory is extended. Existing single-trajectory simulation and model
loading are reused; no training code is imported. All default outputs stay here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

# Importing existing helpers must not create files in their source directories.
sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
from scipy.signal import find_peaks
import torch

import evaluate_parameter_drift as single


CACHE_VERSION = 1
SIMULATION_KEYS = tuple(single.DEFAULTS) + (
    "parameter_start", "parameter_end", "phi_start", "phi_end", "initial_state",
)
PATH_KEYS = ("output_dir", "regime_file", "normalization_stats", "ground_truth", "plotly_js")


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=SCRIPT_DIR / "halvorsen_ensemble_config.json")
    selected, _ = pre.parse_known_args(argv)
    config_path = selected.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("The ensemble config must be a JSON object.")
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--model-runs", type=Path, nargs="+")
    parser.add_argument("--model-labels", nargs="+")
    for key in PATH_KEYS:
        parser.add_argument("--" + key.replace("_", "-"), type=Path)
    parser.add_argument("--regime")
    parser.add_argument("--n-trajectories", type=int, default=200)
    parser.add_argument("--steps", type=int, default=4000, help="Stored states including the post-burn-in initial state.")
    parser.add_argument("--burn-in-steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--initial-state", type=float, nargs=3)
    for key in ("parameter_start", "parameter_end", "phi_start", "phi_end"):
        parser.add_argument("--" + key.replace("_", "-"), type=float)
    parser.add_argument("--drift", choices=tuple(single.DRIFTS), default="linear")
    parser.add_argument("--sigmoid-steepness", type=float, default=10.)
    parser.add_argument("--ramp-fraction", type=float, default=0.5)
    parser.add_argument("--normalization", choices=("trajectory", "fixed", "none"), default="trajectory")
    parser.add_argument("--plot-space", choices=("physical", "standardized"), default="physical")
    parser.add_argument("--component", choices=("x", "y", "z"), default="z")
    parser.add_argument("--peak-prominence", type=float, default=0., help="In the selected plot coordinates; zero means no prominence filter.")
    parser.add_argument("--peak-distance-steps", type=int, default=1)
    parser.add_argument("--max-simulation-attempts", type=int, default=10)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--max-step-multiplier", type=float, default=5.)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--ground-truth-only", action="store_true")
    parser.add_argument("--regenerate-ground-truth", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Reuse both truth and prediction arrays; no ODE integration or inference.")
    parser.add_argument("--dry-run", action="store_true")
    destinations = {action.dest for action in parser._actions}
    unknown = set(config) - destinations
    if unknown:
        raise ValueError(f"Unknown ensemble config fields: {sorted(unknown)}")
    for key in PATH_KEYS:
        if config.get(key) is not None:
            config[key] = (config_path.parent / Path(config[key]).expanduser()).resolve()
    if config.get("model_runs") is not None:
        config["model_runs"] = [(config_path.parent / Path(p).expanduser()).resolve() for p in config["model_runs"]]
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    for key in PATH_KEYS + ("config",):
        if getattr(args, key) is not None:
            setattr(args, key, Path(getattr(args, key)).expanduser().resolve())
    args.model_runs = [Path(p).expanduser().resolve() for p in (args.model_runs or [])]
    if args.output_dir is None:
        args.output_dir = SCRIPT_DIR / "results/ensemble_drift"
    if args.ground_truth is None:
        args.ground_truth = args.output_dir / "ground_truth.npz"
    if args.ground_truth.suffix != ".npz":
        parser.error("--ground-truth must be an .npz file")
    if args.regime is None:
        args.regime = "halvorsen_a_chaotic_to_periodic"
    # Overriding the model list should not retain labels for the preset's model.
    passed = list(sys.argv[1:] if argv is None else argv)
    if "--model-runs" in passed and "--model-labels" not in passed:
        args.model_labels = None
    if any(flag in passed for flag in ("--phi-start", "--phi-end")):
        if not any(flag in passed for flag in ("--parameter-start", "--parameter-end")):
            args.parameter_start = args.parameter_end = None
    if args.model_labels is not None and len(args.model_labels) != len(args.model_runs):
        parser.error("--model-labels needs one entry per model run")
    for key in ("n_trajectories", "rollout_batch_size", "max_simulation_attempts", "cpu_threads", "peak_distance_steps"):
        if not isinstance(getattr(args, key), int) or getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.seed < 0 or args.gpu_id < 0:
        parser.error("seed and GPU ID must be non-negative")
    if not math.isfinite(args.peak_prominence) or args.peak_prominence < 0:
        parser.error("peak prominence must be finite and non-negative")
    if args.plot_only and (args.ground_truth_only or args.regenerate_ground_truth):
        parser.error("--plot-only cannot regenerate ground truth")
    if not args.model_runs and not args.ground_truth_only:
        parser.error("Select --model-runs or --ground-truth-only")
    return args


def simulation_spec(args, regime):
    options = single.parse_args([])
    for key in SIMULATION_KEYS:
        setattr(options, key, getattr(args, key))
    options.normalization_stats = args.normalization_stats
    single.validate_options(options)
    return single.make_simulation_spec(options, regime)


def cache_request(args, regime, regime_path):
    return dict(
        simulation=simulation_spec(args, regime), n_trajectories=args.n_trajectories,
        max_simulation_attempts=args.max_simulation_attempts,
        regime_definition=asdict(regime), regime_file_sha256=single.file_hash(regime_path),
        normalization_stats_sha256=single.file_hash(args.normalization_stats) if args.normalization_stats else None,
    )


def simulate_ensemble(module, regime, spec, n_trajectories, max_attempts=10, scaling_path=None, progress=True):
    values = {key: [] for key in ("x", "observed", "mean", "std", "initial_before_burn")}
    attempts_used, failures = [], []
    children = np.random.SeedSequence(spec["seed"]).spawn(n_trajectories)
    for index, child in enumerate(children):
        # Independent deterministic streams also keep later trajectories stable
        # when an earlier trajectory needs another initial-state draw.
        rng = np.random.default_rng(child)
        for attempt in range(1, max_attempts + 1):
            attempt_spec = dict(spec, seed=int(rng.integers(0, 2**32)))
            try:
                sample = single.simulate_ground_truth(module, regime, attempt_spec, scaling_path)
                break
            except RuntimeError as error:
                failures.append(dict(trajectory=index, attempt=attempt, seed=attempt_spec["seed"], error=str(error)))
                if attempt == max_attempts or spec["initial_state"] is not None:
                    raise RuntimeError(f"Ground-truth trajectory {index}, attempt {attempt}: {error}") from error
        for key in values:
            values[key].append(sample[key])
        attempts_used.append(attempt)
        if progress and ((index + 1) % 10 == 0 or index + 1 == n_trajectories):
            print(f"Ground truth: {index + 1}/{n_trajectories} trajectories", flush=True)
    arrays = {key: np.stack(value) for key, value in values.items()}
    arrays.update({key: sample[key] for key in ("times", "parameter", "phi")})
    arrays["simulation_attempts"] = np.asarray(attempts_used, dtype=np.int64)
    return arrays, failures


def save_npz(path, arrays, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays, metadata_json=json.dumps(metadata, allow_nan=False))
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as stored:
        return ({key: stored[key] for key in stored.files if key != "metadata_json"},
                json.loads(str(stored["metadata_json"].item())))


def ground_truth(args, module, regime, regime_path):
    requested = cache_request(args, regime, regime_path)
    path = args.ground_truth
    if path.exists() and not args.regenerate_ground_truth:
        arrays, metadata = load_npz(path)
        if metadata.get("cache_version") != CACHE_VERSION or metadata.get("kind") != "paired_drift_ensemble":
            raise ValueError(f"Not a supported ensemble cache: {path}")
        # JSON converts tuples in regime definitions into lists.
        if metadata["request"] != json.loads(json.dumps(requested)):
            raise ValueError("Ground-truth cache settings or regime source differ. Choose another output/cache path or use --regenerate-ground-truth.")
        expected = (args.n_trajectories, args.steps, 3)
        if arrays["x"].shape != expected or arrays["observed"].shape != expected or not np.isfinite(arrays["observed"]).all():
            raise ValueError(f"Invalid or incomplete ensemble cache: {path}")
        print(f"Reusing ground truth: {path}", flush=True)
        return arrays, metadata
    if args.plot_only:
        raise FileNotFoundError(f"--plot-only requires existing ground truth: {path}")
    arrays, failures = simulate_ensemble(module, regime, requested["simulation"], args.n_trajectories,
                                       args.max_simulation_attempts, args.normalization_stats)
    metadata = dict(cache_version=CACHE_VERSION, kind="paired_drift_ensemble", request=requested,
                    regime_file=str(regime_path), failed_attempts=failures,
                    sample_convention="Index 0 is post burn-in. steps counts stored states, hence steps-1 model transitions. phi[t] drives x[t] -> x[t+1]; the ODE uses continuous drift.",
                    normalization_note="trajectory uses one affine transform per full recorded truth trajectory, including its future; the same transform is used for its paired model. No time-local rescaling or observation noise.")
    save_npz(path, arrays, metadata)
    return arrays, metadata


@torch.inference_mode()
def rollout_ensemble(model, initial, phi, device, batch_size=256):
    initial = np.asarray(initial, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    if initial.ndim != 2 or initial.shape[1] != model.N or not np.isfinite(initial).all():
        raise ValueError("Initial observations must be a finite (ensemble, observed_dim) array.")
    if phi.ndim != 1 or not phi.size or not np.isfinite(phi).all():
        raise ValueError("phi must be a finite, nonempty, common one-dimensional schedule.")
    output = np.full((len(initial), len(phi), model.N), np.nan, dtype=np.float32)
    failures = np.full(len(initial), -1, dtype=np.int64)
    inputs = torch.as_tensor(phi.copy(), device=device).reshape(-1, 1)
    for start in range(0, len(initial), batch_size):
        stop = min(start + batch_size, len(initial))
        x0 = torch.as_tensor(initial[start:stop].copy(), device=device)
        latent = x0 @ model.B
        latent[:, :model.N] = x0
        count = stop - start
        prediction = torch.full((count, len(phi), model.N), float("nan"), device=device)
        prediction[:, 0] = x0
        alive = torch.ones(count, dtype=torch.bool, device=device)
        failed_at = torch.full((count,), -1, dtype=torch.int64, device=device)
        for step in range(len(phi) - 1):
            value = inputs[step:step + 1].expand(count, -1) if model.phi_dim else None
            updated = model(latent, value)
            finite = torch.isfinite(updated).all(dim=1)
            failed_at = torch.where(alive & ~finite, step + 1, failed_at)
            alive = alive & finite
            latent = torch.where(alive[:, None], updated, 0.)
            prediction[:, step + 1] = torch.where(alive[:, None], latent[:, :model.N], float("nan"))
        output[start:stop] = prediction.cpu().numpy()
        failures[start:stop] = failed_at.cpu().numpy()
    return output, failures


def finite_sections(values):
    valid = np.isfinite(values)
    changes = np.diff(np.r_[False, valid, False].astype(np.int8))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def peak_records(values, times, parameter, phi, source, prominence=0., distance=1):
    """Pair successive peaks within one trajectory AND one finite segment."""
    records = []
    for trajectory, row in enumerate(np.asarray(values)):
        for start, stop in finite_sections(row):
            indices, _ = find_peaks(row[start:stop], distance=distance,
                                    prominence=prominence if prominence > 0 else None)
            indices = indices + start
            for position, index in enumerate(indices):
                following = int(indices[position + 1]) if position + 1 < len(indices) else None
                records.append(dict(source=source, trajectory=int(trajectory), step=int(index),
                                    time=float(times[index]), parameter=float(parameter[index]), phi=float(phi[index]),
                                    value=float(row[index]), next_peak_step=following,
                                    interval_steps=following - int(index) if following is not None else None,
                                    interval_time=float(times[following] - times[index]) if following is not None else None))
    return records


PEAK_FIELDS = ("source", "trajectory", "step", "time", "parameter", "phi", "value",
               "next_peak_step", "interval_steps", "interval_time")


def save_peak_table(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PEAK_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def physical_prediction(prediction, arrays):
    return prediction.astype(np.float64) * arrays["std"][:, None, :] + arrays["mean"][:, None, :]


def save_plots(output, arrays, models, component="z", space="physical", prominence=0., distance=1, plotly_js=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    channel = "xyz".index(component)
    physical = space == "physical"
    truth = arrays["observed"] if physical else arrays["x"]
    series = [("Ground truth", truth, "#2464a0")]
    for index, model in enumerate(models):
        prediction = physical_prediction(model["prediction"], arrays) if physical else model["prediction"]
        series.append((model["label"], prediction, f"C{(index + 1) % 10}"))
    times, parameter, phi = (arrays[key] for key in ("times", "parameter", "phi"))
    xlabel = f"{arrays.get('parameter_name', 'parameter')} (t)"
    unit = "physical" if physical else "standardized"
    ylabels = (f"{component} ({unit})", f"Local maxima of {component} ({unit})", "Time to next maximum Δt")
    filenames = (f"ensemble_{component}.png", f"maxima_{component}.png", f"peak_intervals_{component}.png")
    all_records, records_by_source = [], []
    for source_index, (label, values, color) in enumerate(series):
        source = "ground_truth" if source_index == 0 else f"model_{source_index:02d}"
        records = peak_records(values[:, :, channel], times, parameter, phi, source, prominence, distance)
        records_by_source.append(records)
        all_records.extend(records)
    for kind, (filename, ylabel) in enumerate(zip(filenames, ylabels)):
        figure, axes = plt.subplots(len(series), 1, figsize=(12, 3.5 * len(series)),
                                    sharex=True, sharey=True, squeeze=False, layout="constrained")
        for axis, (label, values, color), records in zip(axes[:, 0], series, records_by_source):
            if kind == 0:
                x = np.tile(parameter, len(values))
                y = values[:, :, channel].reshape(-1)
                point_size, alpha = 0.4, 0.16
            else:
                kept = records if kind == 1 else [r for r in records if r["interval_time"] is not None]
                x = [r["parameter"] for r in kept]
                y = [r["value" if kind == 1 else "interval_time"] for r in kept]
                point_size, alpha = 5., 0.65
            axis.scatter(x, y, s=point_size, alpha=alpha, c=color, linewidths=0, rasterized=True)
            if not np.isfinite(y).any():
                axis.text(0.5, 0.5, "No finite samples / no detected maxima or intervals", ha="center", transform=axis.transAxes)
            axis.set_title(label, loc="left")
            axis.set_ylabel(ylabel)
            if kind == 2:
                axis.set_ylim(bottom=0)
            axis.grid(alpha=0.15)
        axes[-1, 0].set_xlabel(xlabel)
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)
    save_peak_table(output / f"maxima_{component}.csv", all_records)

    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout="constrained")
    for axis, dim in zip(axes[:3], range(3)):
        for label, values, color in series:
            axis.plot(times, values[0, :, dim], label=label, color=color, linewidth=0.85, alpha=0.9)
        axis.set_ylabel(f"{'xyz'[dim]} ({unit})")
        axis.grid(alpha=0.15)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("First paired trajectory: same post-burn-in initial observation")
    axes[3].plot(times, parameter, color="black")
    axes[3].set_ylabel(xlabel)
    axes[3].set_xlabel("Time t")
    figure.savefig(output / "first_trajectory_components.png", dpi=160)
    plt.close(figure)

    figure = plt.figure(figsize=(6 * len(series), 5.3), layout="constrained")
    norm = Normalize(float(times[0]), float(times[-1]))
    limits = []
    for dim in range(3):
        finite = np.concatenate([v[0, :, dim][np.isfinite(v[0, :, dim])] for _, v, _ in series])
        lower, upper = float(finite.min()), float(finite.max())
        padding = max((upper - lower) * 0.04, 1e-3)
        limits.append((lower - padding, upper + padding))
    panels = []
    traces = []
    layout = dict(title="First paired trajectory; color encodes time", showlegend=False)
    for index, (label, values, color) in enumerate(series):
        axis = figure.add_subplot(1, len(series), index + 1, projection="3d")
        first = values[0]
        segments, segment_times = single.colored_segments(first, times)
        collection = Line3DCollection(segments, cmap="viridis", norm=norm, linewidth=0.8)
        collection.set_array(segment_times)
        axis.add_collection3d(collection)
        axis.scatter(*first[0], s=15, c="black")
        axis.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel="x", ylabel="y", zlabel="z", title=label)
        axis.view_init(elev=25, azim=-60)
        axis.set_box_aspect([hi - lo for lo, hi in limits])
        panels.append(axis)
        scene = "scene" if index == 0 else f"scene{index + 1}"
        traces.append(dict(type="scatter3d", mode="lines", name=label, scene=scene,
                           x=single.plot_values(first[:, 0]), y=single.plot_values(first[:, 1]), z=single.plot_values(first[:, 2]),
                           connectgaps=False, line=dict(color=times.tolist(), colorscale="Viridis", cmin=float(times[0]), cmax=float(times[-1]), width=3, showscale=index == 0, colorbar=dict(title="Time t"))))
        layout[scene] = dict(domain=dict(x=[index / len(series) + 0.02, (index + 1) / len(series) - 0.02]),
                             xaxis=dict(title="x", range=limits[0]), yaxis=dict(title="y", range=limits[1]), zaxis=dict(title="z", range=limits[2]),
                             aspectmode="data")
    figure.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=panels, label="Time t", shrink=0.7)
    figure.savefig(output / "first_trajectory_3d.png", dpi=160)
    plt.close(figure)
    layout["annotations"] = [dict(text=label, x=(index + .5) / len(series), y=1.02, xref="paper", yref="paper", showarrow=False) for index, (label, _, _) in enumerate(series)]
    single.write_plot(output / "first_trajectory_3d.html", "First paired trajectory", traces, layout, plotly_js)
    return [dict(source="ground_truth" if index == 0 else f"model_{index:02d}",
                 maxima=len(records), intervals=sum(r["interval_time"] is not None for r in records)) for index, records in enumerate(records_by_source)]


def save_provenance(output, analysis):
    """Keep the identities/settings behind older component plots when replotting."""
    path = output / "ensemble_sources.json"
    manifest = dict(schema_version=1, analyses={}, plots={})
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or "analyses" not in manifest or "plots" not in manifest:
            raise ValueError(f"Unrecognized provenance format: {path}")
    identifier = hashlib.sha256(json.dumps(analysis, sort_keys=True).encode("utf-8")).hexdigest()
    manifest["analyses"][identifier] = analysis
    component = analysis["component"]
    files = [f"ensemble_{component}.png", f"maxima_{component}.png", f"peak_intervals_{component}.png",
             f"maxima_{component}.csv", "first_trajectory_components.png", "first_trajectory_3d.png", "first_trajectory_3d.html"]
    for filename in files:
        manifest["plots"][filename] = dict(analysis_id=identifier)
    referenced = {entry["analysis_id"] for entry in manifest["plots"].values()}
    manifest["analyses"] = {key: value for key, value in manifest["analyses"].items() if key in referenced}
    manifest["latest_analysis_id"] = identifier
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    module, regime_path = single.load_regime_module(args.regime_file)
    regime = single.resolve_regime(module, args.regime)
    spec = simulation_spec(args, regime)
    if args.dry_run:
        print(json.dumps(dict(simulation=spec, n_trajectories=args.n_trajectories,
                              model_runs=[str(p) for p in args.model_runs], output_dir=str(args.output_dir),
                              ground_truth=str(args.ground_truth), device=args.device, gpu_id=args.gpu_id), indent=2))
        return 0
    torch.set_num_threads(args.cpu_threads)
    device = "cpu"
    if not args.plot_only and not args.ground_truth_only and args.device == "cuda":
        if not torch.cuda.is_available() or args.gpu_id >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device unavailable. Use --device cpu for a CPU run.")
        device = f"cuda:{args.gpu_id}"
    if not args.plot_only and not args.ground_truth_only:
        for run in args.model_runs:
            for filename in ("model.pt", "metrics.json"):
                if not (run / filename).is_file():
                    raise FileNotFoundError(f"Missing model file: {run / filename}")
    print(f"{regime.name}: {args.n_trajectories} paired trajectories; {args.burn_in_steps} fixed burn-in steps, then {args.steps} states; raw {spec['parameter_start']} -> {spec['parameter_end']}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, truth_metadata = ground_truth(args, module, regime, regime_path)
    if args.ground_truth_only:
        print(f"Ground-truth cache: {args.ground_truth}", flush=True)
        return 0
    truth_hash = single.file_hash(args.ground_truth)
    models = []
    labels = args.model_labels or [run.name for run in args.model_runs]
    for index, (run, label) in enumerate(zip(args.model_runs, labels), start=1):
        stem = single.model_file_stem(index, run.name)
        prediction_path = args.output_dir / f"{stem}.npz"
        if args.plot_only:
            saved, info = load_npz(prediction_path)
            if info.get("ground_truth_sha256") != truth_hash or info.get("run_dir") != str(run):
                raise ValueError(f"Prediction cache belongs to different ground truth or model path: {prediction_path}")
            prediction, failed_at = saved["prediction"], saved["failed_at"]
            if prediction.shape != arrays["x"].shape or failed_at.shape != (args.n_trajectories,):
                raise ValueError(f"Invalid prediction cache shape: {prediction_path}")
        else:
            model, info = single.load_model(run, device)
            print(f"Rolling out {label} on {device} ({args.n_trajectories} paired initial states)", flush=True)
            prediction, failed_at = rollout_ensemble(model, arrays["x"][:, 0], arrays["phi"], device, args.rollout_batch_size)
            info.update(ground_truth_sha256=truth_hash, metrics_sha256=single.file_hash(run / "metrics.json"),
                        initialization="latent = x0 @ B; first N latent coordinates overwritten by x0; no context or teacher forcing")
            save_npz(prediction_path, dict(prediction=prediction, failed_at=failed_at), info)
            del model
        info = dict(info, label=label, prediction_file=str(prediction_path),
                    nonfinite_trajectories=int(np.sum(failed_at >= 0)))
        models.append(dict(label=label, prediction=prediction, metadata=info))
        print(f"{label}: {info['nonfinite_trajectories']} non-finite rollouts", flush=True)
    arrays["parameter_name"] = regime.parameter
    counts = save_plots(args.output_dir, arrays, models, args.component, args.plot_space,
                        args.peak_prominence, args.peak_distance_steps, args.plotly_js)
    manifest = dict(ground_truth_file=str(args.ground_truth), ground_truth_sha256=truth_hash,
                    ground_truth=truth_metadata, models=[m["metadata"] for m in models], peak_counts=counts,
                    plot_space=args.plot_space, component=args.component, peak_prominence=args.peak_prominence,
                    peak_distance_steps=args.peak_distance_steps,
                    peak_method="scipy.signal.find_peaks on each finite segment of each trajectory; plateaus use the middle sample; no smoothing. Endpoints excluded. Intervals use the earlier peak's parameter; last peaks have no interval. Never pair across trajectories or non-finite gaps.",
                    script_sha256=single.file_hash(Path(__file__)), helper_sha256=single.file_hash(Path(single.__file__)),
                    plotting_note="All ensemble samples are plotted without subsampling. Shared axes compare separate truth/model panels. 3D colors encode time, not parameter.")
    save_provenance(args.output_dir, manifest)
    print(f"Plots, maxima table, arrays and provenance: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
