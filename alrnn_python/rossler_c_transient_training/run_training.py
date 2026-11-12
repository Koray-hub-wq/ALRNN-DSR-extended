"""Train Cphi with fixed population window proportions, then full-range analyses."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import signal
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ALRNN_DIR = HERE.parent
sys.path.insert(0, str(ALRNN_DIR))

from compare_model_bifurcation_diagrams import find_backend
from run_dadras_combined_balanced import Variant, training_step, comparison_step, prepare_data
from run_term_ablations import Step, schedule, validate_gpu_ids

DATASETS = ("rossler_c_4p5_to_5p5", "rossler_c_3_to_6")
DEFAULT_DATA = Path("/export/home/klenkeit/training_data/mixed_parameter_trajectories/rossler_c/rossler_c_transients_50_50")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--backend-script", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=300)
    parser.add_argument("--pwl-units", type=int, default=150)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tail-evaluations", type=int, choices=range(5), default=4)
    parser.add_argument("--attractor-fraction", type=float, default=.5)
    parser.add_argument("--num-parameters", type=int, default=320)
    parser.add_argument("--bifurcation-burn-in-steps", type=int, default=5000)
    parser.add_argument("--bifurcation-context-steps", type=int, default=500)
    parser.add_argument("--bifurcation-forecast-steps", type=int, default=15000)
    parser.add_argument("--orbit-duration", type=float, default=50.)
    parser.add_argument("--orbit-parameters", type=float, nargs="+", default=[3., 4., 4.5, 5., 5.5, 6.])
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    validate_gpu_ids(args.gpu_ids, max_gpus=6)
    if not math.isfinite(args.attractor_fraction) or not 0 < args.attractor_fraction < 1:
        parser.error("attractor-fraction must lie strictly between zero and one.")
    attractor_count = args.batch_size*args.attractor_fraction
    if args.batch_size < 2 or not math.isclose(attractor_count, round(attractor_count), abs_tol=1e-8, rel_tol=0.) or not 0 < round(attractor_count) < args.batch_size:
        parser.error("batch-size must give integer nonzero counts for both populations (half/half requires even batches).")
    for key in ("epochs", "batches_per_epoch", "sequence_length", "cpu_threads", "bifurcation_context_steps"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive.")
    if args.latent_dim < 3 or not 0 <= args.pwl_units <= args.latent_dim:
        parser.error("Invalid latent/PWL dimensions.")
    if args.num_parameters < 2 or args.bifurcation_forecast_steps < 3 or args.bifurcation_burn_in_steps < 0:
        parser.error("Need >=2 parameter values, >=3 forecast steps and nonnegative burn-in.")
    if not math.isfinite(args.orbit_duration) or args.orbit_duration <= 0 or not all(math.isfinite(c) and 3 <= c <= 6 for c in args.orbit_parameters):
        parser.error("Use a positive orbit duration and orbit parameters inside [3,6].")
    if args.analysis_only and args.run_root is None:
        parser.error("--analysis-only requires --run-root.")
    args.training_data_root = args.training_data_root.expanduser().resolve()
    args.run_root = (args.run_root or HERE / "results" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}").expanduser().resolve()
    if args.backend_script is not None:
        args.backend_script = args.backend_script.expanduser().resolve()
    args.system, args.time_sampling_bins = "rossler_c", 10
    args.parameter_min, args.parameter_max, args.components = 3., 6., ["z"]
    return args


def make_steps(args, job, gpu, dt):
    variant = Variant("Cphi", ("additive",), args.latent_dim, args.pwl_units)
    models = args.run_root / "models"
    if job.startswith("train__"):
        dataset = job.removeprefix("train__")
        step = training_step(args, ALRNN_DIR, models, "cphi", variant, dataset, gpu)
        step.command[2] = str(HERE / "train_cphi.py")
        step.command.extend(["--tail-evaluations", str(args.tail_evaluations)])
        step.command.extend(["--attractor-fraction", str(args.attractor_fraction)])
        if args.device == "cpu":
            step.command.append("--force-cpu")
        return [step]
    if job == "bifurcation":
        step = comparison_step(args, ALRNN_DIR, models, args.run_root / "analysis/bifurcations", "cphi", variant, gpu, dt)
        step.command[step.command.index("--device")+1] = args.device
        step.command.remove("--force-recompute")
        step.command.extend(["--burn-in-steps", str(args.bifurcation_burn_in_steps),
                             "--context-steps", str(args.bifurcation_context_steps),
                             "--forecast-steps", str(args.bifurcation_forecast_steps)])
        return [step]
    dataset = job.removeprefix("orbits__")
    return [Step(f"free Rossler-c comparisons: {dataset}",
            [sys.executable, "-u", str(HERE / "plot_trajectories.py"), "--run-root", str(models),
             "--training-data-root", str(args.training_data_root),
             "--output-root", str(args.run_root / "analysis/trajectories"), "--dataset", dataset,
             "--device", args.device, "--gpu-id", str(gpu), "--cpu-threads", str(args.cpu_threads),
             "--duration", str(args.orbit_duration), "--parameters", *map(str, args.orbit_parameters)], HERE)]


def preflight(args):
    import numpy as np
    args.backend_script = find_backend(args.backend_script)
    dt = prepare_data(args, ALRNN_DIR)
    for name in DATASETS:
        folder = args.training_data_root / name
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
        if metadata.get("experiment") not in ("rossler_c_transients_50_50_v1", "rossler_c_transient_populations_v1"):
            raise ValueError(f"Expected the new population-labeled experiment data: {folder}")
        labels = np.load(folder / "trajectory_population.npy", allow_pickle=False)
        lengths = np.load(folder / "trajectory_lengths.npy", allow_pickle=False)
        train = np.load(folder / "trajectory_train_lengths.npy", allow_pickle=False)
        if labels.shape != lengths.shape or set(labels) != {"attractor", "transient"} or labels[0] != "attractor":
            raise ValueError(f"Expected both populations and an attractor trajectory first: {folder}")
        if np.any(lengths-train < 2):
            raise ValueError(f"Retain at least two test samples: {folder}")
        for key in ("trajectory_phi", "trajectory_phi_raw", "trajectory_modes"):
            if not (folder / f"{key}.npy").is_file():
                raise FileNotFoundError(folder / f"{key}.npy")
    if not math.isclose(round(args.orbit_duration/dt)*dt, args.orbit_duration, abs_tol=1e-8) or round(args.orbit_duration/dt) < 2:
        raise ValueError("orbit-duration must be a multiple of dt and contain at least two steps.")
    return dt


def main(argv=None):
    args = parse_args(argv)
    trainings = [f"train__{name}" for name in DATASETS]
    analyses = ["bifurcation", *(f"orbits__{name}" for name in DATASETS)]
    if args.dry_run:
        args.backend_script = args.backend_script or Path.home() / "diffrentbutconst_prams/compare_model_bifurcation_diagrams.py"
        print(f"DRY RUN: {args.run_root}; no files written or GPUs required.")
        for i, job in enumerate(analyses if args.analysis_only else trainings+analyses):
            for step in make_steps(args, job, args.gpu_ids[i % len(args.gpu_ids)], .01):
                print(f"[{job}] {step.label}\n{shlex.join(step.command)}")
        return 0
    import torch
    dt = preflight(args)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; --device cpu is for small local checks.")
        device_count = torch.cuda.device_count()
        validate_gpu_ids(args.gpu_ids, device_count, max_gpus=6)
    else:
        device_count = None
        if args.gpu_ids != [0]:
            raise ValueError("CPU checks use one serial worker: --gpu-ids 0.")
    if not args.analysis_only and args.run_root.exists() and any(args.run_root.iterdir()):
        raise ValueError(f"New training requires a new or empty run folder: {args.run_root}")
    if args.analysis_only and not (args.run_root / "plan.json").is_file():
        raise FileNotFoundError(f"Missing original experiment plan: {args.run_root}")
    log_dir = args.run_root / "logs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}"
    log_dir.mkdir(parents=True)
    gpu_file = args.run_root / "gpu_ids.json"
    gpu_file.write_text(json.dumps(args.gpu_ids)+"\n", encoding="utf-8")
    if not args.analysis_only:
        (args.run_root / "plan.json").write_text(json.dumps(dict(args=vars(args), training_jobs=trainings,
            analysis_jobs=analyses, bifurcation_dt=dt, sampler="fixed population proportions per batch",
            population_windows_per_batch=dict(attractor=round(args.batch_size*args.attractor_fraction),
                transient=args.batch_size-round(args.batch_size*args.attractor_fraction))), indent=2, default=str)+"\n", encoding="utf-8")
    print(f"Models and analyses: {args.run_root}\nProgress: tail -f {shlex.quote(str(log_dir))}/*.log\nGPU queue: {gpu_file}", flush=True)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.cpu_threads)
    def steps(job, gpu):
        return make_steps(args, job, gpu, dt)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    results = {}
    try:
        if not args.analysis_only:
            results.update(schedule(gpu_file, args.gpu_ids, device_count, log_dir, steps, env, job_names=trainings, max_gpus=6))
        if not any(results.values()):
            for dataset in DATASETS:
                for filename in ("model.pt", "metrics.json"):
                    path = args.run_root / "models/cphi" / dataset / filename
                    if not path.is_file():
                        raise FileNotFoundError(path)
            results.update(schedule(gpu_file, args.gpu_ids, device_count, log_dir, steps, env, job_names=analyses, max_gpus=6))
        else:
            results.update({job: 1 for job in analyses})
            print("Analysis skipped: a required training failed. Completed models are preserved.", flush=True)
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
    (log_dir / "summary.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    failed = [job for job, code in results.items() if code]
    print(f"Finished; failed/skipped jobs: {failed}; results: {args.run_root}", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
