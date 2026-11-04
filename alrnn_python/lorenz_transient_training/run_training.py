"""Queue the isolated Lorenz transient experiment, then bifurcations and orbits."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

from run_dadras_combined_balanced import variants, training_step, job_definitions
from run_term_ablations import Step, schedule, validate_gpu_ids
from lorenz_paper_test import evaluate_lorenz as paper

DEFAULT_DATA = Path("/export/home/klenkeit/training_data/mixed_parameter_trajectories/lorenz_rho/lorenz_transients_two_initials")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-root", type=Path, help="New run folder, or existing run with --analysis-only.")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--time-sampling-bins", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cphi-latent-dim", type=int, default=300)
    parser.add_argument("--cphi-pwl-units", type=int, default=150)
    parser.add_argument("--shifted-latent-dim", type=int, default=200)
    parser.add_argument("--shifted-pwl-units", type=int, default=100)
    parser.add_argument("--num-parameters", type=int, default=290)
    parser.add_argument("--warmup-time", type=float, default=200.)
    parser.add_argument("--observation-time", type=float, default=50.)
    parser.add_argument("--orbit-duration", type=float, default=50.)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.system = "lorenz_rho"
    validate_gpu_ids(args.gpu_ids, max_gpus=6)
    for key in ("epochs", "batch_size", "batches_per_epoch", "sequence_length", "time_sampling_bins", "cpu_threads"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive.")
    if args.num_parameters < 2:
        parser.error("num_parameters must be >=2.")
    if not math.isfinite(args.orbit_duration) or args.orbit_duration <= 0:
        parser.error("orbit_duration must be finite and positive.")
    for prefix in ("cphi", "shifted"):
        if not 0 <= getattr(args, prefix+"_pwl_units") <= getattr(args, prefix+"_latent_dim") or getattr(args, prefix+"_latent_dim") < 3:
            parser.error(f"Invalid dimensions for {prefix}.")
    if args.analysis_only and args.run_root is None:
        parser.error("--analysis-only requires --run-root.")
    args.training_data_root = args.training_data_root.expanduser().resolve()
    args.run_root = (args.run_root or HERE / "results" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}").expanduser().resolve()
    return args


def analysis_args(args, gpu=None, variant=None):
    result = ["--run-root", str(args.run_root / "models"),
              "--training-data-root", str(args.training_data_root),
              "--output-root", str(args.run_root / "analysis/bifurcations"),
              "--device", args.device, "--gpu-ids", str(args.gpu_ids[0] if gpu is None else gpu),
              "--num-parameters", str(args.num_parameters), "--warmup-time", str(args.warmup_time),
              "--observation-time", str(args.observation_time), "--cpu-threads", str(args.cpu_threads)]
    if variant:
        result += ["--variant", variant]
    return result


def make_steps(args, model_variants, trainings, job, gpu):
    if job in trainings:
        variant, dataset = trainings[job]
        step = training_step(args, ALRNN_DIR, args.run_root / "models", variant, model_variants[variant], dataset, gpu)
        # The tiny tail only keeps the existing dataset contract intact. It is
        # deliberately skipped for evaluation; independent free orbits follow.
        step.command.extend(["--t-transient", "1000"])
        if args.device == "cpu":
            step.command.append("--force-cpu")
        return [step]
    variant = job.removesuffix("__analysis")
    return [Step(f"two-IC Lorenz bifurcations: {variant}",
                 [sys.executable, "-u", str(ALRNN_DIR / "lorenz_paper_test/evaluate_lorenz.py"),
                  *analysis_args(args, gpu, variant)], HERE),
            Step(f"free transient/late trajectory comparisons: {variant}",
                 [sys.executable, "-u", str(HERE / "plot_trajectories.py"),
                  "--run-root", str(args.run_root / "models"), "--training-data-root", str(args.training_data_root),
                  "--output-root", str(args.run_root / "analysis/trajectories"), "--variant", variant,
                  "--device", args.device, "--gpu-id", str(gpu), "--cpu-threads", str(args.cpu_threads),
                  "--duration", str(args.orbit_duration)], HERE)]


def validate_data(args):
    import numpy as np
    for name in paper.DATASETS:
        folder = args.training_data_root / name
        for stem in ("trajectories", "trajectory_phi", "trajectory_phi_raw", "trajectory_lengths",
                     "trajectory_train_lengths", "trajectory_mean", "trajectory_std",
                     "trajectory_modes", "trajectory_group_names", "trajectory_initial_state_index",
                     "trajectory_initial_states"):
            if not (folder / f"{stem}.npy").is_file():
                raise FileNotFoundError(f"Missing required dataset file: {folder / (stem+'.npy')}")
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
        policy = metadata.get("initial_state_policy", {})
        if policy.get("kind") != "exact_paired" or policy.get("initial_states") != paper.INITIAL_STATES.tolist():
            raise ValueError(f"Expected the new exactly paired Lorenz dataset, not the old noisy dataset: {folder}")
        config = metadata["config"]
        if config["noise_std_fraction"] != 0 or any(spec["burn_in_steps"] != 0 for spec in config["modes"]):
            raise ValueError(f"This experiment requires noise=0 and burn-in=0: {folder}")
        lengths = np.load(folder / "trajectory_lengths.npy", allow_pickle=False)
        train = np.load(folder / "trajectory_train_lengths.npy", allow_pickle=False)
        initial = np.load(folder / "trajectory_initial_state_index.npy", allow_pickle=False)
        if not np.array_equal(np.bincount(initial, minlength=2), [len(lengths)//2]*2):
            raise ValueError(f"Initial states must be split 50/50: {folder}")
        if np.any(train <= args.sequence_length) or np.any(lengths-train < 2):
            raise ValueError(f"Data must retain sequence_length+1 training samples and two test samples: {folder}")
        print(f"Dataset: {folder}; {len(lengths)} trajectories; "
              f"train/test={train.min()}..{train.max()}/{(lengths-train).min()}..{(lengths-train).max()}; ICs 50/50", flush=True)
    check = paper.parse_args(analysis_args(args))
    scales = paper.preflight(check)
    if not math.isclose(round(args.orbit_duration/check.dt)*check.dt, args.orbit_duration, abs_tol=1e-8) or round(args.orbit_duration/check.dt) < 2:
        raise ValueError("orbit_duration must be a multiple of the data dt and include at least two steps.")
    return scales


def main(argv=None):
    args = parse_args(argv)
    model_variants = variants(args)
    trainings, _ = job_definitions(model_variants, dict.fromkeys(paper.DATASETS))
    analyses = [f"{variant}__analysis" for variant in model_variants]
    # Validate analysis times even for dry runs, before any expensive training.
    paper.parse_args(analysis_args(args))
    if args.dry_run:
        jobs = analyses if args.analysis_only else [*trainings, *analyses]
        print(f"DRY RUN: {args.run_root}; no data, CUDA or output files required.")
        for index, job in enumerate(jobs):
            for step in make_steps(args, model_variants, trainings, job, args.gpu_ids[index % len(args.gpu_ids)]):
                print(f"[{job}] {step.label}\n{shlex.join(step.command)}")
        print("Before analysis: shared ground truth is computed once using both exact initial states.")
        return 0

    import torch
    torch.set_num_threads(args.cpu_threads)
    scales = validate_data(args)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. --device cpu is intended for small local checks.")
        device_count = torch.cuda.device_count()
        validate_gpu_ids(args.gpu_ids, device_count, max_gpus=6)
    else:
        device_count = None
        if args.gpu_ids != [0]:
            raise ValueError("CPU smoke checks use --gpu-ids 0 (one serial worker).")
    if not args.analysis_only and args.run_root.exists() and any(args.run_root.iterdir()):
        raise ValueError(f"New training needs a new or empty run folder: {args.run_root}")
    if args.analysis_only and not (args.run_root / "plan.json").is_file():
        raise FileNotFoundError(f"Expected an existing experiment plan: {args.run_root}")
    log_dir = args.run_root / "logs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}"
    log_dir.mkdir(parents=True)
    gpu_file = args.run_root / "gpu_ids.json"
    gpu_file.write_text(json.dumps(args.gpu_ids)+"\n", encoding="utf-8")
    if not args.analysis_only:
        plan = dict(args=vars(args), variants={name: asdict(value) for name, value in model_variants.items()},
                    training_jobs=trainings, analysis_jobs=analyses, initial_states=paper.INITIAL_STATES.tolist())
        (args.run_root / "plan.json").write_text(json.dumps(plan, indent=2, default=str)+"\n", encoding="utf-8")
    print(f"Run: {args.run_root}\nProgress: tail -f {shlex.quote(str(log_dir))}/*.log\nGPU queue: {gpu_file}", flush=True)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.cpu_threads)
    def steps(job, gpu):
        return make_steps(args, model_variants, trainings, job, gpu)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    results = {}
    try:
        if not args.analysis_only:
            results.update(schedule(gpu_file, args.gpu_ids, device_count, log_dir, steps, env,
                                    job_names=list(trainings), max_gpus=6))
        if not any(results.values()):
            for variant in model_variants:
                for dataset in paper.DATASETS:
                    for filename in ("model.pt", "metrics.json"):
                        path = args.run_root / "models" / variant / dataset / filename
                        if not path.is_file():
                            raise FileNotFoundError(f"Missing trained model: {path}")
            truth_args = paper.parse_args(analysis_args(args))
            paper.preflight(truth_args)
            print("Preparing shared Lorenz ground truth for bifurcations.", flush=True)
            paper.ground_truth(truth_args, scales[0], paper.settings(truth_args, scales[0]))
            results.update(schedule(gpu_file, args.gpu_ids, device_count, log_dir, steps, env,
                                    job_names=analyses, max_gpus=6))
        else:
            results.update({name: 1 for name in analyses})
            print("Analysis skipped because a training failed. Completed models are preserved.", flush=True)
    except KeyboardInterrupt:
        print("Stopped; completed models are preserved.", flush=True)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
    (log_dir / "summary.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print(f"Results: {args.run_root}; failed/skipped: {[job for job, code in results.items() if code]}", flush=True)
    return int(any(results.values()))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
