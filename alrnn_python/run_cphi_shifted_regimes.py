"""Train Dadras and Rossler with additive + shifted-ReLU inputs on a GPU queue."""

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import sys

from run_term_ablations import build_steps, load_groups, schedule, validate_gpu_ids


REGIMES = ("dadras_e_49_to_64", "rossler_a_005_to_035")
VARIANT = "cphi_shifted_relu"
TERMS = ("additive", "shifted-relu")
SUBSETS = ("all", "negative", "positive")
MAX_GPUS = 6


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data-root", type=Path, required=True)
    parser.add_argument("--backend-script", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--latent-dim", type=int, default=200)
    parser.add_argument("--pwl-units", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    validate_gpu_ids(args.gpu_ids, max_gpus=MAX_GPUS)
    for name in ("latent_dim", "epochs", "batch_size", "batches_per_epoch", "cpu_threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if not 0 <= args.pwl_units <= args.latent_dim:
        parser.error("--pwl-units must lie between zero and --latent-dim.")
    args.training_data_root = args.training_data_root.expanduser().resolve()
    args.backend_script = args.backend_script.expanduser().resolve()
    return args


def prepare_experiments(args, script_dir):
    required = [
        script_dir / "train_halvorsen_plain.py",
        script_dir / "compare_model_bifurcation_diagrams.py",
        args.backend_script,
    ]
    experiments = {}
    for regime in REGIMES:
        config = copy.copy(args)
        config.regime = regime
        config.data_dir = args.training_data_root / regime
        if not (config.data_dir / "trajectories.npy").is_file():
            if (config.data_dir / "values_8").is_dir():
                config.data_dir /= "values_8"
        for name in ("trajectories.npy", "trajectory_phi.npy", "trajectory_phi_constants.npy", "metadata.json"):
            required.append(config.data_dir / name)
        experiments[regime] = config
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")
    return {
        regime: (config, load_groups(config.data_dir, regime))
        for regime, config in experiments.items()
    }


def job_definitions():
    # Interleave systems so the first two GPUs start one training for each DS.
    trainings = {
        f"{regime.split('_', 1)[0]}_{subset}": (regime, index)
        for index, subset in enumerate(SUBSETS)
        for regime in REGIMES
    }
    comparisons = {f"{regime.split('_', 1)[0]}_bifurcation": regime for regime in REGIMES}
    return trainings, comparisons


def run_job_phases(gpu_file, gpu_ids, device_count, log_dir, make_steps, env,
                   trainings, comparisons, poll_interval=2):
    results = schedule(
        gpu_file, gpu_ids, device_count, log_dir, make_steps, env,
        poll_interval=poll_interval, job_names=list(trainings), max_gpus=MAX_GPUS,
    )
    eligible = []
    skipped = []
    for name, regime in comparisons.items():
        dependencies = [job for job, (system, _) in trainings.items() if system == regime]
        if all(results[job] == 0 for job in dependencies):
            eligible.append(name)
        else:
            print(f"Skipping {name}: at least one training for this system failed.", flush=True)
            skipped.append(name)
            results[name] = 1
    if eligible:
        print("Training phase finished. Starting bifurcation comparisons.", flush=True)
        results.update(schedule(
            gpu_file, gpu_ids, device_count, log_dir, make_steps, env,
            poll_interval=poll_interval, job_names=eligible, max_gpus=MAX_GPUS,
        ))
    return results, skipped


def main(argv=None):
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    experiments = prepare_experiments(args, script_dir)
    for regime, (_, groups) in experiments.items():
        for subset, values in groups.items():
            print(f"{regime} | {subset} ({len(values)} phi values): {values}", flush=True)

    batch_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_m{args.latent_dim}_p{args.pwl_units}"
    run_roots = {
        regime: script_dir / "runs/term_ablations" / regime / VARIANT / batch_id
        for regime in REGIMES
    }
    plot_roots = {
        regime: script_dir / "bifurcation_diagrams/term_ablations" / regime / VARIANT / batch_id
        for regime in REGIMES
    }
    queue_root = script_dir / "runs/term_ablations/_queues" / VARIANT / batch_id
    trainings, comparisons = job_definitions()

    def make_steps(job, gpu):
        if job in trainings:
            regime, index = trainings[job]
        else:
            regime, index = comparisons[job], -1
        config, groups = experiments[regime]
        steps = build_steps(
            config, script_dir, groups, run_roots[regime], plot_roots[regime],
            VARIANT, gpu, terms=TERMS, variant_subdir=False,
        )
        return [steps[index]]

    if args.dry_run:
        print("DRY RUN: six trainings and two comparisons; no CUDA or output files.")
        for index, job in enumerate([*trainings, *comparisons]):
            gpu = args.gpu_ids[index % len(args.gpu_ids)]
            step = make_steps(job, gpu)[0]
            print(f"\n[{job}] GPU {gpu}: {step.label}\ncwd: {step.cwd}\n{shlex.join(step.command)}")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected Python environment.")
    device_count = torch.cuda.device_count()
    validate_gpu_ids(args.gpu_ids, device_count, max_gpus=MAX_GPUS)
    for gpu in args.gpu_ids:
        print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)}", flush=True)

    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.cpu_threads)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    log_dir = queue_root / "logs"
    log_dir.mkdir(parents=True)
    for job in [*trainings, *comparisons]:
        (log_dir / f"{job}.log").touch()
    gpu_file = queue_root / "gpu_ids.json"
    gpu_file.write_text(json.dumps(args.gpu_ids) + "\n", encoding="utf-8")
    plan = {
        "args": vars(args), "phi_terms": TERMS,
        "experiments": {
            regime: {"data_dir": config.data_dir, "groups": groups}
            for regime, (config, groups) in experiments.items()
        },
        "training_jobs": trainings, "comparison_jobs": comparisons,
        "run_roots": run_roots, "plot_roots": plot_roots, "queue_root": queue_root,
    }
    (queue_root / "plan.json").write_text(json.dumps(plan, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nTerms: {TERMS}", flush=True)
    for regime in REGIMES:
        print(f"Models [{regime}]: {run_roots[regime]}", flush=True)
        print(f"Diagrams/cache [{regime}]: {plot_roots[regime]}", flush=True)
    print(f"Shared queue/logs: {queue_root}", flush=True)
    print(f"Live GPU list: {gpu_file}", flush=True)
    print(f"Progress: tail -f {shlex.quote(str(log_dir))}/*.log", flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        results, skipped = run_job_phases(
            gpu_file, args.gpu_ids, device_count, log_dir, make_steps, env,
            trainings, comparisons,
        )
    except KeyboardInterrupt:
        print("Stopped: child processes terminated; completed runs remain on disk.", flush=True)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    summary = {"exit_codes": results, "skipped_comparisons": skipped}
    (queue_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    failed = [job for job, code in results.items() if code != 0]
    if failed:
        print(f"Failed/skipped jobs: {', '.join(failed)}. Logs: {log_dir}", flush=True)
        return 1
    print("Finished six trainings and two bifurcation comparisons. Diagram paths are listed above.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        sys.exit(str(exc))
