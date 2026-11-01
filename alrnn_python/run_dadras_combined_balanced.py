"""Queue six balanced mixed-Dadras trainings and two bifurcation comparisons."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import signal
import sys

from run_term_ablations import Step, schedule, validate_gpu_ids


MAX_GPUS = 6
REGIME = "dadras_e_49_to_64"
DATASETS = {
    "dadras_e_5p9_to_6p2": "e in [5.9, 6.2]",
    "dadras_e_5p2_to_5p7": "e in [5.2, 5.7]",
    "dadras_e_5p2_to_6p2": "e in [5.2, 6.2]",
}


@dataclass(frozen=True)
class Variant:
    label: str
    terms: tuple
    latent_dim: int
    pwl_units: int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data-root", type=Path, required=True)
    parser.add_argument("--backend-script", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--cphi-latent-dim", type=int, default=300)
    parser.add_argument("--cphi-pwl-units", type=int, default=150)
    parser.add_argument("--shifted-latent-dim", type=int, default=200)
    parser.add_argument("--shifted-pwl-units", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--time-sampling-bins", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--parameter-min", type=float, default=4.9)
    parser.add_argument("--parameter-max", type=float, default=6.2)
    parser.add_argument("--num-parameters", type=int, default=320)
    parser.add_argument("--components", nargs="+", choices=("x", "y", "z"), default=["z"])
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--plot-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    validate_gpu_ids(args.gpu_ids, max_gpus=MAX_GPUS)
    for name in ("epochs", "batch_size", "batches_per_epoch", "sequence_length",
                 "time_sampling_bins", "cpu_threads", "num_parameters"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for prefix in ("cphi", "shifted"):
        latent, pwl = getattr(args, f"{prefix}_latent_dim"), getattr(args, f"{prefix}_pwl_units")
        if latent < 3 or not 0 <= pwl <= latent:
            parser.error(f"{prefix}: latent dimension must be >= 3 and PWL units between 0 and latent dimension")
    if not all(math.isfinite(value) for value in (args.parameter_min, args.parameter_max)) or args.parameter_min >= args.parameter_max:
        parser.error("Parameter bounds must be finite and --parameter-min < --parameter-max")
    for name in ("training_data_root", "backend_script", "run_root", "plot_root"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    return args


def variants(args):
    return {
        "cphi": Variant("Cphi", ("additive",), args.cphi_latent_dim, args.cphi_pwl_units),
        "cphi_shifted_relu": Variant("Cphi + shifted ReLU", ("additive", "shifted-relu"),
                                     args.shifted_latent_dim, args.shifted_pwl_units),
    }


def prepare_data(args, script_dir):
    """Fail before long trainings if data or normalization cannot be used."""
    from compare_model_bifurcation_diagrams import load_training_coordinates
    import numpy as np

    for path in (script_dir / "train_halvorsen_plain.py", script_dir / "mixed_trajectory_dataset.py",
                 script_dir / "trajectory_storage.py", script_dir / "compare_model_bifurcation_diagrams.py",
                 args.backend_script):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")
    time_steps = []
    for name in DATASETS:
        folder = args.training_data_root / name
        if (folder / ".compact_in_progress.json").exists():
            raise ValueError(f"Dataset conversion is incomplete: {folder}")
        for filename in ("trajectories.npy", "trajectory_phi.npy", "trajectory_lengths.npy",
                         "trajectory_train_lengths.npy", "trajectory_group_names.npy",
                         "trajectory_mean.npy", "trajectory_std.npy", "metadata.json"):
            if not (folder / filename).is_file():
                raise FileNotFoundError(f"Missing required file: {folder / filename}")
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
        if metadata.get("format") not in ("alrnn_mixed_parameter_trajectories_v1", "alrnn_mixed_parameter_trajectories_v2"):
            raise ValueError(f"Expected a completed mixed dataset: {folder}")
        if metadata.get("config", {}).get("regime") != REGIME:
            raise ValueError(f"Expected {REGIME}: {folder}")
        lengths = np.load(folder / "trajectory_train_lengths.npy", allow_pickle=False)
        if lengths.ndim != 1 or not lengths.size or np.any(lengths <= args.sequence_length):
            raise ValueError(f"Training prefixes must exceed --sequence-length: {folder}")
        scaling = load_training_coordinates(folder)
        dt = float(metadata["config"]["dt"])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"Invalid simulation dt: {folder}")
        time_steps.append(dt)
        print(f"Dataset: {folder} | mean={scaling.mean} | std={scaling.std}", flush=True)
    if not np.allclose(time_steps, time_steps[0], atol=0, rtol=1e-12):
        raise ValueError("Bifurcation comparisons require the same simulation dt for all three datasets")
    return time_steps[0]


def job_definitions(model_variants):
    trainings = {
        f"{variant}__{dataset}": (variant, dataset)
        for dataset in DATASETS for variant in model_variants
    }
    comparisons = {f"{variant}__bifurcation": variant for variant in model_variants}
    return trainings, comparisons


def training_step(args, script_dir, run_root, variant_name, variant, dataset, gpu):
    command = [
        sys.executable, "-u", str(script_dir / "train_halvorsen_plain.py"),
        "--data-dir", str(args.training_data_root / dataset), "--multi-trajectory",
        "--phi-terms", *variant.terms,
        "--trajectory-sampling", "length", "--mixed-balanced-sampling",
        "--time-sampling-bins", str(args.time_sampling_bins),
        "--latent-dim", str(variant.latent_dim), "--pwl-units", str(variant.pwl_units),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--batches-per-epoch", str(args.batches_per_epoch),
        "--sequence-length", str(args.sequence_length), "--seed", str(args.seed),
        "--gpu-id", str(gpu), "--cpu-threads", str(args.cpu_threads), "--no-best-model",
        "--output-dir", str(run_root / variant_name / dataset),
    ]
    return Step(f"train {variant.label}: {DATASETS[dataset]}", command, script_dir)


def comparison_step(args, script_dir, run_root, plot_root, variant_name, variant, gpu, dt):
    output_dir = plot_root / variant_name
    command = [
        sys.executable, "-u", str(script_dir / "compare_model_bifurcation_diagrams.py"),
        "--backend-script", str(args.backend_script), "--use-training-normalization",
        "--model-type", "alrnn", "--regime", REGIME,
        "--model-runs", *(str(run_root / variant_name / name) for name in DATASETS),
        "--training-data-dirs", *(str(args.training_data_root / name) for name in DATASETS),
        "--model-labels", *(f"{variant.label}: {label}" for label in DATASETS.values()),
        "--parameter-min", str(args.parameter_min), "--parameter-max", str(args.parameter_max),
        "--num-parameters", str(args.num_parameters), "--components", *args.components,
        "--dt", str(dt),
        "--output", str(output_dir / "bifurcation_compare.png"),
        "--cache", str(output_dir / "bifurcation_cache.npz"),
        "--force-recompute", "--device", "cuda", "--gpu-id", str(gpu),
        "--torch-threads", str(args.cpu_threads),
    ]
    return Step(f"bifurcations + ground truth: {variant.label}", command, args.backend_script.parent)


def run_phases(gpu_file, gpu_ids, device_count, log_dir, make_steps, env, trainings, comparisons):
    results = schedule(gpu_file, gpu_ids, device_count, log_dir, make_steps, env,
                       job_names=list(trainings), max_gpus=MAX_GPUS)
    eligible, skipped = [], []
    for job, variant in comparisons.items():
        dependencies = [name for name, (model, _) in trainings.items() if model == variant]
        if all(results[name] == 0 for name in dependencies):
            eligible.append(job)
        else:
            print(f"Skipping {job}: a required training failed.", flush=True)
            skipped.append(job)
            results[job] = 1
    if eligible:
        print("Training phase finished; starting bifurcation comparisons.", flush=True)
        results.update(schedule(gpu_file, gpu_ids, device_count, log_dir, make_steps, env,
                                job_names=eligible, max_gpus=MAX_GPUS))
    return results, skipped


def main(argv=None):
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    model_variants = variants(args)
    trainings, comparisons = job_definitions(model_variants)
    run_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
    family = Path("term_ablations/dadras/combined_dadras_balanced")
    run_root = args.run_root or script_dir / "runs" / family / run_id
    plot_root = args.plot_root or script_dir / "bifurcation_diagrams" / family / run_id
    dt = 0.01 if args.dry_run else prepare_data(args, script_dir)

    def make_steps(job, gpu):
        if job in trainings:
            variant, dataset = trainings[job]
            return [training_step(args, script_dir, run_root, variant, model_variants[variant], dataset, gpu)]
        variant = comparisons[job]
        return [comparison_step(args, script_dir, run_root, plot_root, variant, model_variants[variant], gpu, dt)]

    if args.dry_run:
        print("DRY RUN: six trainings, then two bifurcation comparisons; no files or CUDA required.")
        print("GPU assignments below are illustrative; queued jobs start on the next available GPU.")
        for index, job in enumerate([*trainings, *comparisons]):
            gpu = args.gpu_ids[index % len(args.gpu_ids)]
            step = make_steps(job, gpu)[0]
            print(f"\n[{job}] GPU {gpu} | {step.label}\ncwd: {step.cwd}\n{shlex.join(step.command)}")
        return 0

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected Python environment")
    device_count = torch.cuda.device_count()
    validate_gpu_ids(args.gpu_ids, device_count, max_gpus=MAX_GPUS)
    for gpu in args.gpu_ids:
        print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)}", flush=True)
    for folder in (run_root, plot_root):
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            raise ValueError(f"Output root must be new or empty: {folder}")
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True)
    gpu_file = run_root / "gpu_ids.json"
    gpu_file.write_text(json.dumps(args.gpu_ids) + "\n", encoding="utf-8")
    plan = {
        "args": vars(args), "variants": {name: asdict(value) for name, value in model_variants.items()},
        "training_jobs": trainings, "comparison_jobs": comparisons,
        "run_root": run_root, "plot_root": plot_root, "bifurcation_dt": dt,
    }
    (run_root / "plan.json").write_text(json.dumps(plan, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Models, plan and logs: {run_root}\nBifurcation plots/cache: {plot_root}", flush=True)
    print(f"Progress: tail -f {shlex.quote(str(log_dir))}/*.log", flush=True)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.cpu_threads)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        results, skipped = run_phases(gpu_file, args.gpu_ids, device_count, log_dir,
                                      make_steps, env, trainings, comparisons)
    except KeyboardInterrupt:
        print("Stopped: child processes terminated; completed models remain on disk.", flush=True)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    (run_root / "summary.json").write_text(
        json.dumps({"exit_codes": results, "skipped_comparisons": skipped}, indent=2) + "\n", encoding="utf-8"
    )
    failed = [job for job, code in results.items() if code]
    if failed:
        print(f"Failed/skipped jobs: {', '.join(failed)}. See {log_dir}", flush=True)
        return 1
    print(f"Finished six trainings and two bifurcation comparisons. Plots: {plot_root}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
