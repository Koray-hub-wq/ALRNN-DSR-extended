"""Schedule four external-input ablations, one sequential pipeline per GPU."""

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


VARIANTS = {
    "diagonal": ("diagonal",),
    "diagonal_relu": ("diagonal-relu",),
    "shifted_relu": ("shifted-relu",),
    "additive_diagonal": ("additive", "diagonal"),
}
PARAMETER_RANGES = {
    "dadras_e_49_to_64": (4.9, 6.2),
    "rossler_a_005_to_035": (0.05, 0.35),
}


@dataclass
class Step:
    label: str
    command: list
    cwd: Path


def validate_gpu_ids(values, device_count=None):
    if not isinstance(values, list) or not 1 <= len(values) <= 4:
        raise ValueError("GPU IDs must be a JSON list of one to four integers.")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("GPU IDs must be nonnegative integers.")
    if len(values) != len(set(values)):
        raise ValueError("GPU IDs must not contain duplicates.")
    if device_count is not None and any(value >= device_count for value in values):
        raise ValueError(f"Invalid GPU ID; visible CUDA devices: {device_count}.")
    return values


def load_groups(data_dir, regime):
    import numpy as np

    with (data_dir / "metadata.json").open(encoding="utf-8-sig") as stream:
        metadata = json.load(stream)
    values = metadata.get("args", {}).get("phi_values")
    if values is None:
        values = metadata.get("requested_phi_values")
    if values is None:
        raise ValueError("metadata.json contains no phi_values or requested_phi_values.")
    values = sorted(set(float(value) for value in values))
    if len(values) != 8 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected eight finite phi values, found {values}.")
    available = np.unique(np.load(data_dir / "trajectory_phi_constants.npy"))
    if len(available) != len(values) or not np.allclose(
        available, values, atol=1e-5, rtol=0
    ):
        raise ValueError("Metadata phi values do not match trajectory constants.")

    if regime == "dadras_e_49_to_64":
        values = [v for v in values if not math.isclose(v, 1, abs_tol=1e-5, rel_tol=0)]
        negative = [v for v in values if v < 0]
        positive = [v for v in values if v > 0]
        expected = (7, 4, 3)
    else:
        moved = [v for v in values if math.isclose(v, -0.142857, abs_tol=1e-5, rel_tol=0)]
        if len(moved) != 1:
            raise ValueError("Expected exactly one phi near -0.142857 for Rossler.")
        negative = [v for v in values if v < 0 and v != moved[0]]
        positive = [v for v in values if v > 0 or v == moved[0]]
        expected = (8, 3, 5)
    groups = {"all": values, "negative": negative, "positive": positive}
    if tuple(map(len, groups.values())) != expected:
        raise ValueError(f"Expected group sizes {expected}, found {groups}.")
    return groups


def build_steps(args, script_dir, groups, run_root, plot_root, variant, gpu_id):
    steps = []
    run_dirs = [run_root / variant / subset for subset in groups]
    for (subset, phi_values), output_dir in zip(groups.items(), run_dirs):
        command = [
            sys.executable, "-u", str(script_dir / "train_halvorsen_plain.py"),
            "--data-dir", str(args.data_dir), "--multi-trajectory",
            "--phi-terms", *VARIANTS[variant],
            "--gpu-id", str(gpu_id), "--cpu-threads", str(args.cpu_threads),
            "--latent-dim", str(args.latent_dim), "--pwl-units", str(args.pwl_units),
            "--batch-size", str(args.batch_size), "--epochs", str(args.epochs),
            "--batches-per-epoch", str(args.batches_per_epoch),
            "--train-phi-values", *(str(value) for value in phi_values),
            "--output-dir", str(output_dir),
        ]
        steps.append(Step(f"train {subset}", command, script_dir))

    if args.regime == "dadras_e_49_to_64":
        labels = ["all except phi=+1", "negative", "positive except phi=+1"]
    else:
        labels = ["all 8 values", "negative except phi=-0.142857", "positive plus phi=-0.142857"]
    lower, upper = PARAMETER_RANGES[args.regime]
    output_dir = plot_root / variant
    command = [
        sys.executable, "-u", str(script_dir / "compare_model_bifurcation_diagrams.py"),
        "--backend-script", str(args.backend_script),
        "--model-type", "alrnn", "--regime", args.regime,
        "--model-runs", *(str(path) for path in run_dirs),
        "--model-labels", *(f"{variant}: {label}" for label in labels),
        "--parameter-min", str(lower), "--parameter-max", str(upper),
        "--components", "z",
        "--output", str(output_dir / "bifurcation_compare.png"),
        "--cache", str(output_dir / "bifurcation_cache.npz"),
        "--force-recompute", "--device", "cuda", "--gpu-id", str(gpu_id),
        "--torch-threads", str(args.cpu_threads), "--num-parameters", "320",
    ]
    steps.append(Step("bifurcation comparison + ground truth", command, args.backend_script.parent))
    return steps


def stop_processes(active):
    # Each stage owns its process group on Linux, including any child processes.
    for job in active.values():
        process = job["process"]
        if process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
    deadline = time.monotonic() + 5
    for job in active.values():
        process = job["process"]
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
        job["log"].close()


def schedule(gpu_file, gpu_ids, device_count, log_dir, make_steps, env, poll_interval=2):
    pending = deque(VARIANTS)
    active = {}
    results = {}
    last_warning = None

    def start_step(job):
        step = job["steps"].popleft()
        print(f"[{job['variant']}] GPU {job['gpu']}: {step.label}", flush=True)
        job["log"].write(f"\n### {step.label}\n{shlex.join(step.command)}\n")
        job["log"].flush()
        job["process"] = subprocess.Popen(
            step.command, cwd=step.cwd, env=env,
            stdout=job["log"], stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )

    def finish(gpu, returncode):
        job = active.pop(gpu)
        job["log"].close()
        results[job["variant"]] = returncode
        state = "finished" if returncode == 0 else f"FAILED (exit {returncode})"
        print(f"[{job['variant']}] {state}. Log: {job['log'].name}", flush=True)

    try:
        while pending or active:
            try:
                with gpu_file.open(encoding="utf-8-sig") as stream:
                    updated = validate_gpu_ids(json.load(stream), device_count)
                if updated != gpu_ids:
                    gpu_ids = updated
                    print(f"GPU queue updated: {gpu_ids}", flush=True)
                last_warning = None
            except (OSError, ValueError) as exc:
                warning = str(exc)
                if warning != last_warning:
                    print(f"Ignoring invalid GPU file; keeping {gpu_ids}: {warning}", flush=True)
                    last_warning = warning

            for gpu, job in list(active.items()):
                returncode = job["process"].poll()
                if returncode is None:
                    continue
                if returncode != 0 or not job["steps"]:
                    finish(gpu, returncode)
                else:
                    try:
                        start_step(job)
                    except OSError as exc:
                        print(f"[{job['variant']}] Cannot launch stage: {exc}", flush=True)
                        finish(gpu, 1)

            for gpu in gpu_ids:
                if not pending:
                    break
                if gpu in active:
                    continue
                variant = pending.popleft()
                steps = deque(make_steps(variant, gpu))
                log = (log_dir / f"{variant}.log").open("w", encoding="utf-8")
                job = {"variant": variant, "gpu": gpu, "steps": steps, "log": log}
                print(f"Starting {variant} on GPU {gpu}. Log: {log.name}", flush=True)
                try:
                    start_step(job)
                except OSError as exc:
                    log.close()
                    results[variant] = 1
                    print(f"[{variant}] Cannot launch stage: {exc}", flush=True)
                else:
                    active[gpu] = job
            if pending or active:
                time.sleep(poll_interval)
    finally:
        stop_processes(active)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=PARAMETER_RANGES, required=True)
    parser.add_argument("--training-data-root", type=Path, required=True)
    parser.add_argument("--backend-script", type=Path, required=True)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--latent-dim", type=int, default=200)
    parser.add_argument("--pwl-units", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Print all commands; do not train or require CUDA.")
    args = parser.parse_args(argv)
    validate_gpu_ids(args.gpu_ids)
    for name in ("latent_dim", "epochs", "batch_size", "batches_per_epoch", "cpu_threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if not 0 <= args.pwl_units <= args.latent_dim:
        parser.error("--pwl-units must lie between zero and --latent-dim.")
    args.backend_script = args.backend_script.expanduser().resolve()
    args.data_dir = (args.training_data_root.expanduser() / args.regime).resolve()
    if not (args.data_dir / "trajectories.npy").is_file():
        if (args.data_dir / "values_8").is_dir():
            args.data_dir /= "values_8"
    return args


def main(argv=None):
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    for path in [
        script_dir / "train_halvorsen_plain.py",
        script_dir / "compare_model_bifurcation_diagrams.py", args.backend_script,
        *(args.data_dir / name for name in (
            "trajectories.npy", "trajectory_phi.npy", "trajectory_phi_constants.npy", "metadata.json"
        )),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")
    groups = load_groups(args.data_dir, args.regime)
    for subset, values in groups.items():
        print(f"{subset} ({len(values)} phi values): {values}", flush=True)

    batch_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_m{args.latent_dim}_p{args.pwl_units}"
    run_root = script_dir / "runs/term_ablations" / args.regime / batch_id
    plot_root = script_dir / "bifurcation_diagrams/term_ablations" / args.regime / batch_id

    def make_steps(variant, gpu):
        return build_steps(args, script_dir, groups, run_root, plot_root, variant, gpu)

    if args.dry_run:
        print("DRY RUN: no output files or CUDA processes will be created.")
        for index, variant in enumerate(VARIANTS):
            gpu = args.gpu_ids[index % len(args.gpu_ids)]
            for step in make_steps(variant, gpu):
                print(f"\n[{variant}] {step.label}\ncwd: {step.cwd}\n{shlex.join(step.command)}")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected Python environment.")
    device_count = torch.cuda.device_count()
    validate_gpu_ids(args.gpu_ids, device_count)
    for gpu in args.gpu_ids:
        print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)}", flush=True)

    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.cpu_threads)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True)
    for variant in VARIANTS:
        (log_dir / f"{variant}.log").touch()
    gpu_file = run_root / "gpu_ids.json"
    gpu_file.write_text(json.dumps(args.gpu_ids) + "\n", encoding="utf-8")
    plan = {
        "regime": args.regime,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "groups": groups, "variants": VARIANTS, "run_root": str(run_root), "plot_root": str(plot_root),
    }
    (run_root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"\nModels: {run_root}\nDiagrams/cache: {plot_root}", flush=True)
    print(f"Live GPU list: {gpu_file}\nEdit this JSON list to add GPUs for pending variants.", flush=True)
    print(f"Progress: tail -f {shlex.quote(str(log_dir))}/*.log", flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        results = schedule(gpu_file, args.gpu_ids, device_count, log_dir, make_steps, env)
    except KeyboardInterrupt:
        print("Stopped: running process groups terminated; completed runs remain on disk.", flush=True)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    (run_root / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    failed = [variant for variant, code in results.items() if code != 0]
    if failed:
        print(f"Failed variants: {', '.join(failed)}. See logs in {log_dir}.", flush=True)
        return 1
    print(f"Finished all 12 trainings and four bifurcation comparisons. Diagrams: {plot_root}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        sys.exit(str(exc))
