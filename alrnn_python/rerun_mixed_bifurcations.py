"""Repeat only mixed-experiment bifurcation jobs from an existing plan.json.

Training checkpoints and original logs/summary are never modified. GPU IDs for
the retry are explicit; an earlier run's live GPU list is not reused.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import sys

import run_dadras_combined_balanced as runner


def prepare(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Existing training folder containing plan.json.")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--backend-script", type=Path)
    parser.add_argument("--training-data-root", type=Path)
    parser.add_argument("--plot-root", type=Path)
    parser.add_argument("--components", nargs="+", choices=("x", "y", "z"))
    parser.add_argument("--num-parameters", type=int)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    retry = parser.parse_args(argv)
    retry.run_root = retry.run_root.expanduser().resolve()
    plan = json.loads((retry.run_root / "plan.json").read_text(encoding="utf-8-sig"))
    saved = plan["args"]
    # Rebuild the analysis arguments through the normal parser/validation path.
    command = ["--system", saved["system"],
               "--training-data-root", str(retry.training_data_root or saved["training_data_root"]),
               "--backend-script", str(retry.backend_script or saved["backend_script"]),
               "--gpu-ids", *map(str, retry.gpu_ids),
               "--cpu-threads", str(saved["cpu_threads"]),
               "--sequence-length", str(saved["sequence_length"]),
               "--parameter-min", str(saved["parameter_min"]),
               "--parameter-max", str(saved["parameter_max"]),
               "--num-parameters", str(retry.num_parameters if retry.num_parameters is not None else saved["num_parameters"]),
               "--components", *(retry.components or saved["components"])]
    args = runner.parse_args(command)
    variants = {name: runner.Variant(**{**settings, "terms": tuple(settings["terms"])})
                for name, settings in plan["variants"].items()}
    comparisons = plan["comparison_jobs"]
    if not comparisons or any(name not in variants for name in comparisons.values()):
        raise ValueError("Saved plan has no valid bifurcation comparisons.")
    plots = Path(retry.plot_root or plan["plot_root"]).expanduser().resolve()
    return retry, args, variants, comparisons, plots


def main(argv=None):
    retry, args, variants, comparisons, plots = prepare(argv)
    scripts = Path(__file__).resolve().parent
    dt = 0.01
    if retry.dry_run:
        plan = json.loads((retry.run_root / "plan.json").read_text(encoding="utf-8-sig"))
        dt = float(plan.get("bifurcation_dt", dt))
    else:
        dt = runner.prepare_data(args, scripts)
        for variant in comparisons.values():
            for dataset in runner.EXPERIMENTS[args.system]["datasets"]:
                for filename in ("model.pt", "metrics.json"):
                    path = retry.run_root / variant / dataset / filename
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing trained model: {path}")

    def steps(job, gpu):
        name = comparisons[job]
        step = runner.comparison_step(args, scripts, retry.run_root, plots, name, variants[name], gpu, dt)
        if retry.reuse_cache:
            step.command.remove("--force-recompute")
        return [step]

    if retry.dry_run:
        print(f"DRY RUN: {len(comparisons)} bifurcation jobs; no training, no files or CUDA required.")
        for index, job in enumerate(comparisons):
            step = steps(job, args.gpu_ids[index % len(args.gpu_ids)])[0]
            print(f"[{job}]\ncwd: {step.cwd}\n{shlex.join(step.command)}")
        return 0
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the selected environment")
    count = torch.cuda.device_count()
    runner.validate_gpu_ids(args.gpu_ids, device_count=count, max_gpus=runner.MAX_GPUS)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
    logs = retry.run_root / "logs" / f"bifurcation_retry_{stamp}"
    logs.mkdir(parents=True)
    gpu_file = logs / "gpu_ids.json"
    gpu_file.write_text(json.dumps(args.gpu_ids) + "\n", encoding="utf-8")
    print(f"Bifurcation retry only; existing models: {retry.run_root}\nPlots: {plots}\nLogs: {logs}", flush=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(args.cpu_threads)
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        results = runner.schedule(gpu_file, args.gpu_ids, count, logs, steps, env,
                                  job_names=list(comparisons), max_gpus=runner.MAX_GPUS)
    except KeyboardInterrupt:
        print("Stopped bifurcation retry; child processes terminated and trained models preserved.", flush=True)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
    (logs / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return int(any(code != 0 for code in results.values()))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
