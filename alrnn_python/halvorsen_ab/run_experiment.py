"""Generate, train and analyze the full-box two-parameter Halvorsen experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from halvorsen_ab.common import ALRNN_DIR, DEFAULT_ROOT, HERE, SIMULATOR_HELP, load_config, load_dataset_metadata, load_simulator
from halvorsen_ab.generate_data import generation_identity


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "generate", "train", "analyze", "ground-truth", "plot"), default="all")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--simulator-root", type=Path, default=DEFAULT_ROOT, help=SIMULATOR_HELP)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=300)
    parser.add_argument("--pwl-units", type=int, default=150)
    parser.add_argument("--phi-terms", nargs="+", choices=("additive", "diagonal", "diagonal-relu", "shifted-relu"), default=["additive"])
    parser.add_argument("--grid-a", type=int, default=41, help="Bifurcation grid, independently of the 5x5 training grid.")
    parser.add_argument("--grid-b", type=int, default=41)
    parser.add_argument("--evaluation-initial-states", type=int, default=2)
    parser.add_argument("--observation-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--max-plot-peaks", type=int, default=128)
    parser.add_argument("--smoke", action="store_true", help="Tiny end-to-end check; never a scientific training result.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.epochs, args.batches_per_epoch, args.batch_size = 2, 2, 25
        args.latent_dim, args.pwl_units, args.sequence_length = 12, 4, 32
        args.grid_a, args.grid_b, args.evaluation_initial_states = 3, 3, 1
        args.observation_steps, args.warmup_steps = 128, 16
    args.data_dir = args.data_dir or ALRNN_DIR / (
        "runs/halvorsen_ab_smoke/data" if args.smoke else "runs/halvorsen_ab/data_grid5x5")
    args.run_dir = args.run_dir or ALRNN_DIR / (
        "runs/halvorsen_ab_smoke/model" if args.smoke else "runs/halvorsen_ab/cphi_m300_p150")
    for name in ("cpu_threads", "epochs", "batches_per_epoch", "batch_size", "sequence_length",
                 "evaluation_initial_states", "evaluation_batch_size", "max_plot_peaks"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive.")
    if args.gpu_id < 0 or min(args.grid_a, args.grid_b, args.observation_steps) < 2 or args.warmup_steps < 0:
        parser.error("Invalid GPU index, analysis grid or observation/warmup lengths.")
    if not 3 <= args.latent_dim or not 0 <= args.pwl_units <= args.latent_dim:
        parser.error("Require latent_dim >=3 and 0 <= pwl_units <= latent_dim.")
    for name in ("config", "simulator_root", "data_dir", "run_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_dir = (args.output_dir or args.run_dir / "analysis").expanduser().resolve()
    return args


def commands(args):
    generate = [sys.executable, "-u", str(HERE / "generate_data.py"), "--config", str(args.config),
                "--output-dir", str(args.data_dir), "--simulator-root", str(args.simulator_root)]
    if args.smoke:
        generate.append("--smoke")
    train = [sys.executable, "-u", str(ALRNN_DIR / "train_halvorsen_plain.py"),
             "--data-dir", str(args.data_dir), "--output-dir", str(args.run_dir),
             "--multi-trajectory", "--mixed-balanced-sampling", "--trajectory-sampling", "uniform",
             "--phi-terms", *args.phi_terms, "--no-best-model", "--gpu-id", str(args.gpu_id),
             "--cpu-threads", str(args.cpu_threads), "--epochs", str(args.epochs),
             "--batches-per-epoch", str(args.batches_per_epoch), "--batch-size", str(args.batch_size),
             "--sequence-length", str(args.sequence_length), "--latent-dim", str(args.latent_dim),
             "--pwl-units", str(args.pwl_units), "--t-gen", str(args.observation_steps),
             "--t-transient", str(args.warmup_steps)]
    if args.device == "cpu":
        train.append("--force-cpu")
    analyze = [sys.executable, "-u", str(HERE / "evaluate.py"), "--data-dir", str(args.data_dir),
               "--run-dir", str(args.run_dir), "--output-dir", str(args.output_dir),
               "--simulator-root", str(args.simulator_root), "--device", args.device,
               "--gpu-id", str(args.gpu_id), "--cpu-threads", str(args.cpu_threads),
               "--grid-a", str(args.grid_a), "--grid-b", str(args.grid_b),
               "--initial-states", str(args.evaluation_initial_states),
               "--observation-steps", str(args.observation_steps), "--warmup-steps", str(args.warmup_steps),
               "--batch-size", str(args.evaluation_batch_size), "--max-plot-peaks", str(args.max_plot_peaks)]
    if args.smoke:
        analyze += ["--report-label", "Smoke test: tiny training, not a scientific result"]
    if args.stage == "ground-truth":
        analyze.append("--ground-truth-only")
    if args.stage == "plot":
        analyze.append("--plot-only")
    return dict(generate=generate, train=train, analyze=analyze)


def run_logged(command, path, env):
    with path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ALRNN_DIR, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    config = load_config(args.config)
    if args.smoke:
        config = replace(config, n_steps=400, burn_in_steps=20, trajectories_per_pair=1)
    if config.train_length <= args.sequence_length:
        raise ValueError("Training prefixes must exceed sequence_length.")
    plan = commands(args)
    stages = ["generate", "train", "analyze"] if args.stage == "all" else [
        "analyze" if args.stage in ("ground-truth", "plot") else args.stage]
    for stage in stages:
        command = plan[stage]
        print(f"[{stage}] {subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command)}", flush=True)
    if args.dry_run:
        print("DRY RUN: no simulations, training or output files created.")
        return 0
    if args.stage != "plot":
        _, simulator_hash = load_simulator(args.simulator_root)
        if (args.data_dir / "metadata.json").is_file():
            saved_config, _, _, metadata = load_dataset_metadata(args.data_dir)
            if saved_config != config or metadata["generation_identity"] != generation_identity(config, simulator_hash):
                raise ValueError("Existing data have a different configuration or simulator/generator source. Use a new data-dir.")
            if "generate" in stages:
                stages.remove("generate")
                print("Reusing verified completed dataset.", flush=True)
        elif "generate" not in stages:
            raise FileNotFoundError(f"Generate the dataset first: {args.data_dir}")
    if "train" in stages:
        if (args.run_dir / "model.pt").exists() or (args.run_dir / "metrics.json").exists():
            raise ValueError("Use a new run-dir for training; existing checkpoints are never overwritten.")
        if args.device == "cuda":
            import torch
            if not torch.cuda.is_available() or args.gpu_id >= torch.cuda.device_count():
                raise RuntimeError("Requested CUDA GPU is unavailable; use --device cpu for local checks.")
    log_dir = args.run_dir / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "plan.json").write_text(json.dumps(dict(args=vars(args), data_config=asdict(config), commands=plan),
                                               indent=2, default=str) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    env.setdefault("MPLCONFIGDIR", str(log_dir / "matplotlib"))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(args.cpu_threads)
    for stage in stages:
        run_logged(plan[stage], log_dir / f"{stage}.log", env)
        if stage == "train":
            _, mean, std, metadata = load_dataset_metadata(args.data_dir)
            manifest = dict(generation_identity=metadata["generation_identity"],
                            data_config=metadata["data_config"], parameter_names=["a", "b"],
                            phi_reference=metadata["phi_reference"],
                            state_mean=mean.tolist(), state_std=std.tolist())
            (args.run_dir / "halvorsen_ab_training.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
