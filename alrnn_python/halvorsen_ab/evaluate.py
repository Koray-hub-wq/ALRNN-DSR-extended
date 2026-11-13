"""Two-parameter Halvorsen bifurcations and aligned free-orbit comparisons."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

from halvorsen_ab.common import (DEFAULT_ROOT, HERE, SIMULATOR_HELP, file_hash, identity, load_dataset_metadata,
                                 load_simulator, simulate_with_retries)

CACHE_VERSION = 1


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def pack_ragged(values):
    lengths = [len(value) for value in values]
    return np.concatenate(values).astype(np.float64), np.r_[0, np.cumsum(lengths, dtype=np.int64)]


def unpack_ragged(values, offsets):
    return [values[int(start):int(end)] for start, end in zip(offsets[:-1], offsets[1:])]


def extrema(states):
    """No invented maxima for equilibria; classify only a settled late trajectory."""
    if not np.isfinite(states).all():
        return np.empty(0), float("nan")
    z = np.asarray(states)[:, 2]
    tolerance = 1e-6 * max(1.0, np.max(np.abs(states)))
    if np.all(np.ptp(states, axis=0) <= tolerance):
        return np.empty(0), float(z.mean())
    return z[find_peaks(z)[0]], float("nan")


def save_cache(path, arrays, metadata):
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays, metadata_json=np.array(json.dumps(finite_json(metadata), allow_nan=False)))
    temporary.replace(path)


def read_cache(path):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files if name != "metadata_json"}
        metadata = json.loads(str(saved["metadata_json"]))
    return arrays, metadata


def ground_truth(args, config, module, simulator_hash):
    settings = dict(cache_version=CACHE_VERSION, data_config=asdict(config),
                    grid_a=args.grid_a, grid_b=args.grid_b, initial_states=args.initial_states,
                    observation_steps=args.observation_steps, warmup_steps=args.warmup_steps,
                    simulator_sha256=simulator_hash, common_sha256=file_hash(HERE / "common.py"),
                    evaluator_sha256=file_hash(__file__), evaluation_seed=config.seed + 100000)
    cache_id = identity(settings)
    path = args.output_dir / "ground_truth.npz"
    if path.is_file():
        arrays, metadata = read_cache(path)
        if metadata.get("identity") == cache_id:
            print(f"Reusing verified ground truth: {path}", flush=True)
            return arrays, metadata
    pairs = np.repeat(config.pairs(args.grid_a, args.grid_b), args.initial_states, axis=0)
    starts = np.full((len(pairs), 3), np.nan)
    initial_states = starts.copy()
    equilibria = np.full(len(pairs), np.nan)
    maxima, failures, retries = [], [], []
    rng = np.random.default_rng(settings["evaluation_seed"])
    for i, pair in enumerate(tqdm(pairs, desc="Ground-truth bifurcations")):
        try:
            states, initial, rejected = simulate_with_retries(
                module, config, pair, rng, n_steps=args.warmup_steps + args.observation_steps)
            starts[i], initial_states[i] = states[0], initial
            peaks, equilibria[i] = extrema(states[args.warmup_steps:])
            maxima.append(peaks)
            if rejected:
                retries.append(dict(index=i, pair=pair.tolist(), attempts=rejected))
        except RuntimeError as exc:
            maxima.append(np.empty(0))
            failures.append(dict(index=i, pair=pair.tolist(), reason=str(exc)))
    peaks, offsets = pack_ragged(maxima)
    arrays = dict(pairs=pairs, starts=starts, initial_states=initial_states,
                  truth_peaks=peaks, truth_offsets=offsets, truth_equilibria=equilibria)
    metadata = dict(identity=cache_id, settings=settings, failures=failures, retries=retries,
                    sample_convention="Start both systems at the same post-burn-in GT state; discard warmup, then find sampled local z maxima.")
    save_cache(path, arrays, metadata)
    return arrays, metadata


def rollout_model(model, starts, pairs, mean, std, config, steps, device):
    """x[0] is the observed seed; prediction[t+1] follows one model transition."""
    import torch
    with torch.inference_mode():
        inputs = torch.as_tensor(config.normalize(pairs), dtype=torch.float32, device=device)
        raw_starts = np.asarray(starts)
        valid = torch.as_tensor(np.isfinite(raw_starts).all(axis=1), device=device)
        x0 = torch.as_tensor(np.nan_to_num((raw_starts - mean) / std), dtype=torch.float32, device=device)
        z = x0 @ model.B
        z[:, :3] = x0
        observed = torch.full((len(starts), steps, 3), float("nan"), device=device)
        observed[valid, 0] = x0[valid]
        failed_at = torch.full((len(starts),), -1, dtype=torch.int64, device=device)
        failed_at[~valid] = 0
        shift = torch.as_tensor(mean, dtype=torch.float32, device=device)
        scale = torch.as_tensor(std, dtype=torch.float32, device=device)
        for step in range(1, steps):
            updated = model(z, inputs)
            raw = updated[:, :3] * scale + shift
            finite = torch.isfinite(updated).all(dim=1) & (raw.abs().amax(dim=1) < config.escape_radius)
            failed_at[valid & ~finite] = step
            valid &= finite
            observed[valid, step] = updated[valid, :3]
            z = torch.where(valid[:, None], updated, torch.zeros_like(updated))
            if not bool(valid.any()):
                break
        raw = observed.cpu().numpy() * std + mean
    return raw, failed_at.cpu().numpy()


def compare_metrics(truth, prediction):
    if not np.isfinite(prediction).all():
        return dict(status="non_finite_or_escaped", dstsp=None, dh=None, rmse=None)
    from metrics import power_spectrum_error, state_space_divergence_binning
    import torch
    result = dict(status="completed", rmse=float(np.sqrt(np.mean((truth - prediction) ** 2))), dstsp=None, dh=None)
    # The existing histogram and spectrum metrics are undefined for constant axes.
    if np.all(np.std(truth, axis=0) > 1e-7):
        result["dstsp"] = state_space_divergence_binning(
            torch.as_tensor(prediction, dtype=torch.float32), torch.as_tensor(truth, dtype=torch.float32))
    if np.all(np.std(truth, axis=0) > 1e-7) and np.all(np.std(prediction, axis=0) > 1e-7):
        result["dh"] = float(power_spectrum_error(torch.as_tensor(truth), torch.as_tensor(prediction)))
    return finite_json(result)


def trajectory_comparisons(args, config, model, mean, std):
    states = np.load(args.data_dir / "trajectories.npy", mmap_mode="r", allow_pickle=False)
    lengths = np.load(args.data_dir / "trajectory_lengths.npy", allow_pickle=False)
    train = np.load(args.data_dir / "trajectory_train_lengths.npy", allow_pickle=False)
    constants = np.load(args.data_dir / "trajectory_parameter_constants.npy", allow_pickle=False)
    pair_indices = np.load(args.data_dir / "trajectory_pair_index.npy", allow_pickle=False)
    indices = [int(np.flatnonzero(pair_indices == i)[0]) for i in range(len(config.pairs()))]
    steps = min(int(np.min(lengths - train)), args.warmup_steps + args.observation_steps)
    if steps - args.warmup_steps < 2:
        raise ValueError("Not enough test-tail samples after warmup for trajectory comparisons.")
    offsets = np.r_[0, lengths.cumsum()]
    truth = np.stack([states[int(offsets[i] + train[i]):int(offsets[i] + train[i] + steps)] * std + mean for i in indices])
    pairs = constants[indices]
    if model is None:
        prediction = np.full_like(truth, np.nan)
        failures = np.full(len(indices), -1, dtype=np.int64)
    else:
        prediction, failures = rollout_model(model, truth[:, 0], pairs, mean, std, config, steps, args.device)
    metrics = []
    for i in range(len(indices)):
        result = (dict(status="ground_truth_only", dstsp=None, dh=None, rmse=None) if model is None else
                  compare_metrics(truth[i, args.warmup_steps:], prediction[i, args.warmup_steps:]))
        result.update(a=float(pairs[i, 0]), b=float(pairs[i, 1]), trajectory_index=indices[i],
                      failed_at=int(failures[i]), warmup_steps=args.warmup_steps,
                      metric_steps=steps - args.warmup_steps)
        metrics.append(result)
    return dict(orbit_pairs=pairs, orbit_truth=truth[:, args.warmup_steps:],
                orbit_model=prediction[:, args.warmup_steps:]), metrics


def analyze(args):
    import torch
    from alrnn_model import load_saved_model
    config, mean, std, dataset = load_dataset_metadata(args.data_dir)
    module, simulator_hash = load_simulator(args.simulator_root)
    if simulator_hash != dataset["simulator_sha256"]:
        raise ValueError("Simulator differs from the one used to generate the training data.")
    torch.set_num_threads(args.cpu_threads)
    if not args.ground_truth_only and args.device == "cuda":
        if not torch.cuda.is_available() or args.gpu_id >= torch.cuda.device_count():
            raise RuntimeError("Requested CUDA GPU is unavailable.")
        args.device = f"cuda:{args.gpu_id}"
    model, checkpoint_hash = None, None
    if not args.ground_truth_only:
        manifest_path = args.run_dir / "halvorsen_ab_training.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest["generation_identity"] != dataset["generation_identity"]:
            raise ValueError("Model training coordinates belong to a different dataset.")
        model, _, checkpoint = load_saved_model(args.run_dir, args.device)
        if model.N != 3 or model.phi_dim != 2:
            raise ValueError("Expected three observations and two external parameters [a,b].")
        checkpoint_hash = file_hash(checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth, truth_metadata = ground_truth(args, config, module, simulator_hash)
    peaks = [np.empty(0) for _ in truth["pairs"]]
    equilibria = np.full(len(peaks), np.nan)
    failures = np.full(len(peaks), -1, dtype=np.int64)
    if model is not None:
        for start in tqdm(range(0, len(peaks), args.batch_size), desc="Model bifurcations"):
            stop = min(start + args.batch_size, len(peaks))
            predicted, failed = rollout_model(model, truth["starts"][start:stop], truth["pairs"][start:stop],
                                              mean, std, config, args.warmup_steps + args.observation_steps, args.device)
            failures[start:stop] = failed
            for local, index in enumerate(range(start, stop)):
                peaks[index], equilibria[index] = extrema(predicted[local, args.warmup_steps:])
    model_peaks, model_offsets = pack_ragged(peaks)
    orbit_arrays, metrics = trajectory_comparisons(args, config, model, mean, std)
    arrays = dict(**truth, model_peaks=model_peaks, model_offsets=model_offsets,
                  model_equilibria=equilibria, model_failed_at=failures, **orbit_arrays)
    metadata = dict(cache_version=CACHE_VERSION, label=args.report_label,
                    ground_truth_only=args.ground_truth_only, data_config=asdict(config),
                    ground_truth=truth_metadata, model_sha256=checkpoint_hash,
                    model_run=str(args.run_dir), data_dir=str(args.data_dir),
                    generation_identity=dataset["generation_identity"], state_mean=mean.tolist(), state_std=std.tolist(),
                    metrics=metrics, model_failures=int(np.count_nonzero(failures >= 0)),
                    orbit_convention="Same held-out observed seed at t=0; free model from t=1; discard warmup for plots and metrics.")
    save_cache(args.output_dir / "analysis.npz", arrays, metadata)
    (args.output_dir / "metrics.json").write_text(json.dumps(finite_json(metadata), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return arrays, metadata


def write_orbit_overview(arrays, metadata, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = arrays["orbit_pairs"]
    count_b = metadata["data_config"]["grid_b"]
    selected = list(dict.fromkeys([0, count_b - 1, len(pairs) // 2, len(pairs) - count_b, len(pairs) - 1]))
    fig = plt.figure(figsize=(16, 10))
    for panel, index in enumerate(selected, 1):
        ax = fig.add_subplot(2, 3, panel, projection="3d")
        for key, name, color in (("orbit_truth", "Ground truth", "#2563eb"), ("orbit_model", "ALRNN", "#e87926")):
            if key == "orbit_model" and metadata["ground_truth_only"]:
                continue
            orbit = arrays[key][index]
            ax.plot(*orbit.T, linewidth=.6, color=color, label=name, alpha=.8)
        a, b = pairs[index]
        ax.set(title=f"a={a:g}, b={b:g}", xlabel="x", ylabel="y", zlabel="z")
        ax.legend(fontsize=8)
    fig.suptitle(metadata["label"] + " | held-out test tails, physical coordinates")
    fig.tight_layout()
    fig.savefig(output / "trajectories.png", dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulator-root", type=Path, default=DEFAULT_ROOT, help=SIMULATOR_HELP)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--grid-a", type=int, default=41)
    parser.add_argument("--grid-b", type=int, default=41)
    parser.add_argument("--initial-states", type=int, default=2)
    parser.add_argument("--observation-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-plot-peaks", type=int, default=128)
    parser.add_argument("--ground-truth-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--report-label", default="Halvorsen: full-box a,b reconstruction")
    args = parser.parse_args(argv)
    if not args.plot_only and (args.data_dir is None or (not args.ground_truth_only and args.run_dir is None)):
        parser.error("Analysis requires --data-dir and, for model comparison, --run-dir.")
    if min(args.grid_a, args.grid_b, args.observation_steps) < 2 or args.warmup_steps < 0:
        parser.error("Grid/observation sizes must be >=2; warmup must be nonnegative.")
    if min(args.initial_states, args.cpu_threads, args.batch_size, args.max_plot_peaks) < 1 or args.gpu_id < 0:
        parser.error("Invalid initial-state count, thread count, batch size, peak limit or GPU index.")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.plot_only:
        arrays, metadata = read_cache(args.output_dir / "analysis.npz")
    else:
        arrays, metadata = analyze(args)
    from halvorsen_ab.report import write_report
    write_report(arrays, metadata, args.output_dir / "bifurcation.html", args.max_plot_peaks)
    write_orbit_overview(arrays, metadata, args.output_dir)
    print(f"Interactive report: {(args.output_dir / 'bifurcation.html').resolve()}", flush=True)


if __name__ == "__main__":
    main()
