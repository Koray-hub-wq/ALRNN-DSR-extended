"""Generate long, post-burn-in trajectories on a Cartesian (a,b) grid."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from halvorsen_ab.common import (DEFAULT_ROOT, HERE, SIMULATOR_HELP, file_hash, identity, load_config,
                                 load_simulator, simulate_with_retries)


def generation_identity(config, simulator_hash):
    return identity(dict(config=asdict(config), simulator_sha256=simulator_hash,
                         generator_sha256=file_hash(__file__),
                         common_sha256=file_hash(HERE / "common.py")))


def generate(config, output, simulator_root=DEFAULT_ROOT, progress=True):
    module, simulator_hash = load_simulator(simulator_root)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use a new or empty data folder: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pairs = config.pairs()
    constants = np.repeat(pairs, config.trajectories_per_pair, axis=0)
    count = len(constants)
    total = count * config.n_steps
    arrays = {name: np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=np.float32,
              shape=(total, 3 if name == "trajectories" else 2))
              for name in ("trajectories", "trajectory_phi", "trajectory_phi_raw")}
    rng = np.random.default_rng(config.seed)
    initial_states, records = [], []
    # Parallel variance merge: fit one fixed transform using training prefixes only.
    sample_count, mean, m2 = 0, np.zeros(3), np.zeros(3)
    for i, pair in enumerate(tqdm(constants, desc="Halvorsen (a,b)", disable=not progress)):
        states, initial, failures = simulate_with_retries(module, config, pair, rng)
        start = i * config.n_steps
        section = slice(start, start + config.n_steps)
        arrays["trajectories"][section] = states
        arrays["trajectory_phi_raw"][section] = pair
        arrays["trajectory_phi"][section] = config.normalize(pair)
        prefix = states[:config.train_length]
        n = len(prefix)
        local_mean = prefix.mean(axis=0)
        delta = local_mean - mean
        m2 += ((prefix - local_mean) ** 2).sum(axis=0) + delta ** 2 * sample_count * n / (sample_count + n)
        mean += delta * n / (sample_count + n)
        sample_count += n
        initial_states.append(initial)
        records.append(dict(trajectory_index=i, pair_index=i // config.trajectories_per_pair,
                            a=float(pair[0]), b=float(pair[1]), initial_state=initial.tolist(),
                            simulation_attempts=len(failures) + 1, failed_attempts=failures))
    std = np.maximum(np.sqrt(m2 / sample_count), 1e-8)
    for i in range(count):
        section = slice(i * config.n_steps, (i + 1) * config.n_steps)
        arrays["trajectories"][section] = (arrays["trajectories"][section] - mean) / std
    for value in arrays.values():
        value.flush()
    extras = dict(
        trajectory_lengths=np.full(count, config.n_steps, dtype=np.int64),
        trajectory_train_lengths=np.full(count, config.train_length, dtype=np.int64),
        trajectory_group_names=np.array([f"a_{a:.6g}_b_{b:.6g}" for a, b in constants]),
        trajectory_modes=np.array(["constant"] * count),
        trajectory_mean=np.broadcast_to(mean, (count, 1, 3)).astype(np.float32).copy(),
        trajectory_std=np.broadcast_to(std, (count, 1, 3)).astype(np.float32).copy(),
        trajectory_parameter_constants=constants,
        trajectory_initial_states=np.asarray(initial_states),
        trajectory_pair_index=np.repeat(np.arange(len(pairs)), config.trajectories_per_pair),
    )
    for name, value in extras.items():
        np.save(output / f"{name}.npy", value, allow_pickle=False)
    metadata = dict(
        format="alrnn_mixed_parameter_trajectories_v2", experiment="halvorsen_ab",
        parameter_names=["a", "b"], data_config=asdict(config),
        config=dict(standardization="global", noise_std_fraction=0.0, dt=config.dt),
        phi_reference=dict(boundary=config.center.tolist(), scale=config.scale.tolist()),
        phi_convention="Componentwise (raw - midpoint) / half_width, parameter order [a,b].",
        state_mean=mean.tolist(), state_std=std.tolist(),
        standardization_fit="All training prefixes only, before float32 storage; no test-tail statistics.",
        sample_convention="x[0] is post-burn-in; phi[t] drives x[t] -> x[t+1].",
        fixed_parameters=dict(c=config.c, drive=0.0),
        grid_pairs=pairs.tolist(), trajectories=records,
        simulator_sha256=simulator_hash,
        generation_identity=generation_identity(config, simulator_hash),
        shapes={name: list(value.shape) for name, value in arrays.items()},
    )
    # Completion marker is written last; interrupted generations cannot be reused.
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {count} trajectories, {len(pairs)} parameter pairs, phi_dim=2: {output}", flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulator-root", type=Path, default=DEFAULT_ROOT, help=SIMULATOR_HELP)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="5x5 pairs, one short trajectory per pair.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.smoke:
        config = replace(config, n_steps=400, burn_in_steps=20, trajectories_per_pair=1)
    if args.dry_run:
        count = len(config.pairs()) * config.trajectories_per_pair
        print(json.dumps(dict(config=asdict(config), grid_pairs=config.pairs().tolist(),
                              trajectories=count, stored_samples=count * config.n_steps,
                              array_bytes=count * config.n_steps * 7 * 4), indent=2))
        return
    generate(config, args.output_dir, args.simulator_root)


if __name__ == "__main__":
    main()
