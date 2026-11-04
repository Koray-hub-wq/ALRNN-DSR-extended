"""Generate short, noiseless Lorenz trajectories with exactly paired initial states.

Uses the existing physical simulator and compact dataset schema without changing
the shared generator. Each distinct parameter schedule is integrated from both
exact initial states once, never from a perturbed or resampled initial state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
GENERATOR_DIR = WORKSPACE / "generate_trainingdata/mixed_parameter_trajectories"
sys.path.insert(0, str(GENERATOR_DIR))

import numpy as np
from tqdm import tqdm
from generate_mixed_parameter_trajectories import (
    GeneratorConfig, ModeSpec, fit_standardization, get_regime, phi_scale,
    simulate_trajectory, trajectory_plan,
)

INITIAL_STATES = np.array([[1., 1., 1.], [-1., -1., 1.]])
RECIPE_KEYS = {"output_root", "dt", "seed", "train_fraction", "standardization",
               "noise_std_fraction", "burn_in_steps", "n_steps", "initial_states",
               "n_constant_parameter_values", "n_dynamic_schedules_per_mode",
               "dynamic_min_span_fraction", "n_periods", "datasets"}


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or set(recipe) != RECIPE_KEYS:
        raise ValueError(f"Recipe must contain exactly these keys: {sorted(RECIPE_KEYS)}")
    if recipe["initial_states"] != INITIAL_STATES.tolist():
        raise ValueError("This experiment uses exactly (1,1,1) and (-1,-1,1).")
    if recipe["burn_in_steps"] != 0 or recipe["noise_std_fraction"] != 0:
        raise ValueError("This experiment requires burn_in_steps=0 and noise_std_fraction=0.")
    if recipe["standardization"] != "global":
        raise ValueError("Use global standardization for the shared bifurcation analysis.")
    for key, minimum in (("n_constant_parameter_values", 2), ("n_dynamic_schedules_per_mode", 1)):
        if type(recipe[key]) is not int or recipe[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    if not 0 < recipe["dynamic_min_span_fraction"] < 1:
        raise ValueError("dynamic_min_span_fraction must lie strictly between 0 and 1.")
    if not isinstance(recipe["output_root"], str) or not recipe["output_root"]:
        raise ValueError("output_root must be a non-empty folder name or path.")
    datasets = recipe["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list.")
    names = set()
    for dataset in datasets:
        if set(dataset) != {"name", "parameter_min", "parameter_max"}:
            raise ValueError("Each dataset needs name, parameter_min and parameter_max.")
        name = dataset["name"]
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", "..") or name in names:
            raise ValueError(f"Invalid or repeated dataset name: {name}")
        names.add(name)
        low, high = dataset["parameter_min"], dataset["parameter_max"]
        if not 1 <= low < high <= 30:
            raise ValueError("Lorenz parameter ranges must lie inside [1,30].")
        expand_config(recipe, dataset)


def expand_config(recipe, dataset):
    low, high = dataset["parameter_min"], dataset["parameter_max"]
    common = dict(parameter_space="raw", n_steps=recipe["n_steps"], burn_in_steps=0)
    count = recipe["n_constant_parameter_values"]
    modes = [ModeSpec("constant", low, high, n_trajectories=2*count,
                      n_parameter_values=count, **common)]
    # Distinct sweeps, each paired across the two initial states. Full-range
    # sweeps are included; other sweeps vary their upper endpoint.
    endpoints = np.linspace(low + (high-low)*recipe["dynamic_min_span_fraction"],
                            high, recipe["n_dynamic_schedules_per_mode"])
    for mode in ("sinusoidal", "linear_increasing", "linear_decreasing"):
        for endpoint in endpoints:
            modes.append(ModeSpec(mode, low, float(endpoint), n_trajectories=2,
                                 n_periods=recipe["n_periods"] if mode == "sinusoidal" else None,
                                 **common))
    config = GeneratorConfig(regime="lorenz_rho_1_to_30", modes=modes,
                             dt=recipe["dt"], seed=recipe["seed"],
                             train_fraction=recipe["train_fraction"],
                             standardization="global", noise_std_fraction=0.,
                             max_simulation_attempts=1)
    train = int(recipe["n_steps"]*recipe["train_fraction"])
    if train < 2 or recipe["n_steps"]-train < 2:
        raise ValueError("Keep at least two training and two test samples per trajectory.")
    return config


def paired_plan(config):
    plan = trajectory_plan(config)
    seen = {}
    for record in plan:
        group = record["group"]
        initial = seen.get(group, 0)
        if initial >= 2:
            raise ValueError(f"Expected exactly two trajectories per schedule: {group}")
        record.update(initial_state_index=initial, initial_state=INITIAL_STATES[initial].tolist())
        seen[group] = initial+1
    if any(count != 2 for count in seen.values()):
        raise ValueError("Every schedule must contain both initial states.")
    return plan


def generation_identity(config):
    # Include the imported simulator as well: changed equations must not reuse
    # a previously completed dataset silently.
    sources = [Path(__file__), GENERATOR_DIR / "generate_mixed_parameter_trajectories.py",
               GENERATOR_DIR.parent / "time_varying_bifurcation_data/generate_dynamic_regime_trajectories.py"]
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    settings = dict(config=asdict(config), initial_states=INITIAL_STATES.tolist(), sources=source_hashes)
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def generate(config, output, progress=True):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output folder must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    regime = get_regime(config.regime)
    plan = paired_plan(config)
    lengths = np.array([r["n_steps"] for r in plan], dtype=np.int64)
    offsets = np.r_[0, lengths.cumsum()]
    shape = (int(offsets[-1]), 3)
    arrays = {name: np.lib.format.open_memmap(output / f"{name}.npy", mode="w+",
              dtype=np.float32, shape=shape if name == "trajectories" else (shape[0], 1))
              for name in ("trajectories", "trajectory_phi", "trajectory_phi_raw")}
    rng = np.random.default_rng(config.seed)
    for record in tqdm(plan, desc=output.name, disable=not progress):
        i = record["trajectory_index"]
        exact_regime = replace(regime, init_center=tuple(record["initial_state"]), init_scale=(0., 0., 0.))
        try:
            values = simulate_trajectory(config, config.modes[record["block_index"]],
                                         exact_regime, rng, record["constant_value"])
        except RuntimeError as error:
            raise RuntimeError(f"Trajectory {i}, exact initial state {record['initial_state']}: {error}. No resampling.") from error
        for array, value in zip(arrays.values(), values):
            array[offsets[i]:offsets[i+1]] = value
    means, stds = fit_standardization(arrays["trajectories"], plan, "global")
    for i in range(len(plan)):
        section = slice(offsets[i], offsets[i+1])
        arrays["trajectories"][section] = (arrays["trajectories"][section]-means[i])/stds[i]
    for array in arrays.values():
        array.flush()
    extras = dict(trajectory_lengths=lengths,
                  trajectory_train_lengths=np.array([r["train_length"] for r in plan], dtype=np.int64),
                  trajectory_block_index=np.array([r["block_index"] for r in plan], dtype=np.int64),
                  trajectory_modes=np.array([r["mode"] for r in plan]),
                  trajectory_group_names=np.array([r["group"] for r in plan]),
                  trajectory_regime_index=np.zeros(len(plan), dtype=np.int64),
                  trajectory_regime_names=np.array([regime.name]*len(plan)),
                  trajectory_mean=means.astype(np.float32), trajectory_std=stds.astype(np.float32),
                  trajectory_initial_state_index=np.array([r["initial_state_index"] for r in plan], dtype=np.int8),
                  trajectory_initial_states=np.array([r["initial_state"] for r in plan], dtype=np.float32))
    for name, array in extras.items():
        np.save(output / f"{name}.npy", array, allow_pickle=False)
    # The nominal historical Regime is retained for equation/phi metadata only;
    # record the actual deterministic initial-state policy explicitly.
    regime_info = asdict(regime)
    regime_info.update(init_center=None, init_scale=(0., 0., 0.), initial_states=INITIAL_STATES.tolist())
    metadata = dict(format="alrnn_mixed_parameter_trajectories_v2", config=asdict(config),
                    regimes=[regime_info], phi_reference=dict(boundary=regime.boundary, scale=phi_scale(regime)),
                    generation_identity=generation_identity(config),
                    layout="Compact concatenated trajectories, offsets=[0,cumsum(trajectory_lengths)]; no padding",
                    standardization_fit="Training prefixes only; no observation noise",
                    initial_state_policy=dict(kind="exact_paired", initial_states=INITIAL_STATES.tolist(),
                        fraction_each=0.5, perturbation_std=0., resampling=False,
                        pairing="Exactly one trajectory from each initial state for every distinct parameter schedule"),
                    trajectories=plan, shapes={name: list(array.shape) for name, array in arrays.items()})
    temporary = output / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(output / "metadata.json")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--restart-incomplete", action="store_true",
                        help="Preserve incomplete folders under .incomplete and regenerate them; no trajectory checkpointing.")
    args = parser.parse_args(argv)
    recipe = json.loads(args.config.read_text(encoding="utf-8-sig"))
    validate_recipe(recipe)
    output_root = args.output_root or GENERATOR_DIR / "datasets" / recipe["output_root"]
    for dataset in recipe["datasets"]:
        config = expand_config(recipe, dataset)
        plan = paired_plan(config)
        output = output_root / dataset["name"]
        print(f"{dataset['name']}: {len(plan)} x {recipe['n_steps']} steps; "
              f"train/test {plan[0]['train_length']}/{recipe['n_steps']-plan[0]['train_length']}; "
              f"exact ICs 50/50; output: {output}", flush=True)
        if args.dry_run:
            continue
        marker = output / "metadata.json"
        if marker.is_file() and args.skip_completed:
            saved = json.loads(marker.read_text(encoding="utf-8-sig"))
            if saved.get("generation_identity") != generation_identity(config):
                raise ValueError(f"Completed dataset differs from this generation configuration/code: {output}")
            for name in ("trajectories", "trajectory_phi", "trajectory_phi_raw", "trajectory_lengths",
                         "trajectory_train_lengths", "trajectory_mean", "trajectory_std", "trajectory_group_names",
                         "trajectory_modes", "trajectory_initial_states", "trajectory_initial_state_index"):
                if not (output / f"{name}.npy").is_file():
                    raise FileNotFoundError(f"Completed dataset is missing {name}.npy: {output}")
            print("  Reusing completed dataset.", flush=True)
            continue
        if output.exists() and any(output.iterdir()) and not marker.exists() and args.restart_incomplete:
            archive = output_root / ".incomplete" / f"{output.name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
            # Both paths are fixed children of the explicitly selected output root.
            if output.resolve().parent != output_root.resolve() or archive.resolve().parent.parent != output_root.resolve():
                raise ValueError("Archive paths must remain inside output_root.")
            archive.parent.mkdir(parents=True, exist_ok=True)
            output.rename(archive)
            print(f"  Preserved incomplete generation: {archive}", flush=True)
        generate(config, output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
