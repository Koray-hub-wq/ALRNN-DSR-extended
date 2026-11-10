"""Free Rossler-c comparisons across [3,6], including extrapolation checks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
from scipy.integrate import solve_ivp
import torch
from alrnn_model import load_saved_model
from compare_model_bifurcation_diagrams import load_training_coordinates
from lorenz_paper_test.evaluate_lorenz import model_tails, digest

DATASETS = ("rossler_c_4p5_to_5p5", "rossler_c_3_to_6")


def initial_states(seed):
    return np.random.default_rng(seed).normal((0., -5., .5), (1., 1., .2), size=(2, 3))


def rossler_rhs(time, state, c):
    x, y, z = state
    return np.array([-y-z, x+.2*y, .2+z*(x-c)])


def simulate_truth(parameters, initials, duration, dt):
    steps = int(round(duration/dt))
    if steps < 2 or not np.isclose(steps*dt, duration):
        raise ValueError("duration must be a multiple of dt and include at least two steps.")
    times = np.arange(steps+1)*dt
    values = []
    for c in parameters:
        for x0 in initials:
            solution = solve_ivp(lambda t, x: rossler_rhs(t, x, c), (0., times[-1]), x0, t_eval=times,
                                 method="DOP853", rtol=1e-7, atol=1e-9, max_step=5*dt)
            if not solution.success or solution.y.shape != (3, steps+1) or not np.isfinite(solution.y).all():
                raise RuntimeError(f"Ground truth failed at c={c}: {solution.message}")
            values.append(solution.y.T)
    return times, np.asarray(values)


def simulate_model(model, scaling, parameters, initials, duration, dt, device):
    steps = int(round(duration/dt))
    initial = (np.tile(initials, (len(parameters), 1))-scaling.mean)/scaling.std
    phi = (np.repeat(parameters, len(initials))-scaling.phi_boundary)/scaling.phi_scale
    result = np.full((len(initial), steps+1, 3), np.nan, dtype=np.float32)
    result[:, 0] = initial
    failures = np.full(len(initial), -1, dtype=np.int64)
    for start, stop, states, failed in model_tails(model, initial, phi, 1, steps-1, device, len(initial), progress=True):
        result[start:stop, 1:] = states
        failures[start:stop] = failed
    return result, failures


def plot_pair(folder, times, truth, model, parameters, initials, label, bounds, failures):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    folder.mkdir(parents=True, exist_ok=True)
    plots = []
    for p, c in enumerate(parameters):
        tag = f"c_{c:g}".replace(".", "p")
        scope = "interpolation/training range" if bounds[0] <= c <= bounds[1] else "EXTRAPOLATION"
        title = f"Cphi; {label}; c={c:g}; {scope}"
        figure, axes = plt.subplots(3, 4, figsize=(16, 8), sharex="col", constrained_layout=True)
        for ic in range(len(initials)):
            index = p*len(initials)+ic
            windows = ((0., min(5., times[-1])), (max(0., times[-1]-10.), times[-1]))
            for window, (low, high) in enumerate(windows):
                mask = (times >= low) & (times <= high)
                column = 2*ic+window
                for dim, component in enumerate("xyz"):
                    axis = axes[dim, column]
                    axis.plot(times[mask], truth[index, mask, dim], label="Ground truth", linewidth=.9)
                    axis.plot(times[mask], model[index, mask, dim], label="ALRNN", linewidth=.9, alpha=.85)
                    axis.set_ylabel(f"{component} (standardized)")
                    if dim == 2:
                        axis.set_xlabel("time")
                suffix = f"; non-finite at step {failures[index]}" if failures[index] >= 0 else ""
                axes[0, column].set_title(f"IC {ic+1}; " + ("initial transient" if window == 0 else "late orbit") + suffix, fontsize=9)
        axes[0, 0].legend()
        figure.suptitle(title)
        path = folder / f"{tag}_time_series.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        plots.append(str(path))

        figure = plt.figure(figsize=(12, 10), constrained_layout=True)
        norm = Normalize(0., times[-1])
        for ic in range(len(initials)):
            index = p*len(initials)+ic
            combined = np.concatenate((truth[index], model[index]))
            valid = combined[np.isfinite(combined).all(axis=1)]
            for col, (name, values) in enumerate((("Ground truth", truth), ("ALRNN", model))):
                axis = figure.add_subplot(2, 2, 2*ic+col+1, projection="3d")
                finite = np.isfinite(values[index]).all(axis=1)
                axis.scatter(*values[index, finite].T, c=times[finite], cmap="viridis", norm=norm, s=1.5, rasterized=True)
                axis.set(xlabel="x (std.)", ylabel="y (std.)", zlabel="z (std.)", title=f"{name}; IC {ic+1}")
                for dim, setter in enumerate((axis.set_xlim, axis.set_ylim, axis.set_zlim)):
                    low, high = valid[:, dim].min(), valid[:, dim].max()
                    margin = max(.05*(high-low), .05)
                    setter(low-margin, high+margin)
        figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=figure.axes, label="time", shrink=.65)
        figure.suptitle(title)
        path = folder / f"{tag}_3d.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        plots.append(str(path))
    return plots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--parameters", type=float, nargs="+", default=[3., 4., 4.5, 5., 5.5, 6.])
    parser.add_argument("--duration", type=float, default=50.)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not np.isfinite(args.duration) or args.duration <= 0 or not np.isfinite(args.parameters).all() or not all(3 <= c <= 6 for c in args.parameters):
        parser.error("Use a positive duration and parameters inside [3,6].")
    torch.set_num_threads(args.cpu_threads)
    folder = args.training_data_root / args.dataset
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
    dt = metadata["config"]["dt"]
    bounds = (min(m["parameter_min"] for m in metadata["config"]["modes"]),
              max(m["parameter_max"] for m in metadata["config"]["modes"]))
    initials = initial_states(args.seed)
    times, physical_truth = simulate_truth(args.parameters, initials, args.duration, dt)
    scaling = load_training_coordinates(folder)
    reference = load_training_coordinates(args.training_data_root / DATASETS[0])
    truth = (physical_truth-reference.mean)/reference.std
    run = args.run_root / "cphi" / args.dataset
    device = f"cuda:{args.gpu_id}" if args.device == "cuda" else "cpu"
    model, _, _ = load_saved_model(run, device)
    if model.N != 3 or model.phi_dim != 1 or model.phi_terms != ("additive",):
        raise ValueError("Expected a three-observation additive Cphi model.")
    values, failures = simulate_model(model, scaling, args.parameters, initials, args.duration, dt, device)
    learned = (values.astype(np.float64)*scaling.std+scaling.mean-reference.mean)/reference.std
    output = args.output_root / args.dataset
    paths = plot_pair(output, times, truth, learned, args.parameters, initials, args.dataset, bounds, failures)
    np.savez_compressed(output / "orbits.npz", time=times, parameters=np.repeat(args.parameters, len(initials)),
                        initial_states=initials, initial_state_index=np.tile(np.arange(len(initials)), len(args.parameters)),
                        ground_truth=truth.astype(np.float32), model=learned.astype(np.float32), first_non_finite_step=failures)
    provenance = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), dt=dt, duration=args.duration,
                      parameters=args.parameters, initial_states=initials.tolist(), seed=args.seed,
                      training_bounds=bounds, extrapolation_parameters=[c for c in args.parameters if not bounds[0] <= c <= bounds[1]],
                      method="Independent noiseless ground truth and free model evolution from the same physical IC; constant c, no teacher forcing or burn-in; standardized using the first dataset reference",
                      source_model=str(run), checkpoint_sha256=digest(run / "model.pt"),
                      metrics_sha256=digest(run / "metrics.json"), script_sha256=digest(__file__),
                      rollout_helper_sha256=digest(HERE.parent / "lorenz_paper_test/evaluate_lorenz.py"),
                      checkpoint_loader_sha256=digest(HERE.parent / "alrnn_model.py"),
                      model_coordinates=scaling.metadata(), reference_coordinates=reference.metadata(),
                      first_non_finite_step=failures.tolist(), plots=paths)
    (output / "trajectory_sources.json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Free Rossler-c comparisons: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
