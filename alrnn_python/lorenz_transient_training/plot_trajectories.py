"""Independent free-orbit checks from both exact Lorenz initial states.

For each trained model, compare early/late x,y,z time traces and time-colored
3D orbits. These are new simulations, not the tiny withheld training tails.
"""
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
from lorenz_paper_test import evaluate_lorenz as paper


def free_orbits(model, scaling, parameters, duration, dt, device):
    """Return normalized observations, including the exact observed t=0 state."""
    count = len(parameters)*2
    steps = int(round(duration/dt))
    if steps < 2 or not np.isclose(steps*dt, duration):
        raise ValueError("duration must be a multiple of dt and contain at least two steps.")
    initial = (np.tile(paper.INITIAL_STATES, (len(parameters), 1))-scaling.mean)/scaling.std
    phi = (np.repeat(parameters, 2)-scaling.phi_boundary)/scaling.phi_scale
    result = np.full((count, steps+1, 3), np.nan, dtype=np.float32)
    result[:, 0] = initial
    failures = np.full(count, -1, dtype=np.int64)
    # model_tails retains steps warmup..warmup+observation, inclusively.
    # With warmup=1 it returns all states t=dt..duration; prepend t=0 above.
    for start, stop, states, failed in paper.model_tails(model, initial, phi, 1, steps-1,
                                                       device, batch_size=count, progress=True):
        result[start:stop, 1:] = states
        failures[start:stop] = failed
    return result, failures


def true_orbits(parameters, duration, dt):
    steps = int(round(duration/dt))
    times = np.arange(steps+1)*dt
    values = []
    for rho in parameters:
        for initial in paper.INITIAL_STATES:
            solution = solve_ivp(lambda t, state: paper.lorenz_rhs(t, state, rho),
                                 (0., times[-1]), initial, t_eval=times,
                                 method="DOP853", rtol=1e-7, atol=1e-9, max_step=5*dt)
            if not solution.success or solution.y.shape != (3, steps+1) or not np.isfinite(solution.y).all():
                raise RuntimeError(f"Ground truth failed at rho={rho}, initial={initial}: {solution.message}")
            values.append(solution.y.T)
    return times, np.asarray(values)


def plot_pair(folder, times, truth, learned, parameters, label, failures, dpi=150):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for parameter_index, rho in enumerate(parameters):
        tag = f"rho_{rho:g}".replace(".", "p")
        figure, axes = plt.subplots(3, 4, figsize=(16, 8), sharex="col", constrained_layout=True)
        for initial in range(2):
            index = parameter_index*2+initial
            windows = ((0., min(5., times[-1])), (max(0., times[-1]-10.), times[-1]))
            for window, (low, high) in enumerate(windows):
                mask = (times >= low) & (times <= high)
                column = 2*initial+window
                for dim, component in enumerate("xyz"):
                    axis = axes[dim, column]
                    axis.plot(times[mask], truth[index, mask, dim], label="Ground truth", linewidth=.9)
                    axis.plot(times[mask], learned[index, mask, dim], label="ALRNN", linewidth=.9, alpha=.85)
                    axis.set_ylabel(f"{component} (standardized)")
                    if dim == 2:
                        axis.set_xlabel("time")
                title = f"IC {paper.INITIAL_STATES[initial].astype(int).tolist()}; " + ("initial transient" if window == 0 else "late orbit")
                if failures[index] >= 0:
                    title += f"; non-finite at step {failures[index]}"
                axes[0, column].set_title(title, fontsize=9)
        axes[0, 0].legend()
        figure.suptitle(f"{label}; rho={rho:g}; free evolution from t=0")
        path = folder / f"{tag}_time_series.png"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        paths.append(str(path))

        figure = plt.figure(figsize=(12, 10), constrained_layout=True)
        norm = Normalize(0., times[-1])
        for initial in range(2):
            index = parameter_index*2+initial
            for column, (name, values) in enumerate((("Ground truth", truth), ("ALRNN", learned))):
                axis = figure.add_subplot(2, 2, 2*initial+column+1, projection="3d")
                finite = np.isfinite(values[index]).all(axis=1)
                axis.scatter(*values[index, finite].T, c=times[finite], cmap="viridis", norm=norm, s=1.5, rasterized=True)
                axis.set(xlabel="x (std.)", ylabel="y (std.)", zlabel="z (std.)",
                         title=f"{name}; IC {paper.INITIAL_STATES[initial].astype(int).tolist()}")
                # Shared coordinate limits per row allow direct comparison.
                combined = np.concatenate((truth[index], learned[index]))
                valid = combined[np.isfinite(combined).all(axis=1)]
                for dim, setter in enumerate((axis.set_xlim, axis.set_ylim, axis.set_zlim)):
                    low, high = valid[:, dim].min(), valid[:, dim].max()
                    margin = max(.05*(high-low), .05)
                    setter(low-margin, high+margin)
        figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=figure.axes, label="time", shrink=.65)
        figure.suptitle(f"{label}; rho={rho:g}; independent free orbits")
        path = folder / f"{tag}_3d.png"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        paths.append(str(path))
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(paper.VARIANTS), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--duration", type=float, default=50.)
    parser.add_argument("--parameters", type=float, nargs="+", default=[5., 14., 23., 28.])
    args = parser.parse_args(argv)
    if not np.isfinite(args.duration) or args.duration <= 0 or not np.isfinite(args.parameters).all() or not all(1 <= rho <= 30 for rho in args.parameters):
        parser.error("Use a positive finite duration and parameters inside [1,30].")
    torch.set_num_threads(args.cpu_threads)
    device = f"cuda:{args.gpu_id}" if args.device == "cuda" else "cpu"
    dt_values = []
    scales = []
    for name in paper.DATASETS:
        folder = args.training_data_root / name
        dt_values.append(json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))["config"]["dt"])
        scales.append(paper.load_training_coordinates(folder))
    if not np.allclose(dt_values, dt_values[0], rtol=1e-12, atol=0):
        raise ValueError("Dataset sampling intervals must match.")
    dt = dt_values[0]
    if not np.isclose(round(args.duration/dt)*dt, args.duration) or round(args.duration/dt) < 2:
        raise ValueError("duration must be a multiple of dt and include at least two steps.")
    times, physical_truth = true_orbits(args.parameters, args.duration, dt)
    reference = scales[0]
    truth = (physical_truth-reference.mean)/reference.std
    provenance = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                      script_sha256=paper.digest(__file__),
                      rollout_helper_sha256=paper.digest(paper.__file__),
                      checkpoint_loader_sha256=paper.digest(HERE.parent / "alrnn_model.py"),
                      dt=dt, duration=args.duration, parameters=args.parameters,
                      initial_states=paper.INITIAL_STATES.tolist(),
                      method="Independent free evolution from exact physical initial states; no burn-in or teacher forcing; same training-coordinate reference as bifurcations.",
                      reference_coordinates=reference.metadata(), models=[])
    folder = args.output_root / args.variant
    folder.mkdir(parents=True, exist_ok=True)
    for name, scaling in zip(paper.DATASETS, scales):
        run = args.run_root / args.variant / name
        model, _, _ = paper.load_saved_model(run, device)
        if model.N != 3 or model.phi_dim != 1:
            raise ValueError(f"Expected three observations and one parameter: {run}")
        values, failures = free_orbits(model, scaling, args.parameters, args.duration, dt, device)
        learned = (values.astype(np.float64)*scaling.std+scaling.mean-reference.mean)/reference.std
        destination = folder / name
        paths = plot_pair(destination, times, truth, learned, args.parameters,
                          f"{paper.VARIANTS[args.variant]}; {name}", failures)
        np.savez_compressed(destination / "orbits.npz", time=times, parameters=np.repeat(args.parameters, 2),
                            initial_state_index=np.tile([0, 1], len(args.parameters)),
                            ground_truth=truth.astype(np.float32), model=learned.astype(np.float32),
                            first_non_finite_step=failures)
        provenance["models"].append(dict(**paper.model_source(run, scaling), plots=paths,
                                         first_non_finite_step=failures.tolist()))
        del model
    (folder / "trajectory_sources.json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Free trajectory comparisons: {folder}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
