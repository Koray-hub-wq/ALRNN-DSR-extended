"""Shared simulation, parameter coordinates and dataset validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
ALRNN_DIR = HERE.parent
if str(ALRNN_DIR) not in sys.path:
    sys.path.insert(0, str(ALRNN_DIR))
from halvorsen_ab.sync_simulator import DEFAULT_ROOT, SIMULATOR_HELP, simulator_paths


@dataclass(frozen=True)
class DataConfig:
    a_min: float = 1.4
    a_max: float = 2.1
    b_min: float = 2.0
    b_max: float = 4.0
    c: float = 4.0
    grid_a: int = 5
    grid_b: int = 5
    trajectories_per_pair: int = 8
    n_steps: int = 100000
    burn_in_steps: int = 5000
    dt: float = 0.01
    train_fraction: float = 0.8
    seed: int = 8
    rtol: float = 1e-8
    atol: float = 1e-10
    max_step_multiplier: float = 5.0
    max_simulation_attempts: int = 10
    escape_radius: float = 10000.0

    def __post_init__(self):
        for name, minimum in (("grid_a", 2), ("grid_b", 2), ("trajectories_per_pair", 1),
                              ("n_steps", 4), ("burn_in_steps", 0), ("seed", 0),
                              ("max_simulation_attempts", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{field.name} must be finite and numeric.")
        for name in ("dt", "rtol", "atol", "max_step_multiplier", "escape_radius"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not self.a_min < self.a_max or not self.b_min < self.b_max:
            raise ValueError("Each parameter interval must have positive width.")
        if not 0 < self.train_fraction < 1 or not 2 <= self.train_length <= self.n_steps - 2:
            raise ValueError("Keep at least two training and two test samples per trajectory.")

    @property
    def train_length(self):
        return int(self.n_steps * self.train_fraction)

    @property
    def center(self):
        return np.array([(self.a_min + self.a_max) / 2, (self.b_min + self.b_max) / 2])

    @property
    def scale(self):
        return np.array([(self.a_max - self.a_min) / 2, (self.b_max - self.b_min) / 2])

    def pairs(self, grid_a=None, grid_b=None):
        a = np.linspace(self.a_min, self.a_max, self.grid_a if grid_a is None else grid_a)
        b = np.linspace(self.b_min, self.b_max, self.grid_b if grid_b is None else grid_b)
        aa, bb = np.meshgrid(a, b, indexing="ij")
        return np.column_stack((aa.ravel(), bb.ravel()))

    def normalize(self, raw):
        return (np.asarray(raw, dtype=np.float64) - self.center) / self.scale


def load_config(path=HERE / "config.json"):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return DataConfig(**data)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_simulator(root=DEFAULT_ROOT):
    root = Path(root).expanduser().resolve()
    paths = simulator_paths(root)
    if any(path.read_bytes() != paths[0].read_bytes() for path in paths[1:]):
        raise ValueError("The two generate_dynamic_regime_trajectories.py files are not synchronized.")
    spec = importlib.util.spec_from_file_location("halvorsen_ab_simulator", paths[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not {"b", "c"} <= set(inspect.signature(module.halvorsen_rhs).parameters):
        raise ValueError(
            f"Simulator {paths[0]} lacks b/c arguments. Update it or run sync_simulator.py "
            f"--simulator-root {str(root)!r} --apply first."
        )
    print(f"Using simulator: {paths[0]}", flush=True)
    return module, file_hash(paths[0])


def simulate(module, config, pair, initial_state, n_steps=None, burn_in_steps=None):
    """Return x[0] after burn-in, then samples on the exact dt grid; never clip dx."""
    a, b = np.asarray(pair, dtype=np.float64)
    n_steps = config.n_steps if n_steps is None else n_steps
    burn_in_steps = config.burn_in_steps if burn_in_steps is None else burn_in_steps
    state = np.asarray(initial_state, dtype=np.float64)
    if state.shape != (3,) or not np.isfinite(state).all():
        raise ValueError("Initial state must contain three finite values.")

    def rhs(t, x):
        return module.halvorsen_rhs(t, x, float(a), b=float(b), c=config.c, drive=0.0)

    def escape(t, x):
        return config.escape_radius - np.max(np.abs(x))

    escape.terminal = True
    escape.direction = -1
    options = dict(method="DOP853", rtol=config.rtol, atol=config.atol,
                   max_step=config.dt * config.max_step_multiplier, events=escape)
    if burn_in_steps:
        end = burn_in_steps * config.dt
        burn = solve_ivp(rhs, (0.0, end), state, t_eval=[end], **options)
        if not burn.success or np.asarray(burn.y).shape != (3, 1) or not np.isfinite(burn.y).all():
            raise RuntimeError(f"Burn-in failed at a={a:g}, b={b:g}: {burn.message}")
        state = burn.y[:, -1]
    times = np.arange(n_steps, dtype=np.float64) * config.dt
    sol = solve_ivp(rhs, (0.0, times[-1]), state, t_eval=times, **options)
    if not sol.success or np.asarray(sol.y).shape != (3, n_steps) or not np.isfinite(sol.y).all():
        raise RuntimeError(f"Trajectory failed at a={a:g}, b={b:g}: {sol.message}")
    return sol.y.T


def simulate_with_retries(module, config, pair, rng, **kwargs):
    failures = []
    for attempt in range(config.max_simulation_attempts):
        initial = rng.normal(loc=(-4.0, 0.0, 0.0), scale=(0.8, 0.8, 0.8))
        try:
            result = simulate(module, config, pair, initial, **kwargs)
            return result, initial, failures
        except RuntimeError as exc:
            failures.append(dict(initial_state=initial.tolist(), reason=str(exc)))
    raise RuntimeError(f"All {len(failures)} initial states failed for {list(pair)}: {failures[-1]}")


def load_dataset_metadata(folder):
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
    if metadata.get("experiment") != "halvorsen_ab" or metadata.get("parameter_names") != ["a", "b"]:
        raise ValueError(f"Expected a completed two-parameter Halvorsen dataset: {folder}")
    config = DataConfig(**metadata["data_config"])
    reference = metadata["phi_reference"]
    if not np.allclose(reference["boundary"], config.center) or not np.allclose(reference["scale"], config.scale):
        raise ValueError("Saved phi coordinates disagree with the data configuration.")
    mean, std = np.asarray(metadata["state_mean"]), np.asarray(metadata["state_std"])
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite([mean, std]).all() or np.any(std <= 0):
        raise ValueError("Invalid global training coordinates.")
    return config, mean, std, metadata
