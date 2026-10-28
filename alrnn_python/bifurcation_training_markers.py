"""Resolve constant training values or a time-varying training range for plots."""

from pathlib import Path

import numpy as np


class TrainingParameterRange(np.ndarray):
    """Two raw-parameter bounds, tagged as a filled region rather than lines.

    Keep NumPy compatibility because the external plotting backend logs its
    resolved training parameters with np.array2string.
    """

    def __new__(cls, low, high):
        return np.asarray([low, high], dtype=np.float64).view(cls)


def _lengths(data_dir, filename, count, default):
    path = data_dir / filename
    if not path.is_file():
        return np.full(count, default, dtype=np.int64)
    values = np.load(path, allow_pickle=False)
    if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{filename} must contain one integer length per trajectory")
    return values


def alrnn_training_parameters(data_dir, metadata, regime, settings):
    """Inspect actual training prefixes, including models without phi inputs.

    Return None when only legacy constant metadata is available, allowing the
    backend to keep its original discovery logic for that dataset.
    """
    data_dir = Path(data_dir)
    raw_path = data_dir / "trajectory_phi_raw.npy"
    phi_path = data_dir / "trajectory_phi.npy"
    if not raw_path.is_file() and not phi_path.is_file():
        return None
    regimes = metadata.get("regimes", [])
    matches = [item for item in regimes if item.get("name") == regime.name]
    if not matches:
        raise ValueError(f"Training dataset has no metadata for regime {regime.name!r}")
    definition = matches[0]
    reference = metadata.get("phi_reference", {})
    boundary = float(reference.get("boundary", definition["boundary"]))
    scale = float(reference.get("scale", max(
        abs(float(definition["start"]) - boundary),
        abs(float(definition["end"]) - boundary), 1e-8,
    )))
    if not np.isfinite([boundary, scale]).all() or scale <= 0:
        raise ValueError("Invalid training parameter normalization")

    raw_source = raw_path.is_file()
    array = np.load(raw_path if raw_source else phi_path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3 or array.shape[2] != 1 or not array.shape[0]:
        raise ValueError("Trajectory parameters must have shape (trajectories, time, 1)")
    count, padded_time, _ = array.shape
    lengths = _lengths(data_dir, "trajectory_lengths.npy", count, padded_time)
    train_path = data_dir / "trajectory_train_lengths.npy"
    if train_path.is_file():
        train_lengths = _lengths(data_dir, train_path.name, count, padded_time)
    else:
        train_lengths = np.minimum(lengths, int(settings.get("split_idx", padded_time)))

    keep = np.ones(count, dtype=bool)
    names_path = data_dir / "trajectory_regime_names.npy"
    if names_path.is_file():
        names = np.load(names_path, allow_pickle=False).astype(str)
        if names.shape != (count,):
            raise ValueError("Training regime labels do not match the parameter array")
        keep &= names == regime.name
    elif len(regimes) != 1:
        raise ValueError("Multiple training regimes require per-trajectory regime labels")

    single = settings.get("multi_trajectory") is False
    if single:
        trajectory_id = int(settings.get("trajectory_id", 0))
        if not 0 <= trajectory_id < count:
            raise ValueError("ALRNN trajectory_id is outside the training dataset")
        keep &= np.arange(count) == trajectory_id

    constants = []
    lower, upper, time_varying = np.inf, -np.inf, False
    for index in np.flatnonzero(keep):
        length, train_length = int(lengths[index]), int(train_lengths[index])
        if not 1 <= train_length <= length <= padded_time:
            raise ValueError(f"Invalid training/trajectory lengths for trajectory {index}")
        values = np.asarray(array[index, :train_length, 0], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite training parameters in trajectory {index}")
        if not raw_source:
            values = boundary + values * scale
        low, high = float(values.min()), float(values.max())
        constant = bool(np.isclose(low, high, atol=1e-6, rtol=0.0))
        if not single:
            requested_phi = settings.get("train_phi_values")
            requested_raw = settings.get("train_raw_parameter_values")
            if requested_phi is not None and (
                not constant or not np.isclose((low - boundary) / scale, requested_phi, atol=1e-5).any()
            ):
                continue
            if requested_raw is not None and (
                not constant or not np.isclose(low, requested_raw, atol=1e-5).any()
            ):
                continue
        lower, upper = min(lower, low), max(upper, high)
        time_varying |= not constant
        constants.append(float(values[0]))
    if not constants:
        raise ValueError("No training parameters remain for this regime and model selection")
    if time_varying:
        return TrainingParameterRange(lower, upper)
    unique = np.unique(constants)
    return unique[np.r_[True, ~np.isclose(unique[1:], unique[:-1], atol=1e-7, rtol=0.0)]]


def add_training_range(ax, bounds, parameter_values):
    """Shade the visible part of the training range behind bifurcation points."""
    low = max(float(bounds[0]), float(np.min(parameter_values)))
    high = min(float(bounds[1]), float(np.max(parameter_values)))
    if high <= low:
        return
    ax.axvspan(low, high, color="tab:blue", alpha=0.12, linewidth=0,
               zorder=0, label="Training parameter range")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)


def install_training_markers(backend):
    """Extend the existing backend without changing its simulation or caches."""
    original_values = backend.training_parameter_values
    original_markers = backend.add_training_markers

    def training_parameter_values(data_dir, metadata, regime, settings, model_type):
        if model_type == "alrnn":
            values = alrnn_training_parameters(data_dir, metadata, regime, settings)
            if values is not None:
                return values
        return original_values(data_dir, metadata, regime, settings, model_type)

    def add_training_markers(ax, values, parameter_values):
        if isinstance(values, TrainingParameterRange):
            add_training_range(ax, values, parameter_values)
        else:
            original_markers(ax, values, parameter_values)

    backend.training_parameter_values = training_parameter_values
    backend.add_training_markers = add_training_markers
