"""Views of padded (v1) and concatenated, padding-free (v2) trajectories."""

import numpy as np


def trajectory_offsets(lengths):
    lengths = np.asarray(lengths)
    if lengths.ndim != 1 or not lengths.size or not np.issubdtype(lengths.dtype, np.integer) or np.any(lengths < 1):
        raise ValueError("Trajectory lengths must be a non-empty array of positive integers")
    return np.r_[np.int64(0), np.cumsum(lengths, dtype=np.int64)]


def validate_trajectory_array(array, lengths, offsets=None):
    offsets = trajectory_offsets(lengths) if offsets is None else offsets
    if array.ndim == 2:
        valid = array.shape[0] == offsets[-1] and array.shape[1] > 0
    elif array.ndim == 3:
        valid = array.shape[0] == len(lengths) and array.shape[1] >= max(lengths) and array.shape[2] > 0
    else:
        valid = False
    if not valid:
        raise ValueError("Trajectory array shape does not match saved lengths (expected padded 3D or compact 2D)")


def trajectory_view(array, index, lengths, offsets):
    if array.ndim == 2:
        return array[int(offsets[index]):int(offsets[index + 1])]
    return array[index, :int(lengths[index])]
