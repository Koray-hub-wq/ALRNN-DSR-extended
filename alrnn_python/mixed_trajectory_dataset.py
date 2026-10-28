"""Uniform trajectory sampling for padded, variable-length mixed datasets."""

import numpy as np
import torch


class MixedTrajectoryDataset:
    def __init__(
        self,
        data,
        lengths,
        train_lengths,
        external_inputs=None,
        group_names=None,
        sequence_length=200,
        batch_size=16,
    ):
        data = np.asarray(data)
        if data.ndim != 3 or not data.shape[0] or not data.shape[2]:
            raise ValueError("data must have shape (trajectories, padded_time, channels)")
        self.num_trajectories = data.shape[0]
        self.lengths = self._validate_lengths(lengths, "lengths")
        self.train_lengths = self._validate_lengths(train_lengths, "train_lengths")
        if np.any(self.lengths > data.shape[1]):
            raise ValueError("lengths exceed the padded time dimension")
        if sequence_length < 1 or batch_size < 1:
            raise ValueError("sequence_length and batch_size must be positive")
        if np.any(self.train_lengths < sequence_length + 1):
            raise ValueError("Every training prefix needs at least sequence_length + 1 samples")
        self.test_lengths = self.lengths - self.train_lengths
        if np.any(self.test_lengths < 2):
            raise ValueError("Every trajectory must retain at least two test samples")
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_indices = np.arange(self.num_trajectories)
        if external_inputs is not None:
            external_inputs = np.asarray(external_inputs)
            if external_inputs.ndim != 3 or external_inputs.shape[:2] != data.shape[:2]:
                raise ValueError("external_inputs must match data's trajectory and time dimensions")
        for index, length in enumerate(self.lengths):
            if not np.isfinite(data[index, :length]).all():
                raise ValueError(f"Non-finite observations inside valid trajectory {index}")
            if external_inputs is not None and not np.isfinite(external_inputs[index, :length]).all():
                raise ValueError(f"Non-finite phi inside valid trajectory {index}")
        self.X_all = torch.as_tensor(data, dtype=torch.float32)
        self.S_all = None if external_inputs is None else torch.as_tensor(external_inputs, dtype=torch.float32)
        if group_names is None:
            group_names = [f"trajectory_{index}" for index in self.train_indices]
        self.group_names = np.asarray(group_names, dtype=str)
        if self.group_names.shape != (self.num_trajectories,):
            raise ValueError("group_names must contain one name per trajectory")
        self.eval_group_names = list(dict.fromkeys(self.group_names.tolist()))
        self.eval_indices_by_group = {
            name: np.flatnonzero(self.group_names == name) for name in self.eval_group_names
        }

    def _validate_lengths(self, lengths, name):
        values = np.asarray(lengths)
        if values.shape != (self.num_trajectories,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must be an integer array with one entry per trajectory")
        if np.any(values < 1):
            raise ValueError(f"{name} must be positive")
        return values.astype(np.int64)

    def __len__(self):
        return int(np.sum(self.train_lengths - self.sequence_length))

    def _sample_one(self, trajectory_index):
        # Include the final valid start: its shifted target ends at train_length.
        n_starts = self.train_lengths[trajectory_index] - self.sequence_length
        start = np.random.randint(0, n_starts)
        end = start + self.sequence_length
        x = self.X_all[trajectory_index, start:end]
        y = self.X_all[trajectory_index, start + 1:end + 1]
        phi = None if self.S_all is None else self.S_all[trajectory_index, start:end]
        return x, y, phi

    def sample_batch(self):
        # Each original trajectory has equal probability, regardless of its
        # length, parameter count, mode, or the number of blocks with that mode.
        indices = np.random.randint(0, self.num_trajectories, size=self.batch_size)
        samples = [self._sample_one(int(index)) for index in indices]
        x, y, phi = zip(*samples)
        return torch.stack(x), torch.stack(y), None if phi[0] is None else torch.stack(phi)

    def eval_reference(self, length=10000, trajectory_index=0):
        end = min(length, int(self.train_lengths[trajectory_index]))
        x = self.X_all[trajectory_index, :end]
        phi = None if self.S_all is None else self.S_all[trajectory_index, :end]
        return x, phi

    def test_trajectory(self, trajectory_index, length=None):
        start = int(self.train_lengths[trajectory_index])
        end = int(self.lengths[trajectory_index])
        if length is not None:
            end = min(end, start + length)
        x = self.X_all[trajectory_index, start:end]
        phi = None if self.S_all is None else self.S_all[trajectory_index, start:end]
        return x, phi
