"""Sampling of mixed trajectories, optionally balancing constants and time sections."""

import numpy as np
import torch

from trajectory_storage import trajectory_offsets, trajectory_view, validate_trajectory_array


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
        trajectory_sampling="length",
        balanced_sampling=False,
        time_sampling_bins=10,
        sampling_parameters=None,
        trajectory_modes=None,
    ):
        data = np.asarray(data)
        self.num_trajectories = len(lengths)
        self.lengths = self._validate_lengths(lengths, "lengths")
        self.offsets = trajectory_offsets(self.lengths)
        validate_trajectory_array(data, self.lengths, self.offsets)
        self.train_lengths = self._validate_lengths(train_lengths, "train_lengths")
        if sequence_length < 1 or batch_size < 1:
            raise ValueError("sequence_length and batch_size must be positive")
        if np.any(self.train_lengths < sequence_length + 1):
            raise ValueError("Every training prefix needs at least sequence_length + 1 samples")
        self.test_lengths = self.lengths - self.train_lengths
        if np.any(self.test_lengths < 2):
            raise ValueError("Every trajectory must retain at least two test samples")
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        if trajectory_sampling not in ("length", "uniform"):
            raise ValueError("trajectory_sampling must be 'length' or 'uniform'")
        self.trajectory_sampling = trajectory_sampling
        if trajectory_sampling == "length":
            weights = self.train_lengths.astype(np.float64)
            self.sampling_strategy = "proportional_to_training_length"
        else:
            weights = np.ones(self.num_trajectories, dtype=np.float64)
            self.sampling_strategy = "uniform_per_trajectory"
        self.trajectory_probabilities = weights / weights.sum()
        self.base_trajectory_probabilities = self.trajectory_probabilities.copy()
        self.train_indices = np.arange(self.num_trajectories)
        if external_inputs is not None:
            external_inputs = np.asarray(external_inputs)
            validate_trajectory_array(external_inputs, self.lengths, self.offsets)
            if external_inputs.ndim != data.ndim or external_inputs.shape[:-1] != data.shape[:-1]:
                raise ValueError("external_inputs must match data's trajectory and time dimensions")
        for index, length in enumerate(self.lengths):
            if not np.isfinite(trajectory_view(data, index, self.lengths, self.offsets)).all():
                raise ValueError(f"Non-finite observations inside valid trajectory {index}")
            if external_inputs is not None and not np.isfinite(trajectory_view(external_inputs, index, self.lengths, self.offsets)).all():
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
        self.balanced_sampling = bool(balanced_sampling)
        if (isinstance(time_sampling_bins, (bool, np.bool_))
                or not isinstance(time_sampling_bins, (int, np.integer)) or time_sampling_bins < 1):
            raise ValueError("time_sampling_bins must be a positive integer")
        self.time_sampling_bins = int(time_sampling_bins)
        self.constant_mask = np.zeros(self.num_trajectories, dtype=bool)
        self.constant_phi_values = np.empty(0, dtype=np.float64)
        self.constant_parameter_pools = []
        self._constant_pool_probabilities = []
        self.effective_time_bins = np.zeros(self.num_trajectories, dtype=np.int64)
        self._time_bin_orders = [np.empty(0, dtype=np.int64) for _ in self.train_indices]
        self._time_bin_positions = np.zeros(self.num_trajectories, dtype=np.int64)
        self.sampled_trajectory_counts = np.zeros(self.num_trajectories, dtype=np.int64)
        if self.balanced_sampling:
            parameters = external_inputs if sampling_parameters is None else np.asarray(sampling_parameters)
            self._configure_balancing(parameters, trajectory_modes)
            self.sampling_strategy += "_balanced_constants_and_time_sections"
        self.sampled_time_bin_counts = [np.zeros(n, dtype=np.int64) for n in self.effective_time_bins]

    def _configure_balancing(self, parameters, trajectory_modes):
        if parameters is None:
            raise ValueError("Balanced mixed sampling requires phi arrays, also for models without phi inputs")
        validate_trajectory_array(parameters, self.lengths, self.offsets)
        phi_dim = parameters.shape[-1]
        modes = None if trajectory_modes is None else np.asarray(trajectory_modes, dtype=str)
        if modes is not None and modes.shape != (self.num_trajectories,):
            raise ValueError("trajectory_modes must contain one mode per trajectory")
        first_values = np.empty((self.num_trajectories, phi_dim), dtype=np.float64)
        for index, train_length in enumerate(self.train_lengths):
            values = trajectory_view(parameters, index, self.lengths, self.offsets)[:int(train_length)]
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite sampling parameters in training trajectory {index}")
            first_values[index] = values[0]
            constant = bool(np.all(np.isclose(values.min(axis=0), values.max(axis=0), atol=1e-6, rtol=0.0)))
            if modes is not None:
                if modes[index] == "constant" and not constant:
                    raise ValueError(f"Trajectory {index} is labelled constant but its training parameter varies")
                constant = modes[index] == "constant"
            self.constant_mask[index] = constant
        # Merge equal parameter values across repeated constant blocks. The
        # tolerance only collapses float32 rounding differences, not time bins.
        ordered = np.flatnonzero(self.constant_mask)
        ordered = ordered[np.lexsort(first_values[ordered].T[::-1])]
        values, pools = [], []
        for index in ordered:
            # Near-equal vectors need not be adjacent in lexicographic order:
            # (0,0), (0,1), (epsilon,0) must put the first and last in one pool.
            candidates = (range(max(0, len(values) - 1), len(values)) if phi_dim == 1 else range(len(values)))
            position = next((i for i in candidates
                             if np.allclose(first_values[index], values[i], atol=1e-6, rtol=0.0)), None)
            if position is None:
                values.append(first_values[index].copy())
                pools.append([])
                position = len(pools) - 1
            pools[position].append(int(index))
        self.constant_phi_values = np.asarray(values, dtype=np.float64).reshape(-1, phi_dim)
        # Preserve the existing scalar metadata/API for one-parameter datasets.
        if phi_dim == 1:
            self.constant_phi_values = self.constant_phi_values[:, 0]
        self.constant_parameter_pools = [np.asarray(pool, dtype=np.int64) for pool in pools]
        constant_mass = self.base_trajectory_probabilities[self.constant_mask].sum()
        for pool in self.constant_parameter_pools:
            probabilities = self.base_trajectory_probabilities[pool]
            probabilities = probabilities / probabilities.sum()
            self._constant_pool_probabilities.append(probabilities)
            self.trajectory_probabilities[pool] = constant_mass / len(pools) * probabilities
        # Stratify valid starts, so every shifted target stays in the training
        # prefix. A very short prefix gets fewer, non-empty sections.
        dynamic = ~self.constant_mask
        self.effective_time_bins[dynamic] = np.minimum(
            self.time_sampling_bins, self.train_lengths[dynamic] - self.sequence_length
        )

    def sampling_metadata(self):
        return {
            "balanced": self.balanced_sampling,
            "trajectory_weighting": self.trajectory_sampling,
            "base_trajectory_probabilities": self.base_trajectory_probabilities.tolist(),
            "constant_balance": "equal_counts_per_batch_random_remainder" if self.balanced_sampling else None,
            "constant_phi_values": self.constant_phi_values.tolist(),
            "constant_trajectory_mask": self.constant_mask.tolist(),
            "time_sampling": "shuffled_cycles_per_dynamic_trajectory" if self.balanced_sampling else "uniform_valid_start",
            "requested_time_bins": self.time_sampling_bins if self.balanced_sampling else None,
            "effective_time_bins": self.effective_time_bins.tolist(),
            "time_bin_domain": "valid_window_starts_inside_training_prefix",
            "sampled_trajectory_counts": self.sampled_trajectory_counts.tolist(),
            "sampled_time_bin_counts": [counts.tolist() for counts in self.sampled_time_bin_counts],
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

    def _sample_start(self, trajectory_index):
        n_starts = int(self.train_lengths[trajectory_index] - self.sequence_length)
        n_bins = int(self.effective_time_bins[trajectory_index])
        if not n_bins:
            return np.random.randint(0, n_starts)
        position = int(self._time_bin_positions[trajectory_index])
        if position == len(self._time_bin_orders[trajectory_index]):
            self._time_bin_orders[trajectory_index] = np.random.permutation(n_bins)
            position = 0
        section = int(self._time_bin_orders[trajectory_index][position])
        self._time_bin_positions[trajectory_index] = position + 1
        self.sampled_time_bin_counts[trajectory_index][section] += 1
        low, high = section * n_starts // n_bins, (section + 1) * n_starts // n_bins
        return np.random.randint(low, high)

    def _sample_one(self, trajectory_index):
        # Include the final valid start: its shifted target ends at train_length.
        start = self._sample_start(trajectory_index)
        end = start + self.sequence_length
        states = trajectory_view(self.X_all, trajectory_index, self.lengths, self.offsets)
        x = states[start:end]
        y = states[start + 1:end + 1]
        phi = None if self.S_all is None else trajectory_view(self.S_all, trajectory_index, self.lengths, self.offsets)[start:end]
        return x, y, phi

    def _sample_trajectory_indices(self):
        # Weight by training samples, excluding test tails and NaN padding.
        # Retain the original RNG calls for reproducible uniform-sampling runs.
        if self.trajectory_sampling == "length":
            indices = np.random.choice(
                self.num_trajectories, size=self.batch_size, replace=True,
                p=self.base_trajectory_probabilities,
            )
        else:
            indices = np.random.randint(0, self.num_trajectories, size=self.batch_size)
        if self.balanced_sampling and self.constant_parameter_pools:
            # Preserve exactly the originally drawn constant count and every
            # dynamic choice. Only redistribute the constant slots by value.
            positions = np.flatnonzero(self.constant_mask[indices])
            count, remainder = divmod(len(positions), len(self.constant_parameter_pools))
            counts = np.full(len(self.constant_parameter_pools), count, dtype=np.int64)
            counts[np.random.permutation(len(counts))[:remainder]] += 1
            replacements = []
            for pool, probabilities, size in zip(
                self.constant_parameter_pools, self._constant_pool_probabilities, counts
            ):
                replacements.extend(np.random.choice(pool, size=int(size), replace=True, p=probabilities))
            np.random.shuffle(replacements)
            indices[positions] = replacements
        return indices

    def sample_batch(self):
        indices = self._sample_trajectory_indices()
        np.add.at(self.sampled_trajectory_counts, indices, 1)
        samples = [self._sample_one(int(index)) for index in indices]
        x, y, phi = zip(*samples)
        return torch.stack(x), torch.stack(y), None if phi[0] is None else torch.stack(phi)

    def eval_reference(self, length=10000, trajectory_index=0):
        end = min(length, int(self.train_lengths[trajectory_index]))
        x = trajectory_view(self.X_all, trajectory_index, self.lengths, self.offsets)[:end]
        phi = None if self.S_all is None else trajectory_view(self.S_all, trajectory_index, self.lengths, self.offsets)[:end]
        return x, phi

    def test_trajectory(self, trajectory_index, length=None):
        start = int(self.train_lengths[trajectory_index])
        end = int(self.lengths[trajectory_index])
        if length is not None:
            end = min(end, start + length)
        x = trajectory_view(self.X_all, trajectory_index, self.lengths, self.offsets)[start:end]
        phi = None if self.S_all is None else trajectory_view(self.S_all, trajectory_index, self.lengths, self.offsets)[start:end]
        return x, phi
