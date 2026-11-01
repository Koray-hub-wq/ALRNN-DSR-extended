import unittest

import numpy as np
import torch

from mixed_trajectory_dataset import MixedTrajectoryDataset


class MixedBalancedSamplingTests(unittest.TestCase):
    def dataset(self, *, compact=False, policy="length", with_phi=True, batch_size=80,
                records=None, time_bins=10, with_modes=True):
        # Repeated parameter values have unequal counts and training lengths.
        records = records or [(800, "constant", -1), (80, "constant", -1),
                              (160, "constant", 0), (240, "constant", 1),
                              (80, "constant", 1), (400, "linear_increasing", None),
                              (200, "linear_decreasing", None), (100, "sinusoidal", None)]
        train = np.array([row[0] for row in records])
        lengths = train + 8
        states = np.full((len(records), max(lengths), 3), np.nan, dtype=np.float32)
        phi = np.full((len(records), max(lengths), 1), np.nan, dtype=np.float32)
        for index, (length, (_, mode, value)) in enumerate(zip(lengths, records)):
            time = np.arange(length)
            states[index, :length] = np.column_stack((np.full(length, index), time, index * 10000 + time))
            if mode == "constant":
                phi[index, :length, 0] = value
            elif mode == "sinusoidal":
                phi[index, :length, 0] = -np.cos(np.linspace(0, 8 * np.pi, length))
            else:
                phi[index, :length, 0] = np.linspace(-1, 1, length) * (1 if mode == "linear_increasing" else -1)
        if compact:
            states = np.concatenate([states[i, :n] for i, n in enumerate(lengths)])
            phi = np.concatenate([phi[i, :n] for i, n in enumerate(lengths)])
        return MixedTrajectoryDataset(
            states, lengths, train, external_inputs=phi if with_phi else None,
            sampling_parameters=phi, trajectory_modes=[row[1] for row in records] if with_modes else None,
            sequence_length=12, batch_size=batch_size, trajectory_sampling=policy,
            balanced_sampling=True, time_sampling_bins=time_bins,
        )

    def test_constant_counts_are_balanced_and_original_dynamic_draws_are_preserved(self):
        for policy in ("length", "uniform"):
            dataset = self.dataset(policy=policy)
            for seed in range(20):
                np.random.seed(seed)
                if policy == "length":
                    baseline = np.random.choice(dataset.num_trajectories, dataset.batch_size,
                                                p=dataset.base_trajectory_probabilities)
                else:
                    baseline = np.random.randint(dataset.num_trajectories, size=dataset.batch_size)
                np.random.seed(seed)
                actual = dataset._sample_trajectory_indices()
                constant_slots = dataset.constant_mask[baseline]
                np.testing.assert_array_equal(dataset.constant_mask[actual], constant_slots)
                np.testing.assert_array_equal(actual[~constant_slots], baseline[~constant_slots])
                counts = [np.isin(actual, pool).sum() for pool in dataset.constant_parameter_pools]
                self.assertLessEqual(max(counts) - min(counts), 1)

    def test_effective_probabilities_balance_values_and_keep_length_ratio_within_value(self):
        dataset = self.dataset()
        baseline = dataset.base_trajectory_probabilities
        actual = dataset.trajectory_probabilities
        mass = baseline[dataset.constant_mask].sum()
        for pool in dataset.constant_parameter_pools:
            self.assertAlmostEqual(actual[pool].sum(), mass / 3)
        np.testing.assert_array_equal(actual[~dataset.constant_mask], baseline[~dataset.constant_mask])
        self.assertAlmostEqual(actual.sum(), 1)
        self.assertAlmostEqual(actual[0] / actual[1], 10)
        self.assertAlmostEqual(actual[3] / actual[4], 3)
        np.random.seed(17)
        counts = np.zeros(dataset.num_trajectories, dtype=int)
        for _ in range(600):
            counts += np.bincount(dataset._sample_trajectory_indices(), minlength=len(counts))
        self.assertAlmostEqual(counts[0] / counts[1], 10, delta=0.8)
        self.assertAlmostEqual(counts[3] / counts[4], 3, delta=0.2)

    def test_time_sections_cycle_without_replacement_and_never_cross_training_boundary(self):
        dataset = self.dataset()
        np.random.seed(18)
        for index in np.flatnonzero(~dataset.constant_mask):
            n_starts = int(dataset.train_lengths[index] - dataset.sequence_length)
            edges = np.arange(11) * n_starts // 10
            seen = []
            for _ in range(34):
                x, y, phi = dataset._sample_one(index)
                start = int(x[0, 1])
                seen.append(int(np.searchsorted(edges, start, side="right") - 1))
                self.assertLess(y[-1, 1], dataset.train_lengths[index])
                np.testing.assert_array_equal(y[:, 1].numpy(), x[:, 1].numpy() + 1)
                self.assertTrue(phi.isfinite().all())
            for cycle in range(3):
                self.assertEqual(sorted(seen[cycle * 10:(cycle + 1) * 10]), list(range(10)))
            counts = dataset.sampled_time_bin_counts[index]
            self.assertEqual(counts.sum(), 34)
            self.assertLessEqual(counts.max() - counts.min(), 1)

    def test_fewer_valid_starts_than_requested_bins_has_no_empty_sections(self):
        for train_length in (13, 17):
            dataset = self.dataset(records=[(train_length, "linear_increasing", None)])
            bins = train_length - 12
            self.assertEqual(dataset.effective_time_bins.tolist(), [bins])
            starts = [int(dataset._sample_one(0)[0][0, 1]) for _ in range(bins)]
            self.assertEqual(sorted(starts), list(range(bins)))

    def test_compact_padded_and_vanilla_sampling_are_seed_reproducible(self):
        for policy in ("length", "uniform"):
            for with_phi in (True, False):
                padded = self.dataset(policy=policy, with_phi=with_phi)
                compact = self.dataset(compact=True, policy=policy, with_phi=with_phi)
                np.random.seed(31)
                batches = [padded.sample_batch() for _ in range(8)]
                np.random.seed(31)
                for expected in batches:
                    actual = compact.sample_batch()
                    for a, b in zip(expected, actual):
                        if a is None:
                            self.assertIsNone(b)
                        else:
                            self.assertTrue(torch.equal(a, b))
                self.assertEqual(padded.sampling_metadata(), compact.sampling_metadata())

    def test_all_constant_and_all_dynamic_datasets(self):
        for records in ([(80, "constant", -1), (160, "constant", 0), (80, "constant", 1)],
                        [(80, "linear_increasing", None), (160, "sinusoidal", None)]):
            dataset = self.dataset(records=records, batch_size=12)
            x, y, phi = dataset.sample_batch()
            self.assertTrue(x.isfinite().all() and y.isfinite().all() and phi.isfinite().all())
            if dataset.constant_mask.all():
                np.testing.assert_array_equal(np.bincount(x[:, 0, 0].numpy().astype(int)), [4, 4, 4])
            else:
                self.assertFalse(dataset.constant_parameter_pools)

    def test_one_slot_for_many_constant_values_randomizes_the_remainder(self):
        dataset = self.dataset(records=[(80, "constant", -1), (80, "constant", 0), (80, "constant", 1)],
                               batch_size=1)
        np.random.seed(19)
        counts = np.zeros(3, dtype=int)
        for _ in range(600):
            counts[dataset._sample_trajectory_indices()[0]] += 1
        self.assertTrue(np.all((counts > 150) & (counts < 250)))

    def test_metadata_describes_actual_coverage(self):
        dataset = self.dataset()
        for _ in range(9):
            dataset.sample_batch()
        metadata = dataset.sampling_metadata()
        self.assertTrue(metadata["balanced"])
        self.assertEqual(metadata["constant_phi_values"], [-1, 0, 1])
        self.assertEqual(metadata["requested_time_bins"], 10)
        self.assertEqual(sum(metadata["sampled_trajectory_counts"]), 9 * dataset.batch_size)
        for index in np.flatnonzero(~dataset.constant_mask):
            counts = metadata["sampled_time_bin_counts"][index]
            self.assertEqual(sum(counts), metadata["sampled_trajectory_counts"][index])
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_missing_or_invalid_sampling_information_is_rejected(self):
        states = np.zeros((2, 100, 3), dtype=np.float32)
        lengths, train = np.array([100, 100]), np.array([80, 80])
        with self.assertRaisesRegex(ValueError, "requires phi arrays"):
            MixedTrajectoryDataset(states, lengths, train, sequence_length=12, balanced_sampling=True)
        for time_bins in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                self.dataset(time_bins=time_bins)
        varying = np.broadcast_to(np.linspace(-1, 1, 100)[None, :, None], (2, 100, 1)).copy()
        with self.assertRaisesRegex(ValueError, "labelled constant"):
            MixedTrajectoryDataset(states, lengths, train, sampling_parameters=varying,
                                   trajectory_modes=["constant", "sinusoidal"],
                                   sequence_length=12, balanced_sampling=True)

    def test_parameter_classification_uses_training_prefix_and_saved_modes(self):
        data = np.zeros((2, 100, 3), dtype=np.float32)
        phi = np.zeros((2, 100, 1), dtype=np.float32)
        phi[:, 80:] = 10  # Held-out values must not affect the classification.
        dataset = MixedTrajectoryDataset(data, [100, 100], [80, 80], external_inputs=phi,
                                         sequence_length=12, balanced_sampling=True)
        self.assertTrue(dataset.constant_mask.all())
        np.testing.assert_array_equal(dataset.constant_phi_values, [0])
        labelled = MixedTrajectoryDataset(data, [100, 100], [80, 80], external_inputs=phi,
                                           trajectory_modes=["constant", "sinusoidal"],
                                           sequence_length=12, balanced_sampling=True)
        np.testing.assert_array_equal(labelled.effective_time_bins, [0, 10])


if __name__ == "__main__":
    unittest.main()
