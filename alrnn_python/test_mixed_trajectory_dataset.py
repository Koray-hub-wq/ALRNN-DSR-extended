import unittest

import numpy as np
import torch

from mixed_trajectory_dataset import MixedTrajectoryDataset


class MixedTrajectorySamplingTests(unittest.TestCase):
    def test_compact_and_padded_formats_produce_identical_batches_and_evaluation(self):
        for policy in ("length", "uniform"):
            for with_phi in (True, False):
                with self.subTest(policy=policy, with_phi=with_phi):
                    padded = self.dataset([1000, 100, 370], [800, 80, 290],
                                          trajectory_sampling=policy, with_phi=with_phi)
                    def pack(array):
                        return np.concatenate([array[i, :n].numpy() for i, n in enumerate(padded.lengths)])
                    compact = MixedTrajectoryDataset(
                        pack(padded.X_all), padded.lengths, padded.train_lengths,
                        external_inputs=None if padded.S_all is None else pack(padded.S_all),
                        sequence_length=padded.sequence_length, batch_size=padded.batch_size,
                        trajectory_sampling=policy,
                    )
                    np.random.seed(18)
                    expected = padded.sample_batch()
                    np.random.seed(18)
                    actual = compact.sample_batch()
                    for left, right in zip(expected, actual):
                        if left is None:
                            self.assertIsNone(right)
                        else:
                            self.assertTrue(torch.equal(left, right))
                    for i in range(3):
                        for method in ("eval_reference", "test_trajectory"):
                            left = getattr(padded, method)(trajectory_index=i)
                            right = getattr(compact, method)(trajectory_index=i)
                            for a, b in zip(left, right):
                                if a is None:
                                    self.assertIsNone(b)
                                else:
                                    self.assertTrue(torch.equal(a, b))
                    self.assertEqual(len(compact), len(padded))

    def test_compact_shape_mismatch_and_nonfinite_values_are_rejected(self):
        for data in (np.zeros((61, 3), dtype=np.float32), np.full((60, 3), np.nan, dtype=np.float32)):
            with self.assertRaises(ValueError):
                MixedTrajectoryDataset(data, [30, 30], [20, 20], sequence_length=12)
        with self.assertRaises(ValueError):
            MixedTrajectoryDataset(np.zeros((60, 3)), [30, 30], [20, 20],
                                   external_inputs=np.zeros((61, 1)), sequence_length=12)

    def dataset(self, lengths, train_lengths, *, with_phi=True, sequence_length=12,
                batch_size=512, trajectory_sampling="length"):
        data = np.full((len(lengths), max(lengths), 3), np.nan, dtype=np.float32)
        phi = np.full((len(lengths), max(lengths), 1), np.nan, dtype=np.float32)
        for index, length in enumerate(lengths):
            data[index, :length, 0] = index
            data[index, :length, 1] = np.arange(length)
            data[index, :length, 2] = index * 1000000 + np.arange(length)
            phi[index, :length, 0] = data[index, :length, 2]
        return MixedTrajectoryDataset(
            data, np.array(lengths), np.array(train_lengths),
            external_inputs=phi if with_phi else None,
            sequence_length=sequence_length, batch_size=batch_size,
            trajectory_sampling=trajectory_sampling,
        )

    def test_100000_vs_10000_steps_has_exact_expected_ratio_ten(self):
        dataset = self.dataset([100000, 10000], [80000, 8000], sequence_length=200)
        np.testing.assert_allclose(dataset.trajectory_probabilities, [10 / 11, 1 / 11])
        self.assertAlmostEqual(dataset.trajectory_probabilities[0] / dataset.trajectory_probabilities[1], 10)
        self.assertEqual(dataset.sampling_strategy, "proportional_to_training_length")
        # Subtracting the sequence length would change the requested ratio.
        self.assertNotAlmostEqual((80000 - 200) / (8000 - 200), 10)

    def test_weights_use_training_steps_instead_of_total_length(self):
        dataset = self.dataset([1000, 1000], [100, 800])
        np.testing.assert_allclose(dataset.trajectory_probabilities, [1 / 9, 8 / 9])
        equal_train = self.dataset([100, 1000], [80, 80])
        np.testing.assert_allclose(equal_train.trajectory_probabilities, [0.5, 0.5])

    def test_observed_sampling_ratio_and_window_phi_alignment(self):
        dataset = self.dataset([10000, 1000], [8000, 800])
        np.random.seed(17)
        counts = np.zeros(2, dtype=int)
        for _ in range(60):
            x, y, phi = (value.numpy() for value in dataset.sample_batch())
            ids = x[:, 0, 0].astype(int)
            counts += np.bincount(ids, minlength=2)
            np.testing.assert_array_equal(y[:, :, 0], x[:, :, 0])
            np.testing.assert_array_equal(y[:, :, 1], x[:, :, 1] + 1)
            np.testing.assert_array_equal(phi[:, :, 0], x[:, :, 2])
            self.assertTrue(np.all(y[:, -1, 1] < dataset.train_lengths[ids]))
        self.assertAlmostEqual(counts[0] / counts[1], 10, delta=0.7)

    def test_uniform_option_keeps_equal_trajectory_probabilities(self):
        dataset = self.dataset([1000, 100], [800, 80], trajectory_sampling="uniform")
        np.testing.assert_array_equal(dataset.trajectory_probabilities, [0.5, 0.5])
        self.assertEqual(dataset.sampling_strategy, "uniform_per_trajectory")
        np.random.seed(9)
        counts = np.zeros(2, dtype=int)
        for _ in range(20):
            x, _, _ = dataset.sample_batch()
            counts += np.bincount(x[:, 0, 0].numpy().astype(int), minlength=2)
        self.assertAlmostEqual(counts[0] / counts[1], 1, delta=0.1)

    def test_weighted_sampling_without_phi_is_seed_reproducible(self):
        dataset = self.dataset([1000, 100], [800, 80], with_phi=False)
        np.random.seed(31)
        first = dataset.sample_batch()
        np.random.seed(31)
        second = dataset.sample_batch()
        self.assertIsNone(first[2])
        self.assertIsNone(second[2])
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.isfinite(first[0]).all())
        self.assertTrue(torch.isfinite(first[1]).all())

    def test_unknown_sampling_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "trajectory_sampling"):
            self.dataset([100], [80], trajectory_sampling="unknown")


if __name__ == "__main__":
    unittest.main()
