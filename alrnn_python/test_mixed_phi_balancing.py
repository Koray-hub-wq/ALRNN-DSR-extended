import unittest

import numpy as np
import torch

from mixed_trajectory_dataset import MixedTrajectoryDataset


class MixedPhiBalancingTests(unittest.TestCase):
    def dataset(self, values, *, batch_size=80, balanced=True, lengths=None):
        lengths = np.asarray(lengths or [20] * len(values))
        width = int(lengths.max())
        data = np.full((len(values), width, 3), np.nan, dtype=np.float32)
        phi = np.full((len(values), width, 1), np.nan, dtype=np.float32)
        for index, (value, length) in enumerate(zip(values, lengths)):
            data[index, :length, 0] = index
            data[index, :length, 1] = np.arange(length)
            data[index, :length, 2] = -index
            phi[index, :length, 0] = value
        return MixedTrajectoryDataset(data, lengths, lengths - 2, phi,
                                      sequence_length=4, batch_size=batch_size,
                                      balanced_phi=balanced)

    def test_three_phi_values_are_balanced_in_every_batch(self):
        dataset = self.dataset([-1, -.75, -.5] * 8)
        extras = set()
        for _ in range(60):
            x, y, phi = dataset.sample_batch()
            values, counts = np.unique(phi[:, 0, 0].numpy(), return_counts=True)
            self.assertEqual(sorted(counts.tolist()), [26, 27, 27])
            extras.add(float(values[np.argmin(counts)]))
            torch.testing.assert_close(y[:, :, 1], x[:, :, 1] + 1)
        self.assertEqual(len(extras), 3)
        self.assertEqual(dataset.sampling, "balanced_per_phi_per_batch")
        self.assertEqual(dataset.sampling_metadata()["trajectories_per_phi"], [8, 8, 8])

    def test_five_phi_values_have_exactly_sixteen_windows_each(self):
        dataset = self.dataset([-1, -.5, 0, .5, 1] * 8)
        for _ in range(10):
            _, _, phi = dataset.sample_batch()
            np.testing.assert_array_equal(np.unique(phi[:, 0, 0], return_counts=True)[1], [16] * 5)

    def test_balancing_does_not_depend_on_trajectory_counts(self):
        dataset = self.dataset([-1, -1, -1, .5], batch_size=12)
        trajectory_ids = set()
        for _ in range(10):
            x, _, phi = dataset.sample_batch()
            np.testing.assert_array_equal(np.unique(phi[:, 0, 0], return_counts=True)[1], [6, 6])
            trajectory_ids.update(x[:, 0, 0].tolist())
        self.assertEqual(trajectory_ids, {0, 1, 2, 3})

    def test_small_batches_randomly_cover_all_groups(self):
        dataset = self.dataset([-1, 0, 1], batch_size=2)
        seen = set()
        for _ in range(30):
            _, _, phi = dataset.sample_batch()
            values, counts = np.unique(phi[:, 0, 0], return_counts=True)
            self.assertEqual(counts.tolist(), [1, 1])
            seen.update(values.tolist())
        self.assertEqual(seen, {-1, 0, 1})

    def test_padding_and_near_equal_constants_are_handled(self):
        dataset = self.dataset([-.5, -.5000002, .5], batch_size=8, lengths=[11, 15, 20])
        self.assertEqual(len(dataset.unique_phi), 2)
        self.assertEqual(dataset.sampling_metadata()["trajectories_per_phi"], [2, 1])
        x, y, phi = dataset.sample_batch()
        self.assertTrue(torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(phi).all())
        for window in y:
            index = int(window[0, 0])
            self.assertLess(window[-1, 1].item(), dataset.train_lengths[index])

    def test_dynamic_and_combined_datasets_keep_original_sampling(self):
        for values in ([np.linspace(-1, 1, 20)], [-1, np.linspace(-1, 1, 20), .5]):
            requested = self.dataset(values)
            original = self.dataset(values, balanced=False)
            self.assertFalse(requested.balanced_phi)
            self.assertEqual(requested.sampling_reason, "time_varying_phi_present")
            np.random.seed(7)
            batch_requested = requested.sample_batch()
            np.random.seed(7)
            batch_original = original.sample_batch()
            for actual, expected in zip(batch_requested, batch_original):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_disabled_option_preserves_uniform_trajectory_sampling(self):
        dataset = self.dataset([-1, -.5, 1], balanced=False)
        np.random.seed(7)
        expected = np.random.randint(0, 3, size=80)
        np.random.seed(7)
        x, _, _ = dataset.sample_batch()
        np.testing.assert_array_equal(x[:, 0, 0].numpy(), expected)
        self.assertEqual(dataset.sampling, "uniform_per_trajectory")

    def test_balancing_requires_scalar_external_input(self):
        data = np.ones((2, 20, 3), dtype=np.float32)
        for phi in (None, np.ones((2, 20, 2), dtype=np.float32)):
            with self.assertRaisesRegex(ValueError, "scalar external phi"):
                MixedTrajectoryDataset(data, np.array([20, 20]), np.array([18, 18]), phi,
                                       sequence_length=4, balanced_phi=True)


if __name__ == "__main__":
    unittest.main()
