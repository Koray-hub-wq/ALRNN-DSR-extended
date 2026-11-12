import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch
from half_half_dataset import HalfHalfDataset
import plot_trajectories as plots
import run_training as runner
from alrnn_model import SavedALRNN


class RosslerExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        # Synthetic sampler fixtures: no data-generation code or source tree
        # is required to run the training/model tests on the server.
        lengths = np.array([400]*3+[256]*10)
        values = np.r_[np.linspace(3., 6., 3), np.tile(np.linspace(3., 6., 5), 2)]
        phi = np.concatenate([np.full((length, 1), (value-4.3)/1.7, dtype=np.float32)
                              for length, value in zip(lengths, values)])
        cls.arrays = dict(trajectories=np.random.default_rng(8).normal(size=(lengths.sum(), 3)).astype(np.float32),
                          trajectory_lengths=lengths,
                          trajectory_train_lengths=np.array([340]*3+[253]*10),
                          trajectory_phi=phi, trajectory_modes=np.array(["constant"]*13),
                          trajectory_group_names=np.array([f"long_{i}" for i in range(3)]+[f"short_{i%5}" for i in range(10)]),
                          trajectory_population=np.array(["attractor"]*3+["transient"]*10))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def load(self, name):
        return self.arrays[name].copy()

    def sampler(self, **kwargs):
        return HalfHalfDataset(self.load("trajectories"), lengths=self.load("trajectory_lengths"),
               train_lengths=self.load("trajectory_train_lengths"), external_inputs=self.load("trajectory_phi"),
               sampling_parameters=self.load("trajectory_phi"), trajectory_modes=self.load("trajectory_modes"),
               group_names=self.load("trajectory_group_names"), population_labels=self.load("trajectory_population"),
               balanced_sampling=True, sequence_length=64, batch_size=kwargs.pop("batch_size", 80), **kwargs)

    def test_every_batch_is_exactly_half_even_with_different_grids(self):
        sampler = self.sampler()
        for _ in range(25):
            indices = sampler._sample_trajectory_indices()
            labels = sampler.population_labels[indices]
            self.assertEqual(np.count_nonzero(labels == "attractor"), 40)
            self.assertEqual(np.count_nonzero(labels == "transient"), 40)
            for label, pools in sampler.population_pools.items():
                counts = [np.isin(indices, pool).sum() for pool in pools]
                self.assertLessEqual(max(counts)-min(counts), 1)
        probabilities = sampler.trajectory_probabilities
        self.assertAlmostEqual(probabilities[:3].sum(), .5)
        self.assertAlmostEqual(probabilities[3:].sum(), .5)
        np.testing.assert_allclose(probabilities[:3], .5/3)
        np.testing.assert_allclose(probabilities[3:], .5/10)

    def test_windows_and_telemetry_match_actual_sampling(self):
        sampler = self.sampler(batch_size=8)
        for _ in range(3):
            x, y, phi = sampler.sample_batch()
            self.assertEqual(x.shape, (8, 64, 3))
            self.assertTrue(torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(phi).all())
        metadata = sampler.sampling_metadata()
        self.assertEqual(metadata["population_sampled_windows"], {"attractor": 12, "transient": 12})
        self.assertEqual(metadata["parameter_values_per_population"], {"attractor": 3, "transient": 5})

    def test_seventy_thirty_batches_probabilities_and_actual_window_counts(self):
        sampler = self.sampler(attractor_fraction=.7)
        for _ in range(20):
            indices = sampler._sample_trajectory_indices()
            self.assertEqual(np.count_nonzero(sampler.population_labels[indices] == "attractor"), 56)
            self.assertEqual(np.count_nonzero(sampler.population_labels[indices] == "transient"), 24)
            for label, pools in sampler.population_pools.items():
                counts = [np.isin(indices, pool).sum() for pool in pools]
                self.assertLessEqual(max(counts)-min(counts), 1)
        np.testing.assert_allclose(sampler.trajectory_probabilities[:3], .7/3)
        np.testing.assert_allclose(sampler.trajectory_probabilities[3:], .3/10)
        for _ in range(4):
            sampler.sample_batch()
        metadata = sampler.sampling_metadata()
        self.assertEqual(metadata["population_windows_per_batch"], {"attractor": 56, "transient": 24})
        self.assertEqual(metadata["population_sampled_windows"], {"attractor": 224, "transient": 96})
        with self.assertRaisesRegex(ValueError, "integer"):
            self.sampler(batch_size=8, attractor_fraction=.7)

    def test_reject_odd_batches_and_keep_at_most_four_long_tail_groups(self):
        with self.assertRaisesRegex(ValueError, "even"):
            self.sampler(batch_size=7)
        sampler = self.sampler()
        self.assertEqual(len(sampler.eval_group_names), 3)
        for name in sampler.eval_group_names:
            self.assertEqual(sampler.population_labels[sampler.eval_indices_by_group[name][0]], "attractor")
        self.assertEqual(len(self.sampler(tail_evaluations=0).eval_group_names), 1)

    def test_free_rollout_uses_same_physical_initials_and_constant_c(self):
        state = dict(A=torch.ones(5), W=torch.zeros(5, 5), h=torch.zeros(5), B=torch.zeros(3, 5),
                     C=torch.tensor([[1.], [2.], [3.], [0.], [0.]]))
        model = SavedALRNN(state, pwl_units=0)
        scaling = type("Scale", (), dict(mean=np.zeros(3), std=np.ones(3), phi_boundary=0., phi_scale=1.))()
        initials = plots.initial_states(8)
        with contextlib.redirect_stdout(io.StringIO()):
            values, failures = plots.simulate_model(model, scaling, [3., 6.], initials, .04, .01, "cpu")
        np.testing.assert_allclose(values[:, 0], np.tile(initials, (2, 1)), atol=1e-6)
        for index, c in enumerate([3., 3., 6., 6.]):
            expected = initials[index % 2] + np.arange(5)[:, None]*c*np.array([1., 2., 3.])
            np.testing.assert_allclose(values[index], expected, atol=1e-5)
        np.testing.assert_array_equal(failures, [-1]*4)
        times, truth = plots.simulate_truth([3., 6.], initials, .1, .01)
        self.assertEqual(truth.shape, (4, 11, 3))
        np.testing.assert_array_equal(truth[:, 0], np.tile(initials, (2, 1)))
        np.testing.assert_allclose(plots.rossler_rhs(0, [1., 2., 3.], 5.), [-5., 1.4, -11.8])

    def test_launcher_trains_only_cphi_and_requests_full_range_analyses(self):
        args = runner.parse_args(["--dry-run", "--run-root", str(self.root / "run")])
        args.backend_script = self.root / "backend.py"
        step = runner.make_steps(args, "train__rossler_c_3_to_6", 0, .01)[0]
        self.assertTrue(step.command[2].endswith("train_cphi.py"))
        self.assertEqual(step.command[step.command.index("--latent-dim")+1], "300")
        self.assertEqual(step.command[step.command.index("--pwl-units")+1], "150")
        self.assertEqual(step.command[step.command.index("--tail-evaluations")+1], "4")
        self.assertIn("--no-best-model", step.command)
        comparison = runner.make_steps(args, "bifurcation", 0, .01)[0]
        self.assertEqual(comparison.command[comparison.command.index("--parameter-min")+1], "3.0")
        self.assertEqual(comparison.command[comparison.command.index("--parameter-max")+1], "6.0")
        self.assertEqual(args.orbit_parameters, [3., 4., 4.5, 5., 5.5, 6.])
        args = runner.parse_args(["--dry-run", "--attractor-fraction", ".7"])
        step = runner.make_steps(args, "train__rossler_c_3_to_6", 0, .01)[0]
        self.assertEqual(step.command[step.command.index("--attractor-fraction")+1], "0.7")


if __name__ == "__main__":
    unittest.main()
