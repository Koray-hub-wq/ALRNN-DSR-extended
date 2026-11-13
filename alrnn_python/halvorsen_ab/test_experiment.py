"""Physical equations, Cartesian coverage, sampling and rollout alignment checks."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from alrnn_model import SavedALRNN
from mixed_trajectory_dataset import MixedTrajectoryDataset
from halvorsen_ab import common, evaluate, generate_data, run_experiment, sync_simulator


class HalvorsenABTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name) / "data"
        cls.config = replace(common.load_config(), grid_a=2, grid_b=2, trajectories_per_pair=2,
                             n_steps=100, burn_in_steps=5)
        cls.module, _ = common.load_simulator()
        cls.metadata = generate_data.generate(cls.config, cls.folder, progress=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_available_simulators_agree_and_old_equation_is_preserved(self):
        paths = sync_simulator.simulator_paths(common.DEFAULT_ROOT)
        for path in paths[1:]:
            self.assertEqual(paths[0].read_bytes(), path.read_bytes())
        state = np.array([1., 2., 3.])
        rhs = self.module.halvorsen_rhs
        for a in (1.4, 1.75, 2.1):
            expected = [-a - 8 - 12 - 4, -2*a - 12 - 4 - 9, -3*a - 4 - 8 - 1]
            np.testing.assert_allclose(rhs(0., state, a, drive=0.), expected)
        np.testing.assert_allclose(rhs(0., state, 1.5, b=2.5, c=5.5),
                                   [-1.5-5-16.5-4, -3-7.5-5.5-9, -4.5-2.5-11-1])
        self.assertEqual(sync_simulator.updated_source(paths[0].read_bytes()), paths[0].read_bytes())

    def test_grid_and_parameter_coordinates_are_independent(self):
        config = common.load_config()
        pairs = config.pairs()
        self.assertEqual(pairs.shape, (25, 2))
        np.testing.assert_allclose(np.unique(pairs[:, 0]), [1.4, 1.575, 1.75, 1.925, 2.1])
        np.testing.assert_allclose(np.unique(pairs[:, 1]), [2., 2.5, 3., 3.5, 4.])
        np.testing.assert_allclose(config.normalize([[1.4, 2.], [1.75, 3.], [2.1, 4.]]), [[-1,-1], [0,0], [1,1]])
        np.testing.assert_allclose(config.normalize([[1.4, 4.], [2.1, 2.]]), [[-1,1], [1,-1]])

    def test_saved_data_fit_global_scaling_from_training_only(self):
        config, mean, std, metadata = common.load_dataset_metadata(self.folder)
        initials = np.load(self.folder / "trajectory_initial_states.npy")
        pairs = np.load(self.folder / "trajectory_parameter_constants.npy")
        phi = np.load(self.folder / "trajectory_phi.npy")
        states = np.load(self.folder / "trajectories.npy")
        raw = np.stack([common.simulate(self.module, config, pair, initial)
                        for pair, initial in zip(pairs, initials)])
        expected = raw[:, :config.train_length].reshape(-1, 3)
        np.testing.assert_allclose(mean, expected.mean(0), atol=1e-10)
        np.testing.assert_allclose(std, expected.std(0), atol=1e-10)
        np.testing.assert_allclose(states.reshape(len(pairs), config.n_steps, 3), (raw-mean)/std, atol=3e-7)
        np.testing.assert_allclose(phi.reshape(len(pairs), config.n_steps, 2)[:, 0], config.normalize(pairs), atol=1e-7)
        self.assertFalse(np.allclose(raw.reshape(-1, 3).mean(0), mean))
        self.assertEqual(metadata["fixed_parameters"]["c"], 4.)

    def test_balanced_sampling_groups_complete_vectors_and_preserves_channels(self):
        load = lambda name: np.load(self.folder / (name+".npy"), allow_pickle=False)
        dataset = MixedTrajectoryDataset(load("trajectories"), load("trajectory_lengths"),
                    load("trajectory_train_lengths"), external_inputs=load("trajectory_phi"),
                    trajectory_modes=load("trajectory_modes"), group_names=load("trajectory_group_names"),
                    balanced_sampling=True, sequence_length=12, batch_size=12)
        self.assertEqual(dataset.constant_phi_values.shape, (4, 2))
        for _ in range(8):
            x, y, phi = dataset.sample_batch()
            self.assertEqual(tuple(phi.shape), (12, 12, 2))
            self.assertEqual(tuple(x.shape), (12, 12, 3))
            self.assertTrue(torch.isfinite(y).all())
            values, counts = np.unique(phi[:, 0].numpy(), axis=0, return_counts=True)
            np.testing.assert_array_equal(counts, [3,3,3,3])
            self.assertEqual(len(values), 4)
        # Changes in the second component alone must classify a trajectory as dynamic.
        parameters = load("trajectory_phi").copy()
        parameters[:80, 1] = np.linspace(-1, 1, 80)
        with self.assertRaisesRegex(ValueError, "labelled constant"):
            MixedTrajectoryDataset(load("trajectories"), load("trajectory_lengths"),
                load("trajectory_train_lengths"), external_inputs=parameters,
                trajectory_modes=load("trajectory_modes"), balanced_sampling=True, sequence_length=12)

    def test_rollout_keeps_seed_and_applies_distinct_a_b_coefficients(self):
        state = dict(A=torch.ones(5), W=torch.zeros(5,5), h=torch.zeros(5), B=torch.zeros(3,5),
                     C=torch.tensor([[1.,2.], [3.,-1.], [-2.,4.], [0.,0.], [0.,0.]]))
        model = SavedALRNN(state, pwl_units=0)
        starts = np.array([[1.,2.,3.], [-1.,-2.,-3.]])
        pairs = np.array([[2.1,3.], [1.75,4.]])
        predicted, failed = evaluate.rollout_model(model, starts, pairs, np.zeros(3), np.ones(3),
                                                   common.load_config(), 4, "cpu")
        np.testing.assert_array_equal(predicted[:, 0], starts)
        np.testing.assert_array_equal(failed, [-1, -1])
        increments = np.array([[1.,3.,-2.], [2.,-1.,4.]])
        np.testing.assert_allclose(predicted, starts[:, None] + np.arange(4)[None,:,None]*increments[:,None], atol=1e-6)

    def test_vector_rounding_groups_do_not_depend_on_sort_adjacency(self):
        constants = np.array([[0.,0.], [0.,1.], [1e-8,0.]])
        parameters = np.repeat(constants[:, None, :], 24, axis=1)
        dataset = MixedTrajectoryDataset(np.zeros((3,24,3)), [24]*3, [18]*3,
                    external_inputs=parameters, balanced_sampling=True, sequence_length=4, batch_size=8)
        self.assertEqual(dataset.constant_phi_values.shape, (2,2))
        self.assertEqual([pool.tolist() for pool in dataset.constant_parameter_pools], [[0,2], [1]])

    def test_failed_model_is_visible_and_not_turned_into_finite_peaks(self):
        state = dict(A=torch.full((4,), 1e10), W=torch.zeros(4,4), h=torch.zeros(4), B=torch.zeros(3,4), C=torch.zeros(4,2))
        model = SavedALRNN(state, pwl_units=0)
        predicted, failed = evaluate.rollout_model(model, [[1.,2.,3.]], [[1.75,3.]], np.zeros(3), np.ones(3),
                                                   common.load_config(), 6, "cpu")
        np.testing.assert_array_equal(failed, [1])
        self.assertTrue(np.isnan(predicted[0, 1:]).all())
        peaks, equilibrium = evaluate.extrema(predicted[0])
        self.assertEqual(len(peaks), 0)
        self.assertTrue(np.isnan(equilibrium))

    def test_equilibria_and_local_maxima_have_distinct_meanings(self):
        peaks, equilibrium = evaluate.extrema(np.tile([1.,2.,3.], (20,1)))
        self.assertEqual(len(peaks), 0)
        self.assertEqual(equilibrium, 3.)
        states = np.zeros((7,3)); states[:,2] = [0,2,0,3,0,4,0]
        peaks, equilibrium = evaluate.extrema(states)
        np.testing.assert_array_equal(peaks, [2,3,4])
        self.assertTrue(np.isnan(equilibrium))

    def test_runner_defaults_cover_full_box_and_request_vector_sampling(self):
        args = run_experiment.parse_args(["--dry-run"])
        commands = run_experiment.commands(args)
        self.assertIn("--mixed-balanced-sampling", commands["train"])
        self.assertIn("--no-best-model", commands["train"])
        self.assertEqual(args.batch_size, 100)
        self.assertEqual(args.grid_a, 41)
        self.assertEqual(args.grid_b, 41)
        smoke = run_experiment.parse_args(["--smoke", "--dry-run"])
        self.assertIn("halvorsen_ab_smoke", str(smoke.data_dir))
        self.assertIn("halvorsen_ab_smoke", str(smoke.run_dir))
        before = generate_data.generation_identity(self.config, "test")
        after = generate_data.generation_identity(replace(self.config, c=5.), "test")
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
