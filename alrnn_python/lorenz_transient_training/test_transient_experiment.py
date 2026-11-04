"""Physical/data-contract checks for the isolated Lorenz experiment."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import generate_data as generator
import run_training as runner
import plot_trajectories as plots
from mixed_trajectory_dataset import MixedTrajectoryDataset
from alrnn_model import SavedALRNN


class TransientExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.recipe = json.loads((generator.HERE / "config.json").read_text())
        cls.recipe.update(n_constant_parameter_values=4, n_dynamic_schedules_per_mode=2, n_steps=256)
        cls.config = generator.expand_config(cls.recipe, cls.recipe["datasets"][0])
        cls.data = cls.root / "data"
        cls.metadata = generator.generate(cls.config, cls.data, progress=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def load(self, name):
        return np.load(self.data / f"{name}.npy", allow_pickle=False)

    def test_default_recipe_focuses_on_short_constant_transients(self):
        recipe = json.loads((generator.HERE / "config.json").read_text())
        generator.validate_recipe(recipe)
        for entry in recipe["datasets"]:
            config = generator.expand_config(recipe, entry)
            plan = generator.paired_plan(config)
            self.assertEqual(len(plan), 304)
            self.assertEqual(sum(r["mode"] == "constant" for r in plan), 256)
            self.assertTrue(all((r["n_steps"], r["train_length"], r["burn_in_steps"]) == (500, 495, 0) for r in plan))
            self.assertEqual(sum(r["initial_state_index"] == 0 for r in plan), 152)

    def test_exact_starts_and_lorenz_symmetry_survive_normalization(self):
        states = self.load("trajectories").reshape(-1, 256, 3)
        physical = states*self.load("trajectory_std") + self.load("trajectory_mean")
        np.testing.assert_allclose(physical[:, 0], self.load("trajectory_initial_states"), atol=2e-6)
        records = self.metadata["trajectories"]
        for group in {r["group"] for r in records}:
            pair = [r["trajectory_index"] for r in records if r["group"] == group]
            np.testing.assert_array_equal(self.load("trajectory_initial_state_index")[pair], [0, 1])
            np.testing.assert_allclose(physical[pair[0]]*[-1, -1, 1], physical[pair[1]], atol=5e-5, rtol=1e-5)
        np.testing.assert_allclose(self.load("trajectory_mean")[0, 0, :2], 0., atol=1e-12)
        self.assertFalse((self.data / "trajectories_clean.npy").exists())
        self.assertEqual(self.metadata["config"]["noise_std_fraction"], 0.)

    def test_scaling_uses_training_prefixes_and_compact_storage(self):
        states = self.load("trajectories").reshape(-1, 256, 3)
        train = self.load("trajectory_train_lengths")
        prefix = np.concatenate([state[:length] for state, length in zip(states, train)])
        np.testing.assert_allclose(prefix.mean(axis=0, dtype=np.float64), 0., atol=1e-6)
        np.testing.assert_allclose(prefix.std(axis=0, dtype=np.float64), 1., atol=1e-6)
        self.assertTrue(np.all(256-train == 3))
        self.assertEqual(self.load("trajectories").shape, (len(states)*256, 3))
        raw, phi = self.load("trajectory_phi_raw"), self.load("trajectory_phi")
        np.testing.assert_allclose(phi, (raw-24.74)/23.74, atol=2e-7)

    def test_existing_sampler_accepts_paired_data_and_retains_mixture(self):
        sampler = MixedTrajectoryDataset(self.load("trajectories"), self.load("trajectory_lengths"),
                  self.load("trajectory_train_lengths"), external_inputs=self.load("trajectory_phi"),
                  group_names=self.load("trajectory_group_names"), sequence_length=64, batch_size=8,
                  balanced_sampling=True, trajectory_modes=self.load("trajectory_modes"))
        indices = self.load("trajectory_initial_state_index")
        self.assertAlmostEqual(sampler.trajectory_probabilities[indices == 0].sum(), .5)
        self.assertAlmostEqual(sampler.trajectory_probabilities[sampler.constant_mask].sum(), 8/20)
        for _ in range(3):
            x, y, phi = sampler.sample_batch()
            self.assertTrue(torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(phi).all())
            self.assertEqual(tuple(x.shape), (8, 64, 3))

    def test_fixed_initial_states_are_never_resampled_on_failure(self):
        with patch.object(generator, "simulate_trajectory", side_effect=RuntimeError("solver failed")) as simulate:
            with self.assertRaisesRegex(RuntimeError, "No resampling"):
                generator.generate(self.config, self.root / "failed", progress=False)
        self.assertEqual(simulate.call_count, 1)
        self.assertFalse((self.root / "failed/metadata.json").exists())

    def test_skip_completed_rejects_different_recipe_without_overwriting(self):
        recipe = copy.deepcopy(self.recipe)
        recipe["datasets"] = recipe["datasets"][:1]
        root = self.root / "skip"
        folder = root / recipe["datasets"][0]["name"]
        generator.generate(self.config, folder, progress=False)
        config_path = self.root / "recipe.json"
        config_path.write_text(json.dumps(recipe))
        before = (folder / "trajectories.npy").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            generator.main(["--config", str(config_path), "--output-root", str(root), "--skip-completed"])
        self.assertEqual(before, (folder / "trajectories.npy").read_bytes())
        recipe["train_fraction"] = .98
        config_path.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, "differs"), contextlib.redirect_stdout(io.StringIO()):
            generator.main(["--config", str(config_path), "--output-root", str(root), "--skip-completed"])
        self.assertEqual(before, (folder / "trajectories.npy").read_bytes())

    def test_free_orbits_include_exact_t0_and_correct_number_of_steps(self):
        state = dict(A=torch.ones(5), W=torch.zeros(5, 5), h=torch.zeros(5), B=torch.zeros(3, 5),
                     C=torch.tensor([[1.], [2.], [3.], [0.], [0.]]))
        model = SavedALRNN(state, pwl_units=0)
        scaling = type("Coordinates", (), dict(mean=np.zeros(3), std=np.ones(3), phi_boundary=0., phi_scale=1.))()
        with contextlib.redirect_stdout(io.StringIO()):
            values, failures = plots.free_orbits(model, scaling, [1., 2.], .04, .01, "cpu")
        self.assertEqual(values.shape, (4, 5, 3))
        np.testing.assert_array_equal(values[:, 0], np.tile(generator.INITIAL_STATES, (2, 1)))
        for index, rho in enumerate([1., 1., 2., 2.]):
            expected = generator.INITIAL_STATES[index % 2] + np.arange(5)[:, None]*rho*np.array([1., 2., 3.])
            np.testing.assert_array_equal(values[index], expected)
        np.testing.assert_array_equal(failures, [-1]*4)

    def test_physical_truth_is_paired_for_both_parameter_values(self):
        times, values = plots.true_orbits([5., 23.], .1, .01)
        self.assertEqual(len(times), 11)
        np.testing.assert_array_equal(values[:, 0], np.tile(generator.INITIAL_STATES, (2, 1)))
        np.testing.assert_allclose(values[::2]*[-1, -1, 1], values[1::2], atol=1e-10)

    def test_four_training_jobs_keep_sizes_balancing_and_final_weights(self):
        args = runner.parse_args(["--dry-run", "--run-root", str(self.root / "dry")])
        variants = runner.variants(args)
        trainings, _ = runner.job_definitions(variants, dict.fromkeys(runner.paper.DATASETS))
        self.assertEqual(len(trainings), 4)
        self.assertEqual((variants["cphi"].latent_dim, variants["cphi"].pwl_units), (300, 150))
        self.assertEqual((variants["cphi_shifted_relu"].latent_dim, variants["cphi_shifted_relu"].pwl_units), (200, 100))
        for name in trainings:
            step = runner.make_steps(args, variants, trainings, name, 0)[0]
            self.assertIn("--mixed-balanced-sampling", step.command)
            self.assertIn("--no-best-model", step.command)
            self.assertEqual(step.command[step.command.index("--t-transient")+1], "1000")
        analysis = runner.make_steps(args, variants, trainings, "cphi__analysis", 0)
        self.assertEqual(len(analysis), 2)
        self.assertTrue(analysis[0].command[2].endswith("evaluate_lorenz.py"))
        self.assertTrue(analysis[1].command[2].endswith("plot_trajectories.py"))


if __name__ == "__main__":
    unittest.main()
