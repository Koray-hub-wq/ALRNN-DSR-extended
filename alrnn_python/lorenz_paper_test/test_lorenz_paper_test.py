import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import evaluate_lorenz as experiment


class LorenzPaperTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "trained_run"
        self.data = self.root / "data"
        for name in experiment.DATASETS:
            folder = self.data / name
            folder.mkdir(parents=True)
            metadata = dict(config=dict(regime="lorenz_rho_1_to_30", standardization="global", dt=.01),
                            regimes=[dict(name="lorenz_rho_1_to_30", boundary=24.74, start=1., end=30.)],
                            phi_reference=dict(boundary=24.74, scale=23.74))
            (folder / "metadata.json").write_text(json.dumps(metadata))
            np.save(folder / "trajectory_mean.npy", np.zeros((1, 1, 3)))
            np.save(folder / "trajectory_std.npy", np.ones((1, 1, 3)))
            np.save(folder / "trajectory_phi_raw.npy", np.linspace(5., 23., 10).reshape(-1, 1))
            np.save(folder / "trajectory_lengths.npy", [10])
            np.save(folder / "trajectory_train_lengths.npy", [8])
            for variant in experiment.VARIANTS:
                model_dir = self.run / variant / name
                model_dir.mkdir(parents=True)
                state = dict(A=torch.ones(5), W=torch.zeros((5, 5)), h=torch.zeros(5),
                             B=torch.zeros((3, 5)), C=torch.zeros((5, 1)))
                torch.save(state, model_dir / "model.pt")
                (model_dir / "metrics.json").write_text(json.dumps(dict(args=dict(latent_dim=5, pwl_units=0), phi_dim=1)))

    def args(self, *extra):
        return experiment.parse_args(["--run-root", str(self.run), "--training-data-root", str(self.data),
                                      "--output-root", str(self.root / "results"), "--device", "cpu",
                                      "--num-parameters", "2", "--parameter-min", "5", "--parameter-max", "10",
                                      "--warmup-time", ".1", "--observation-time", ".1", *extra])

    def config(self):
        args = self.args()
        scales = experiment.preflight(args)
        return experiment.settings(args, scales[0])

    def test_defaults_match_intended_times_and_initial_states(self):
        args = experiment.parse_args([])
        self.assertEqual((args.warmup_time, args.observation_time, args.num_parameters), (200., 50., 290))
        np.testing.assert_array_equal(experiment.INITIAL_STATES, [[1, 1, 1], [-1, -1, 1]])
        self.assertEqual(args.output_root.parent, experiment.HERE / "results")

    def test_dt_must_match_training_and_time_windows(self):
        args = self.args("--dt", ".02")
        with self.assertRaisesRegex(ValueError, "sampling interval"):
            experiment.preflight(args)
        args = self.args("--warmup-time", ".123")
        with self.assertRaisesRegex(ValueError, "multiple of dt"):
            experiment.preflight(args)

    def test_constant_fixed_point_produces_a_point_instead_of_no_maxima(self):
        fixed = np.array([np.sqrt(24.), np.sqrt(24.), 9.])
        np.testing.assert_allclose(experiment.lorenz_rhs(0, fixed, 10.), 0., atol=1e-12)
        values, kinds, diagnostic = experiment.extract_points(np.tile(fixed, (40, 1)), np.zeros(3), self.config())
        self.assertEqual(diagnostic["status"], "stationary")
        np.testing.assert_allclose(values, [fixed[0]])
        np.testing.assert_array_equal(kinds, [0])

    def test_all_components_and_rate_are_required_for_stationarity(self):
        tail = np.ones((20, 3))
        tail[:, 1] = np.linspace(0, 1, 20)
        values, _, diagnostic = experiment.extract_points(tail, np.zeros(3), self.config())
        self.assertEqual(diagnostic["status"], "no_extrema_not_stationary")
        self.assertEqual(len(values), 0)
        tail[:] = 1.
        _, _, diagnostic = experiment.extract_points(tail, np.ones(3), self.config())
        self.assertNotEqual(diagnostic["status"], "stationary")

    def test_both_minima_and_maxima_are_returned(self):
        x = np.array([0, 2, 0, -3, 0, 4, 0, -2, 0.])
        tail = np.column_stack([x, x, x])
        values, kinds, diagnostic = experiment.extract_points(tail, [1., 1., 1.], self.config())
        self.assertEqual(diagnostic["status"], "oscillatory_or_transient")
        np.testing.assert_array_equal(np.sort(values[kinds == -1]), [-3, -2])
        np.testing.assert_array_equal(np.sort(values[kinds == 1]), [2, 4])

    def test_model_warms_up_itself_from_initial_state(self):
        state = dict(A=torch.ones(5), W=torch.zeros((5, 5)), h=torch.zeros(5), B=torch.zeros((3, 5)),
                     C=torch.tensor([[1.], [2.], [3.], [0.], [0.]]))
        # Verify the public recurrence, not a mock forecaster.
        from alrnn_model import SavedALRNN
        model = SavedALRNN(state, pwl_units=0)
        initial = np.array([[1., 1., 1.], [-1., -1., 1.]], dtype=np.float32)
        batches = list(experiment.model_tails(model, initial, np.array([1., -1.]), 3, 4, "cpu", 1))
        self.assertEqual(len(batches), 2)
        for start, stop, tails, failures in batches:
            expected = initial[start] + np.arange(3, 8)[:, None] * [1, 2, 3] * (1 if start == 0 else -1)
            np.testing.assert_allclose(tails[0], expected)
            np.testing.assert_array_equal(failures, [-1])

    def test_truth_retains_both_initial_conditions_and_reuses_own_cache(self):
        args = self.args()
        scales = experiment.preflight(args)
        config = experiment.settings(args, scales[0])
        # Controlled constant solutions isolate the IC/parameter pairing.
        def solve(ode, interval, initial, t_eval, **kwargs):
            return type("Solution", (), dict(success=True, message="ok", y=np.tile(initial, (len(t_eval), 1)).T))()
        with patch.object(experiment, "solve_ivp", side_effect=solve), contextlib.redirect_stdout(io.StringIO()):
            points, metadata = experiment.ground_truth(args, scales[0], config)
        self.assertEqual({d["initial_condition"] for d in metadata["diagnostics"]}, {0, 1})
        with patch.object(experiment, "solve_ivp", side_effect=AssertionError("must reuse")), contextlib.redirect_stdout(io.StringIO()):
            cached, _ = experiment.ground_truth(args, scales[0], config)
        np.testing.assert_array_equal(cached["rho"], points["rho"])
        args.num_parameters = 3
        with self.assertRaisesRegex(ValueError, "Cache settings"):
            experiment.ground_truth(args, scales[0], experiment.settings(args, scales[0]))

    def test_plot_only_reuses_prediction_identity_and_never_calls_model(self):
        args = self.args()
        scales = experiment.preflight(args)
        config = experiment.settings(args, scales[0])
        with contextlib.redirect_stdout(io.StringIO()):
            truth, _ = experiment.ground_truth(args, scales[0], config)
            args.variant = "cphi"
            with patch.object(experiment, "plot_results", return_value=[]):
                experiment.worker(args, scales, truth, config)
        path = args.output_root / "cphi/points.npz"
        before = path.read_bytes()
        args.plot_only = True
        with patch.object(experiment, "load_saved_model", side_effect=AssertionError("must not infer")), \
             patch.object(experiment, "plot_results", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            experiment.worker(args, scales, truth, config)
        self.assertEqual(path.read_bytes(), before)

    def test_source_hash_invalidates_model_points_when_checkpoint_changes(self):
        args = self.args()
        scales = experiment.preflight(args)
        run = self.run / "cphi" / experiment.DATASETS[0]
        first = experiment.model_source(run, scales[0])
        (run / "model.pt").write_bytes(b"different checkpoint")
        second = experiment.model_source(run, scales[0])
        self.assertNotEqual(first["checkpoint_sha256"], second["checkpoint_sha256"])

    def test_worker_arguments_preserve_simulation_and_training_coordinates(self):
        args = self.args()
        experiment.preflight(args)
        command = experiment.worker_command(args, "cphi_shifted_relu", 3)
        self.assertEqual(command[command.index("--gpu-ids")+1], "3")
        self.assertEqual(command[command.index("--dt")+1], "0.01")
        self.assertNotIn("train_halvorsen_plain.py", " ".join(command))
        worker = experiment.parse_args(command[3:])
        self.assertEqual(worker.variant, "cphi_shifted_relu")
        self.assertEqual(worker.output_root, args.output_root)

    def test_gpu_parent_queues_only_the_two_evaluations(self):
        args = self.args("--device", "cuda", "--gpu-ids", "2", "3")
        argv = ["--run-root", str(args.run_root), "--training-data-root", str(args.training_data_root),
                "--output-root", str(args.output_root), "--gpu-ids", "2", "3"]

        def scheduler(gpu_file, gpu_ids, device_count, logs, steps, env, job_names, max_gpus):
            self.assertEqual(gpu_ids, [2, 3])
            self.assertEqual(job_names, ["cphi", "cphi_shifted_relu"])
            for variant, gpu in zip(job_names, gpu_ids):
                command = steps(variant, gpu)[0].command
                self.assertEqual(command[command.index("--gpu-ids")+1], str(gpu))
                self.assertIn("--warmup-time", command)
                self.assertNotIn("train_halvorsen_plain.py", " ".join(command))
            return dict.fromkeys(job_names, 0)

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=4), \
             patch.object(experiment, "ground_truth", return_value=({}, {})), \
             patch.object(experiment, "schedule", side_effect=scheduler), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.main(argv), 0)


if __name__ == "__main__":
    unittest.main()
