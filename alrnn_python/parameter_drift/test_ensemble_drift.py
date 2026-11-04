import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import evaluate_ensemble_drift as ensemble


@dataclass
class TestRegime:
    name: str = "test_flow"
    parameter: str = "a"
    start: float = 0.5
    end: float = 1.5
    boundary: float = 1.0
    init_center: tuple = (0., 0., 0.)
    init_scale: tuple = (0.3, 0.3, 0.3)
    observation: str = "identity"
    rhs_key: str = "test"
    params: dict = None


class EnsembleDriftTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(dict(model_runs=["model"], output_dir="results",
                                              regime="test_flow", n_trajectories=3, steps=21,
                                              burn_in_steps=4, dt=0.1, normalization="none",
                                              device="cpu")))
        self.regime = TestRegime(params={})
        self.module = SimpleNamespace(RHS={"test": lambda t, x, p: p * np.array([1, 2, 3])},
                                      observe_state=lambda x, observation: x)

    def args(self, *extra):
        return ensemble.parse_args(["--config", str(self.config), *extra])

    def spec(self):
        return ensemble.simulation_spec(self.args(), self.regime)

    def state(self):
        return dict(A=torch.ones(5), W=torch.zeros(5, 5), h=torch.zeros(5),
                    B=torch.zeros(3, 5), C=torch.tensor([[1.], [2.], [3.], [0.], [0.]]))

    def training_fixture(self):
        folder = self.root / "training"
        folder.mkdir()
        metadata = dict(config=dict(regime=self.regime.name, standardization="global"),
                        phi_reference=dict(boundary=1., scale=.5))
        (folder / "metadata.json").write_text(json.dumps(metadata))
        np.save(folder / "trajectory_mean.npy", np.tile([1., 2., 3.], (2, 1, 1)))
        np.save(folder / "trajectory_std.npy", np.tile([2., 3., 4.], (2, 1, 1)))
        return folder

    def test_global_training_scaling_is_used_for_all_ensemble_members(self):
        folder = self.training_fixture()
        args = self.args("--normalization", "fixed", "--training-data-dir", str(folder))
        self.assertEqual(args.normalization_stats, args.output_dir / "training_scaling.npz")
        with contextlib.redirect_stdout(io.StringIO()):
            ensemble.prepare_training_scaling(args, self.regime)
        spec = ensemble.simulation_spec(args, self.regime)
        arrays, _ = ensemble.simulate_ensemble(self.module, self.regime, spec, 3,
                                              scaling_path=args.normalization_stats, progress=False)
        np.testing.assert_array_equal(arrays["mean"], np.tile([1., 2., 3.], (3, 1)))
        np.testing.assert_array_equal(arrays["std"], np.tile([2., 3., 4.], (3, 1)))
        np.testing.assert_allclose(ensemble.physical_prediction(arrays["x"], arrays), arrays["observed"], atol=1e-6)
        self.assertEqual(args.training_scaling_source["training_data_dir"], str(folder))

    def test_training_snapshot_reuse_and_replot_do_not_require_original_data(self):
        folder = self.training_fixture()
        args = self.args("--normalization", "fixed", "--training-data-dir", str(folder))
        with contextlib.redirect_stdout(io.StringIO()):
            ensemble.prepare_training_scaling(args, self.regime)
            digest = ensemble.single.file_hash(args.normalization_stats)
            ensemble.prepare_training_scaling(args, self.regime)
            self.assertEqual(ensemble.single.file_hash(args.normalization_stats), digest)
            (folder / "metadata.json").unlink()
            args.plot_only = True
            ensemble.prepare_training_scaling(args, self.regime)
        self.assertEqual(ensemble.single.file_hash(args.normalization_stats), digest)

    def test_training_snapshot_rejects_changed_scaling_unless_regenerated(self):
        folder = self.training_fixture()
        args = self.args("--normalization", "fixed", "--training-data-dir", str(folder))
        with contextlib.redirect_stdout(io.StringIO()):
            ensemble.prepare_training_scaling(args, self.regime)
        np.save(folder / "trajectory_std.npy", np.tile([4., 5., 6.], (2, 1, 1)))
        with self.assertRaisesRegex(ValueError, "scaling differs"):
            ensemble.prepare_training_scaling(args, self.regime)
        args.regenerate_ground_truth = True
        with contextlib.redirect_stdout(io.StringIO()):
            ensemble.prepare_training_scaling(args, self.regime)
        _, std = ensemble.single.load_scaling(args.normalization_stats)
        np.testing.assert_array_equal(std, [4., 5., 6.])

    def test_training_regime_and_phi_reference_must_match_the_drift(self):
        folder = self.training_fixture()
        args = self.args("--normalization", "fixed", "--training-data-dir", str(folder))
        path = folder / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["phi_reference"]["scale"] = 5.
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "phi reference"):
            ensemble.prepare_training_scaling(args, self.regime)
        metadata["config"]["regime"] = "another_system"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Expected training regime"):
            ensemble.prepare_training_scaling(args, self.regime)

    def test_dadras_preset_uses_decreasing_raw_e_and_original_phi_reference(self):
        config = Path(ensemble.__file__).with_name("dadras_ensemble_config.json")
        args = ensemble.parse_args(["--config", str(config)])
        module, _ = ensemble.single.load_regime_module()
        regime = ensemble.single.resolve_regime(module, args.regime)
        spec = ensemble.simulation_spec(args, regime)
        times = np.arange(args.steps) * args.dt
        parameter = ensemble.single.parameter_at_time(times, spec)
        np.testing.assert_allclose(parameter[[0, -1]], [6.2, 5.2])
        self.assertTrue(np.all(np.diff(parameter) < 0))
        self.assertEqual((args.n_trajectories, args.steps, args.burn_in_steps), (200, 4000, 5000))
        self.assertEqual(spec["phi_boundary"], regime.boundary)
        self.assertEqual(args.normalization, "fixed")
        self.assertEqual(args.model_runs[0].parent.name, "cphi")
        self.assertEqual(args.model_runs[0].parent.parent.name, "20261102_001412_188043_1162805")

    def test_config_paths_are_portable_and_cli_phi_replaces_raw_preset(self):
        args = self.args()
        self.assertEqual(args.output_dir, self.root / "results")
        self.assertEqual(args.model_runs, [self.root / "model"])
        preset = json.loads(self.config.read_text())
        preset.update(parameter_start=1.4, parameter_end=2.09, model_labels=["Preset"])
        self.config.write_text(json.dumps(preset))
        args = self.args("--phi-start", "-0.5", "--phi-end", "0.5", "--model-runs", "other")
        self.assertIsNone(args.model_labels)
        spec = ensemble.simulation_spec(args, self.regime)
        self.assertEqual((spec["parameter_start"], spec["parameter_end"]), (0.75, 1.25))
        with self.assertRaises(ValueError):
            ensemble.simulation_spec(self.args("--parameter-start", "0", "--phi-start", "0"), self.regime)

    def test_ensemble_burn_in_drift_and_seed_independence(self):
        arrays, failures = ensemble.simulate_ensemble(self.module, self.regime, self.spec(), 3, progress=False)
        self.assertFalse(failures)
        self.assertEqual(arrays["observed"].shape, (3, 21, 3))
        self.assertFalse(np.array_equal(arrays["initial_before_burn"][0], arrays["initial_before_burn"][1]))
        # Four fixed-parameter burn-in transitions of duration 0.1, then
        # integral(0.5 + 0.5*t) = 0.5*t + 0.25*t**2 over the 2-unit drift.
        t = arrays["times"]
        change = (0.2 + 0.5 * t + 0.25 * t**2)[:, None] * [1, 2, 3]
        expected = arrays["initial_before_burn"][:, None, :] + change[None]
        np.testing.assert_allclose(arrays["observed"], expected, atol=1e-7)
        repeat, _ = ensemble.simulate_ensemble(self.module, self.regime, self.spec(), 3, progress=False)
        np.testing.assert_array_equal(repeat["observed"], arrays["observed"])

    def test_retry_does_not_shift_later_initial_states(self):
        normal, _ = ensemble.simulate_ensemble(self.module, self.regime, self.spec(), 2, progress=False)
        original = ensemble.single.simulate_ground_truth
        calls = 0

        def transient_failure(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("integration failed")
            return original(*args, **kwargs)

        with patch.object(ensemble.single, "simulate_ground_truth", side_effect=transient_failure):
            retried, failures = ensemble.simulate_ensemble(self.module, self.regime, self.spec(), 2, progress=False)
        self.assertEqual(len(failures), 1)
        np.testing.assert_array_equal(retried["simulation_attempts"], [2, 1])
        np.testing.assert_array_equal(retried["initial_before_burn"][1], normal["initial_before_burn"][1])
        self.assertFalse(np.array_equal(retried["initial_before_burn"][0], normal["initial_before_burn"][0]))

    def test_batched_rollout_matches_individual_rollouts_and_initial_points(self):
        model = ensemble.single.SavedALRNN(self.state(), pwl_units=0)
        initial = np.array([[1, 2, 3], [4, 0, -2], [0, 1, 0]], dtype=np.float32)
        phi = np.array([-1., .5, 2., 9.], dtype=np.float32)
        prediction, failed = ensemble.rollout_ensemble(model, initial, phi, "cpu", batch_size=2)
        np.testing.assert_array_equal(prediction[:, 0], initial)
        np.testing.assert_array_equal(failed, [-1, -1, -1])
        for index, point in enumerate(initial):
            expected, _ = ensemble.single.rollout(model, point, phi, "cpu")
            np.testing.assert_allclose(prediction[index], expected)
        # Parameter-dependent shifted activations must follow the same loader
        # and recurrence as individual evaluation, too.
        shifted_state = self.state()
        shifted_state["Q_phi"] = torch.ones((5, 1)) * 0.2
        shifted_state["V_phi"] = torch.eye(5) * 0.1
        shifted = ensemble.single.SavedALRNN(shifted_state, pwl_units=2)
        prediction, _ = ensemble.rollout_ensemble(shifted, initial, phi, "cpu")
        expected, _ = ensemble.single.rollout(shifted, initial[1], phi, "cpu")
        np.testing.assert_allclose(prediction[1], expected, rtol=1e-6)

    def test_one_diverging_trajectory_does_not_contaminate_others(self):
        class SelectiveFailure:
            N = 3
            phi_dim = 0
            B = torch.zeros((3, 3))

            def __call__(self, latent, phi):
                return torch.where(latent[:, :1] > 1, float("inf"), latent)

        initial = np.array([[0, 1, 0], [2, 1, 0]], dtype=np.float32)
        result, failed = ensemble.rollout_ensemble(SelectiveFailure(), initial, np.zeros(5), "cpu")
        self.assertTrue(np.isfinite(result[0]).all())
        np.testing.assert_array_equal(result[1, 0], initial[1])
        self.assertTrue(np.isnan(result[1, 1:]).all())
        np.testing.assert_array_equal(failed, [-1, 1])

    def test_peak_intervals_never_cross_trajectories_or_missing_sections(self):
        values = np.array([[0, 4, 0, 0, 3, 0, np.nan, 0, 7, 0, 0, 0],
                           [0, 0, 8, 0, 0, 0, 0, 6, 0, 0, 2, 0]])
        times = np.arange(values.shape[1]) * 0.1
        parameter = np.linspace(1.4, 2.09, len(times))
        records = ensemble.peak_records(values, times, parameter, np.zeros(len(times)), "truth")
        first = [r for r in records if r["trajectory"] == 0]
        self.assertEqual([r["step"] for r in first], [1, 4, 8])
        self.assertEqual([r["next_peak_step"] for r in first], [4, None, None])
        self.assertAlmostEqual(first[0]["interval_time"], 0.3)
        self.assertEqual(first[0]["parameter"], parameter[1])
        second = [r for r in records if r["trajectory"] == 1]
        self.assertEqual([r["interval_steps"] for r in second], [5, 3, None])
        path = self.root / "peaks.csv"
        ensemble.save_peak_table(path, records)
        self.assertEqual(len(path.read_text().splitlines()), len(records) + 1)

    def test_plateaus_empty_peaks_and_optional_prominence(self):
        values = np.array([[0, 1, 1, 1, 0, .1, 0]])
        times = np.arange(7)
        records = ensemble.peak_records(values, times, times, times, "truth")
        self.assertEqual([r["step"] for r in records], [2, 5])
        filtered = ensemble.peak_records(values, times, times, times, "truth", prominence=.5)
        self.assertEqual([r["step"] for r in filtered], [2])
        self.assertEqual(ensemble.peak_records(np.zeros((2, 7)), times, times, times, "truth"), [])

    def test_physical_predictions_invert_each_paired_truth_scaling(self):
        x = np.ones((2, 4, 3))
        arrays = dict(mean=np.array([[1, 2, 3], [-1, 0, 1]]), std=np.array([[2, 3, 4], [1, 2, 3]]))
        physical = ensemble.physical_prediction(x, arrays)
        np.testing.assert_array_equal(physical[:, 0], [[3, 5, 7], [0, 2, 4]])

    def test_truth_cache_reuses_without_solver_and_rejects_changed_size(self):
        args = self.args()
        source = self.root / "regimes.py"
        source.write_text("test source")
        with contextlib.redirect_stdout(io.StringIO()):
            arrays, _ = ensemble.ground_truth(args, self.module, self.regime, source)
            with patch.object(ensemble, "simulate_ensemble", side_effect=AssertionError("must reuse")):
                cached, _ = ensemble.ground_truth(args, self.module, self.regime, source)
        np.testing.assert_array_equal(cached["x"], arrays["x"])
        args.n_trajectories = 4
        with self.assertRaisesRegex(ValueError, "cache settings"):
            ensemble.ground_truth(args, self.module, self.regime, source)

    def test_plot_only_uses_recorded_prediction_identity_without_checkpoint(self):
        # Analytic fake ODE and checkpoint-independent cached prediction exercise
        # the full CLI path without training or a GPU.
        source = self.root / "regimes.py"
        source.write_text("test source")
        args = self.args()
        with contextlib.redirect_stdout(io.StringIO()):
            arrays, _ = ensemble.ground_truth(args, self.module, self.regime, source)
        run = args.model_runs[0]
        path = args.output_dir / f"{ensemble.single.model_file_stem(1, run.name)}.npz"
        recorded_hash = "cached-model-sha256"
        ensemble.save_npz(path, dict(prediction=arrays["x"], failed_at=np.full(3, -1)),
                          dict(run_dir=str(run), model_sha256=recorded_hash,
                               ground_truth_sha256=ensemble.single.file_hash(args.ground_truth)))
        with patch.object(ensemble.single, "load_regime_module", return_value=(self.module, source)), \
             patch.object(ensemble.single, "resolve_regime", return_value=self.regime), \
             patch.object(ensemble.single, "load_model", side_effect=AssertionError("must not load")), \
             patch.object(ensemble, "rollout_ensemble", side_effect=AssertionError("must not infer")), \
             patch.object(ensemble, "simulate_ensemble", side_effect=AssertionError("must not simulate")), \
             patch.object(ensemble, "save_plots", return_value=[]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ensemble.main(["--config", str(self.config), "--plot-only"]), 0)
        manifest = json.loads((args.output_dir / "ensemble_sources.json").read_text())
        analysis = manifest["analyses"][manifest["latest_analysis_id"]]
        self.assertEqual(analysis["models"][0]["model_sha256"], recorded_hash)

    def test_provenance_retains_settings_for_other_component_plots(self):
        first = dict(component="z", models=[dict(model_sha256="first")], peak_prominence=0.)
        second = dict(component="x", models=[dict(model_sha256="second")], peak_prominence=.5)
        ensemble.save_provenance(self.root, first)
        ensemble.save_provenance(self.root, second)
        manifest = json.loads((self.root / "ensemble_sources.json").read_text())
        self.assertEqual(len(manifest["analyses"]), 2)
        first_id = manifest["plots"]["maxima_z.png"]["analysis_id"]
        second_id = manifest["plots"]["maxima_x.png"]["analysis_id"]
        self.assertEqual(manifest["analyses"][first_id], first)
        self.assertEqual(manifest["analyses"][second_id], second)
        self.assertEqual(manifest["plots"]["first_trajectory_3d.png"]["analysis_id"], second_id)


if __name__ == "__main__":
    unittest.main()
