from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from bifurcation_training_markers import (
    TrainingParameterRange,
    add_training_range,
    alrnn_training_parameters,
    install_training_markers,
)


class TrainingMarkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_dir = Path(self.temp.name)
        self.regime = SimpleNamespace(name="halvorsen_test")
        self.metadata = dict(regimes=[dict(name=self.regime.name, boundary=1.75, start=1.4, end=2.09)],
                             phi_reference=dict(boundary=1.75, scale=0.35))
        self.settings = dict(multi_trajectory=True, split_idx=100, use_phi=False)

    def save_phi(self, trajectories, *, train_lengths=None, normalized=False, names=None):
        lengths = [len(values) for values in trajectories]
        array = np.full((len(trajectories), max(lengths), 1), np.nan, dtype=np.float32)
        for index, values in enumerate(trajectories):
            values = np.asarray(values)
            array[index, :len(values), 0] = values if normalized else 1.75 + 0.35 * values
        filename = "trajectory_phi.npy" if normalized else "trajectory_phi_raw.npy"
        np.save(self.data_dir / filename, array)
        np.save(self.data_dir / "trajectory_lengths.npy", lengths)
        if train_lengths is not None:
            np.save(self.data_dir / "trajectory_train_lengths.npy", train_lengths)
        if names is not None:
            np.save(self.data_dir / "trajectory_regime_names.npy", names)

    def resolve(self, **settings):
        return alrnn_training_parameters(self.data_dir, self.metadata, self.regime, self.settings | settings)

    def test_compact_parameters_keep_the_same_training_band_or_constant_markers(self):
        for trajectories, lengths in (([np.linspace(-1, 1, 11), np.zeros(8)], [6, 5]),
                                      ([np.full(8, -1), np.full(5, -0.5)], [6, 3])):
            for normalized in (False, True):
                for name in ("trajectory_phi.npy", "trajectory_phi_raw.npy"):
                    (self.data_dir / name).unlink(missing_ok=True)
                self.save_phi(trajectories, train_lengths=lengths, normalized=normalized)
                expected = self.resolve()
                path = self.data_dir / ("trajectory_phi.npy" if normalized else "trajectory_phi_raw.npy")
                padded = np.load(path)
                np.save(path, np.concatenate([padded[i, :len(values)] for i, values in enumerate(trajectories)]))
                actual = self.resolve()
                self.assertIs(type(actual), type(expected))
                np.testing.assert_array_equal(actual, expected)

    def test_incomplete_conversion_blocks_parameter_reading(self):
        self.save_phi([np.zeros(8)], train_lengths=[5])
        (self.data_dir / ".compact_in_progress.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "conversion is incomplete"):
            self.resolve()

    def test_mixed_training_shades_one_range_and_excludes_test_tail(self):
        self.save_phi([np.full(11, -1.0), np.linspace(-1, 1, 11), np.zeros(8)],
                      train_lengths=[8, 6, 5])
        values = self.resolve(split_idx=2)
        self.assertIsInstance(values, TrainingParameterRange)
        np.testing.assert_allclose(values, [1.4, 1.75], atol=1e-7)
        # This is also how the backend logs the result before rendering it.
        self.assertIn("1.4", np.array2string(values, precision=6))

    def test_constants_outside_dynamic_sweep_extend_the_same_band(self):
        self.save_phi([np.linspace(-1, 0, 6), np.full(6, 0.5)])
        values = self.resolve()
        self.assertIsInstance(values, TrainingParameterRange)
        np.testing.assert_allclose(values, [1.4, 1.925], atol=1e-7)

    def test_constant_only_variable_lengths_keep_discrete_values(self):
        self.save_phi([np.full(8, -1), np.full(5, -0.5), np.zeros(6), np.full(4, -0.5)],
                      train_lengths=[6, 3, 5, 2])
        values = self.resolve()
        self.assertNotIsInstance(values, TrainingParameterRange)
        np.testing.assert_allclose(values, [1.4, 1.575, 1.75], atol=1e-7)

    def test_variation_only_in_test_segment_does_not_create_a_band(self):
        self.save_phi([[0, 0, 0, 0.5, 0.75, 1]], train_lengths=[3])
        values = self.resolve()
        self.assertNotIsInstance(values, TrainingParameterRange)
        np.testing.assert_allclose(values, [1.75])

    def test_legacy_split_and_single_trajectory_selection(self):
        self.save_phi([np.full(6, -0.5), np.linspace(-1, 1, 6)])
        values = self.resolve(multi_trajectory=False, trajectory_id=0, split_idx=4)
        self.assertNotIsInstance(values, TrainingParameterRange)
        np.testing.assert_allclose(values, [1.575], atol=1e-7)
        values = self.resolve(multi_trajectory=False, trajectory_id=1, split_idx=3)
        self.assertIsInstance(values, TrainingParameterRange)
        np.testing.assert_allclose(values, [1.4, 1.68], atol=1e-7)

    def test_constant_parameter_filter_excludes_untrained_trajectories(self):
        self.save_phi([np.full(6, -0.5), np.linspace(-1, 1, 6), np.zeros(6)])
        for selection in (dict(train_phi_values=[-0.5]), dict(train_raw_parameter_values=[1.575])):
            with self.subTest(selection=selection):
                values = self.resolve(**selection)
                self.assertNotIsInstance(values, TrainingParameterRange)
                np.testing.assert_allclose(values, [1.575], atol=1e-7)

    def test_normalized_phi_fallback_uses_saved_reference(self):
        self.save_phi([np.linspace(-1, 0, 6)], normalized=True)
        self.metadata["phi_reference"] = dict(boundary=2.0, scale=0.5)
        values = self.resolve()
        self.assertIsInstance(values, TrainingParameterRange)
        np.testing.assert_allclose(values, [1.5, 2.0])

    def test_other_regimes_are_excluded_before_detecting_time_variation(self):
        self.save_phi([np.zeros(6), np.linspace(-1, 1, 6)], names=[self.regime.name, "other_system"])
        self.metadata["regimes"].append(dict(name="other_system", boundary=0, start=-1, end=1))
        values = self.resolve()
        self.assertNotIsInstance(values, TrainingParameterRange)
        np.testing.assert_allclose(values, [1.75])

    def test_invalid_lengths_and_nonfinite_training_values_are_rejected(self):
        self.save_phi([np.zeros(6)], train_lengths=[7])
        with self.assertRaisesRegex(ValueError, "Invalid training/trajectory lengths"):
            self.resolve()
        self.save_phi([[0, np.nan, 0]], train_lengths=[3])
        with self.assertRaisesRegex(ValueError, "Non-finite training parameters"):
            self.resolve()

    def test_legacy_constants_without_time_series_delegate_to_backend(self):
        np.save(self.data_dir / "trajectory_phi_raw_constants.npy", [1.4, 1.75])
        backend = SimpleNamespace(training_parameter_values=Mock(return_value=np.array([1.4, 1.75])),
                                  add_training_markers=Mock())
        original_values, original_markers = backend.training_parameter_values, backend.add_training_markers
        install_training_markers(backend)
        values = backend.training_parameter_values(self.data_dir, self.metadata, self.regime, self.settings, "alrnn")
        original_values.assert_called_once()
        axis = Mock()
        backend.add_training_markers(axis, values, np.array([1.0, 2.0]))
        original_markers.assert_called_once()

    def test_dynamix_discovery_is_preserved_and_range_rendering_is_installed(self):
        backend = SimpleNamespace(training_parameter_values=Mock(return_value=np.array([1.4])),
                                  add_training_markers=Mock())
        original_values, original_markers = backend.training_parameter_values, backend.add_training_markers
        install_training_markers(backend)
        backend.training_parameter_values(self.data_dir, self.metadata, self.regime, self.settings, "dynamix")
        original_values.assert_called_once()
        axis = Mock()
        backend.add_training_markers(axis, TrainingParameterRange(1.4, 1.75), np.array([1.0, 2.0]))
        original_markers.assert_not_called()
        axis.axvline.assert_not_called()
        axis.axvspan.assert_called_once_with(1.4, 1.75, color="tab:blue", alpha=0.12,
                                            linewidth=0, zorder=0, label="Training parameter range")

    def test_band_is_clipped_to_visible_parameter_range(self):
        axis = Mock()
        add_training_range(axis, TrainingParameterRange(-1, 0), np.array([-0.8, -0.2]))
        self.assertEqual(axis.axvspan.call_args.args, (-0.8, -0.2))
        axis = Mock()
        add_training_range(axis, TrainingParameterRange(-1, 0), np.array([1, 2]))
        axis.axvspan.assert_not_called()
        axis.legend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
