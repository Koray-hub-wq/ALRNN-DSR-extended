import json
from itertools import combinations
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from alrnn_model import AL_RNN, SavedALRNN, load_saved_model
from parameter_terms import PARAMETER_TERMS, ParameterTerm, resolve_phi_terms

TERM_NAMES = ("additive", "diagonal", "diagonal-relu", "shifted-relu")
CONFIGURATIONS = tuple(terms for count in range(len(TERM_NAMES) + 1) for terms in combinations(TERM_NAMES, count))


class ALRNNModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "argv", ["train_halvorsen_plain.py"]):
            cls.training = runpy.run_path(str(Path(__file__).with_name("train_halvorsen_plain.py")))

    def make_model(self, terms, phi_dim=1):
        np.random.seed(13)
        torch.manual_seed(13)
        return AL_RNN(6, 2, 3, phi_dim=phi_dim, phi_terms=terms)

    def test_model_selection_parser_defaults_and_override(self):
        parse_args = self.training["parse_args"]
        self.assertTrue(parse_args([]).use_best_model)
        self.assertFalse(parse_args(["--no-best-model"]).use_best_model)

    def test_model_selection_restores_best_or_keeps_last_without_disabling_monitoring(self):
        train = self.training["train_sh"]
        for use_best, expected in ((True, 0.2), (False, 0.36)):
            with self.subTest(use_best_model=use_best):
                model = torch.nn.Linear(1, 1, bias=False)
                model.N = 1
                with torch.no_grad():
                    model.weight.zero_()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
                x = torch.ones(1, 2, 1)
                dataset = Mock()
                dataset.sample_batch.return_value = (x, x, None)
                dataset.eval_reference.return_value = (x[0], None)
                dstsp = Mock(side_effect=[1.0, 2.0])
                replacements = {
                    "predict_sequence_using_gtf": lambda model, x, *args: model(x),
                    "predict_free_sequence": lambda *args: torch.zeros(1, 2, 1),
                    "state_space_divergence_binning": dstsp,
                    "power_spectrum_error": Mock(return_value=0.0),
                }
                with patch.dict(train.__globals__, replacements):
                    metrics = train(
                        model, dataset, optimizer, scheduler, torch.nn.MSELoss(),
                        num_epochs=2, alpha=1, n_interleave=1, device="cpu",
                        batches_per_epoch=1, ssi=1, use_best_model=use_best,
                    )
                self.assertAlmostEqual(model.weight.item(), expected, places=6)
                self.assertEqual(metrics["dstsp"], [1.0, 2.0])
                self.assertEqual(dstsp.call_count, 2)
                self.assertEqual(metrics["metric_steps"], [2, 2])
                self.assertEqual(metrics["metric_epochs"], [1, 2])

    def test_short_test_sections_are_skipped_and_actual_metric_lengths_are_saved(self):
        main = self.training["main"]
        for mode, transient in (("mixed", 12), ("mixed", 100), ("mixed", 0),
                                ("single", 100), ("multi", 100)):
            with self.subTest(mode=mode, transient=transient), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data_dir = root / "data"
                output = root / "output"
                data_dir.mkdir()
                data = np.random.default_rng(17).normal(size=(3, 48, 3)).astype(np.float32)
                np.save(data_dir / "trajectories.npy", data)
                if mode == "mixed":
                    np.save(data_dir / "trajectory_lengths.npy", np.array([40, 44, 48]))
                    np.save(data_dir / "trajectory_train_lengths.npy", np.array([30, 30, 30]))
                    np.save(data_dir / "trajectory_group_names.npy", np.array(["a", "b", "c"]))
                flags = ["--data-dir", str(data_dir), "--output-dir", str(output),
                         "--force-cpu", "--latent-dim", "6", "--pwl-units", "2",
                         "--split-idx", "30", "--sequence-length", "4",
                         "--t-gen", "5", "--t-transient", str(transient)]
                if mode != "single":
                    flags.append("--multi-trajectory")
                parsed = self.training["parse_args"](flags)
                dstsp = Mock(return_value=1.25)
                replacements = {
                    "args": parsed,
                    "train_sh": Mock(return_value={"losses": [], "dstsp": [], "dh": []}),
                    "state_space_divergence_binning": dstsp,
                    "power_spectrum_error": Mock(return_value=0.25),
                }
                with patch.dict(main.__globals__, replacements), \
                     patch.object(torch, "set_num_interop_threads"):
                    main()
                self.assertTrue((output / "model.pt").is_file())
                saved = json.loads((output / "metrics.json").read_text())
                results = saved["evaluation"]
                expected_steps = [0, 2, 5] if transient == 12 else ([5, 5, 5] if transient == 0 else [0] * len(results))
                self.assertEqual([result["metric_steps"] for result in results], expected_steps)
                self.assertEqual(dstsp.call_count, sum(count > 0 for count in expected_steps))
                if mode == "mixed":
                    self.assertEqual([result["test_steps_available"] for result in results], [10, 14, 18])
                for result, count in zip(results, expected_steps):
                    plot = output / f'{result["file_stem"]}_free_rollout_3d.png'
                    self.assertEqual(plot.exists(), count > 0)
                    if count:
                        self.assertEqual(result["status"], "completed")
                        self.assertEqual(result["metric_stop_step_exclusive"], transient + count)
                    else:
                        self.assertEqual(result["status"], "skipped")
                        self.assertEqual(result["skip_reason"], "insufficient_test_samples_after_transient")
                        self.assertIsNone(result["dstsp"])
                        self.assertIsNone(result["dh"])
                self.assertEqual(saved["primary_metric_steps"], expected_steps[0])
                if not expected_steps[0]:
                    self.assertIsNone(saved["primary_dstsp"])

    def test_parser_rejects_negative_transient_and_too_short_requested_horizon(self):
        for flags in (["--t-transient", "-1"], ["--t-gen", "1"]):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.training["parse_args"](flags)

    def test_all_sixteen_configurations_match_explicit_equation(self):
        z = torch.tensor([[1., -2., 3., -4., -5., 6.], [-2., 3., -1., 2., 4., -6.]])
        phi = torch.tensor([[0.75], [-0.5]])
        for terms in CONFIGURATIONS:
            with self.subTest(terms=terms):
                model = self.make_model(terms)
                with torch.no_grad():
                    if model.C is not None:
                        model.C.copy_(torch.arange(6).reshape(6, 1) / 10)
                    if model.D_phi is not None:
                        model.D_phi.fill_(0.3)
                    if model.D_relu_phi is not None:
                        model.D_relu_phi.fill_(0.7)
                    if model.Q_phi is not None:
                        model.Q_phi.fill_(0.4)
                        model.V_phi.copy_(torch.arange(36).reshape(6, 6) / 100)
                activated = z.clone()
                activated[:, -2:] = torch.relu(activated[:, -2:])
                expected = model.A * z + activated @ model.W.t() + model.h
                if "additive" in terms:
                    expected = expected + phi @ model.C.t()
                if "diagonal" in terms:
                    expected = expected + (phi @ model.D_phi.t()) * z
                if "diagonal-relu" in terms:
                    expected = expected + (phi @ model.D_relu_phi.t()) * activated
                if "shifted-relu" in terms:
                    shifted = z + phi @ model.Q_phi.t()
                    shifted[:, -2:] = torch.relu(shifted[:, -2:])
                    expected = expected + shifted @ model.V_phi.t()
                original = z.clone()
                torch.testing.assert_close(model(z, phi if terms else None), expected)
                torch.testing.assert_close(z, original)
                self.assertEqual("C" in model.state_dict(), "additive" in terms)
                self.assertEqual("D_phi" in model.state_dict(), "diagonal" in terms)
                self.assertEqual("D_relu_phi" in model.state_dict(), "diagonal-relu" in terms)
                self.assertEqual("Q_phi" in model.state_dict(), "shifted-relu" in terms)
                self.assertEqual("V_phi" in model.state_dict(), "shifted-relu" in terms)
                self.assertEqual(model.phi_dim, 1 if terms else 0)

    def test_zero_terms_preserve_vanilla_output_and_initialization(self):
        base = self.make_model(())
        both = self.make_model(("additive", "diagonal", "diagonal-relu"))
        for name in ("A", "W", "h", "B"):
            torch.testing.assert_close(getattr(base, name), getattr(both, name), rtol=0, atol=0)
        z = torch.randn(5, 6)
        torch.testing.assert_close(base(z), both(z, torch.randn(5, 1)), rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in both.parameters()) - sum(p.numel() for p in base.parameters()), 18)

    def test_diagonal_uses_unactivated_negative_state_and_multiple_phi_components(self):
        model = self.make_model(("diagonal",), phi_dim=2)
        with torch.no_grad():
            model.A.zero_()
            model.W.zero_()
            model.h.zero_()
            model.D_phi.copy_(torch.tensor([[1., 2.]]).repeat(6, 1))
        z = -torch.ones(2, 6)
        phi = torch.tensor([[1., 2.], [3., -1.]])
        torch.testing.assert_close(model(z, phi), torch.tensor([[-5.] * 6, [-1.] * 6]))

    def test_gradient_reaches_each_enabled_term_and_encoder_through_teacher_forcing(self):
        predict = self.training["predict_sequence_using_gtf"]
        for terms in (terms for terms in CONFIGURATIONS if terms):
            with self.subTest(terms=terms):
                model = self.make_model(terms)
                x = torch.randn(4, 9, 3)
                phi = torch.full((4, 9, 1), 0.5)
                prediction = predict(model, x, alpha=1, n_interleave=3, phi=phi)
                loss = (prediction[:, :, :3] - torch.randn(4, 9, 3)).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                term_weights = [weight.name for term in terms for weight in PARAMETER_TERMS[term].weights]
                for name in ("A", "W", "h", "B", *term_weights):
                    gradient = getattr(model, name).grad
                    self.assertIsNotNone(gradient, name)
                    self.assertTrue(torch.isfinite(gradient).all(), name)
                    self.assertGreater(float(gradient.abs().sum()), 0, name)

    def test_checkpoint_roundtrip_for_all_configurations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for terms in CONFIGURATIONS:
                with self.subTest(terms=terms):
                    model = self.make_model(terms)
                    with torch.no_grad():
                        for term in terms:
                            for weight in PARAMETER_TERMS[term].weights:
                                getattr(model, weight.name).fill_(0.2)
                    torch.save(model.state_dict(), path / "model.pt")
                    metadata = dict(args=dict(latent_dim=6, pwl_units=2, phi_terms=list(terms)),
                                    phi_dim=model.phi_dim, model_config=model.model_config())
                    (path / "metrics.json").write_text(json.dumps(metadata))
                    loaded, _, _ = load_saved_model(path)
                    self.assertEqual(loaded.phi_terms, terms)
                    z, phi = torch.randn(3, 6), torch.randn(3, 1)
                    torch.testing.assert_close(model(z, phi), loaded(z, phi), rtol=0, atol=0)

    def test_legacy_additive_checkpoint_and_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model = self.make_model(("additive",))
            torch.save(model.state_dict(), path / "model.pt")
            metadata = dict(args=dict(latent_dim=6, pwl_units=2, use_phi=True), phi_dim=1)
            (path / "metrics.json").write_text(json.dumps(metadata))
            loaded, _, _ = load_saved_model(path)
            self.assertEqual(loaded.phi_terms, ("additive",))
            metadata["model_config"] = dict(model.model_config(), phi_terms=["diagonal"])
            (path / "metrics.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "phi_terms"):
                load_saved_model(path)

    def test_phi_shapes_and_inconsistent_checkpoint_dimensions_are_rejected(self):
        model = self.make_model(("diagonal",), phi_dim=2)
        for phi in (None, torch.ones(3), torch.ones(3, 1), torch.ones(2, 2)):
            with self.assertRaisesRegex(ValueError, "phi_t"):
                model(torch.ones(3, 6), phi)
        state = self.make_model(("additive", "diagonal")).state_dict()
        state["D_phi"] = torch.zeros(6, 2)
        with self.assertRaisesRegex(ValueError, "same phi_dim"):
            SavedALRNN(state, 2)

    def test_parser_selection_legacy_alias_and_explicit_override(self):
        parse = self.training["parse_args"]
        for flags, expected in (([], []), (["--use-phi"], ["additive"]),
                                (["--phi-terms", "diagonal"], ["diagonal"]),
                                (["--phi-terms", "additive", "diagonal"], ["additive", "diagonal"]),
                                (["--phi-terms", "diagonal-relu"], ["diagonal-relu"]),
                                (["--phi-terms", "shifted-relu"], ["shifted-relu"]),
                                (["--phi-terms", *TERM_NAMES], list(TERM_NAMES)),
                                (["--use-phi", "--phi-terms", "none"], [])):
            args = parse(flags)
            self.assertEqual(args.phi_terms, expected)
            self.assertEqual(args.use_phi, bool(expected))
        for requested in (("none", "additive"), ("diagonal", "diagonal"), ("unknown",)):
            with self.assertRaises(ValueError):
                resolve_phi_terms(requested)

    def test_registry_extension_is_used_by_model_parser_and_checkpoint(self):
        term = ParameterTerm("test_weights", lambda z, phi, weight: phi @ weight.t())
        with patch.dict(PARAMETER_TERMS, test_extra=term):
            args = self.training["parse_args"](["--phi-terms", "test_extra"])
            model = self.make_model(args.phi_terms)
            with torch.no_grad():
                model.test_weights.fill_(0.7)
            restored = SavedALRNN(model.state_dict(), 2)
            self.assertEqual(restored.phi_terms, ("test_extra",))
            z, phi = torch.randn(3, 6), torch.randn(3, 1)
            torch.testing.assert_close(model(z, phi), restored(z, phi))

    def test_relu_coupling_applies_relu_only_to_last_p_dimensions(self):
        for P in (0, 2, 6):
            with self.subTest(P=P):
                model = AL_RNN(6, P, 3, phi_dim=2, phi_terms=["diagonal-relu"])
                with torch.no_grad():
                    model.A.zero_()
                    model.W.zero_()
                    model.h.zero_()
                    model.D_relu_phi.copy_(torch.tensor([[1., 2.]]).repeat(6, 1))
                z = torch.tensor([[-1., 2., -3., 4., -5., 6.]])
                expected = z.clone()
                if P:
                    expected[:, -P:] = torch.relu(expected[:, -P:])
                torch.testing.assert_close(model(z, torch.tensor([[1., 2.]])), 5 * expected)
                self.assertIsNone(model.D_phi)
                self.assertIsNone(model.C)

    def test_raw_and_relu_coupling_weights_are_independent(self):
        model = self.make_model(("diagonal", "diagonal-relu"))
        with torch.no_grad():
            model.A.zero_()
            model.W.zero_()
            model.h.zero_()
            model.D_phi.fill_(2)
            model.D_relu_phi.fill_(3)
        z = torch.tensor([[-1., 2., -3., 4., -5., 6.]])
        expected = torch.tensor([[-5., 10., -15., 20., -10., 30.]])
        torch.testing.assert_close(model(z, torch.ones(1, 1)), expected)
        self.assertNotEqual(model.D_phi.data_ptr(), model.D_relu_phi.data_ptr())

    def test_shifted_relu_shifts_before_activation_and_projects_after_it(self):
        for P in (0, 2, 6):
            with self.subTest(P=P):
                model = AL_RNN(6, P, 3, phi_dim=2, phi_terms=["shifted-relu"])
                with torch.no_grad():
                    model.A.zero_()
                    model.W.zero_()
                    model.h.zero_()
                    model.Q_phi.copy_(torch.tensor([[1., 0.5]]).repeat(6, 1))
                    model.V_phi.copy_(torch.arange(36).reshape(6, 6) / 100)
                z = torch.tensor([[-1., 2., -3., 4., -0.25, 6.]])
                original = z.clone()
                shifted = z + 0.5
                if P:
                    shifted[:, -P:] = torch.relu(shifted[:, -P:])
                expected = shifted @ model.V_phi.t()
                torch.testing.assert_close(model(z, torch.tensor([[1., -1.]])), expected)
                torch.testing.assert_close(z, original)

    def test_shifted_relu_initialization_allows_immediate_shift_gradients(self):
        model = self.make_model(("shifted-relu",), phi_dim=2)
        self.assertTrue(torch.count_nonzero(model.Q_phi) == 0)
        self.assertGreater(float(torch.count_nonzero(model.V_phi)), 0)
        self.assertNotEqual(model.V_phi.data_ptr(), model.W.data_ptr())
        loss = model(torch.ones(3, 6), torch.ones(3, 2)).square().mean()
        loss.backward()
        for name in ("Q_phi", "V_phi"):
            self.assertGreater(float(getattr(model, name).grad.abs().sum()), 0)
        base = self.make_model(())
        extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters())
        self.assertEqual(extra, 6 * 6 + 6 * 2)
        self.assertIsNone(base.Q_phi)
        self.assertIsNone(base.V_phi)

    def test_shifted_relu_checkpoint_requires_both_weights_and_valid_projection(self):
        model = self.make_model(("shifted-relu",))
        for missing in ("Q_phi", "V_phi"):
            state = dict(model.state_dict())
            del state[missing]
            with self.assertRaisesRegex(ValueError, "Incomplete shifted-relu"):
                SavedALRNN(state, 2)
        state = dict(model.state_dict())
        state["V_phi"] = torch.zeros(6, 5)
        with self.assertRaisesRegex(ValueError, "V_phi"):
            SavedALRNN(state, 2)


if __name__ == "__main__":
    unittest.main()
