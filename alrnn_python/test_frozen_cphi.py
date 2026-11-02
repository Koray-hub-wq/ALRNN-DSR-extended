import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from alrnn_model import AL_RNN, SavedALRNN, load_saved_model


class FrozenCphiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "argv", ["train_halvorsen_plain.py"]):
            cls.training = runpy.run_path(str(Path(__file__).with_name("train_halvorsen_plain.py")))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.base = AL_RNN(6, 2, 3, phi_terms=[])
        self.target = AL_RNN(6, 2, 3, phi_dim=1, phi_terms=["additive"])
        self.save_base()

    def save_base(self, *, reference=None, model=None):
        model = self.base if model is None else model
        torch.save(model.state_dict(), self.run / "model.pt")
        metrics = {"args": {"latent_dim": model.M, "pwl_units": model.P,
                            "train_phi_values": [-.75] if reference is None else reference,
                            "phi_terms": list(model.phi_terms)},
                   "phi_dim": model.phi_dim, "model_config": model.model_config()}
        (self.run / "metrics.json").write_text(json.dumps(metrics))

    def initialize(self):
        return self.training["initialize_frozen_cphi"](self.target, self.run)

    def test_zero_initialized_c_matches_base_at_every_phi(self):
        transfer = self.initialize()
        self.assertEqual(transfer["reference_phi"], -.75)
        self.assertEqual(len(transfer["base_checkpoint_sha256"]), 64)
        z = torch.randn(4, 6)
        for phi in (-1., -.75, 0., 1.):
            torch.testing.assert_close(self.target(z, torch.full((4, 1), phi)), self.base(z), rtol=0, atol=0)
        for name in ("A", "W", "h", "B"):
            self.assertFalse(getattr(self.target, name).requires_grad)
            self.assertTrue(torch.equal(getattr(self.target, name), getattr(self.base, name)))
        self.assertTrue(self.target.C.requires_grad)

    def test_reference_dynamics_are_unchanged_even_after_c_changes(self):
        self.initialize()
        with torch.no_grad():
            self.target.C.fill_(3.)
        z = torch.randn(5, 6)
        torch.testing.assert_close(self.target(z, torch.full((5, 1), -.75)), self.base(z), rtol=0, atol=0)
        self.assertFalse(torch.equal(self.target(z, torch.zeros(5, 1)), self.base(z)))

    def test_only_c_changes_after_recurrent_training_update(self):
        transfer = self.initialize()
        before = {name: value.detach().clone() for name, value in self.target.named_parameters()}
        optimizer = torch.optim.SGD([p for p in self.target.parameters() if p.requires_grad], lr=.1)
        x = torch.randn(4, 12, 3)
        phi = torch.zeros(4, 12, 1)
        prediction = self.training["predict_sequence_using_gtf"](self.target, x, 1, 4, phi)
        (prediction[:, :, :3] - torch.randn(4, 12, 3)).square().mean().backward()
        self.assertIsNotNone(self.target.C.grad)
        self.assertGreater(self.target.C.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(self.target.C, before["C"]))
        for name in ("A", "W", "h", "B"):
            self.assertIsNone(getattr(self.target, name).grad)
            self.assertTrue(torch.equal(getattr(self.target, name), before[name]))
        self.training["verify_frozen_weights"](self.target, transfer)
        self.assertTrue(transfer["frozen_weights_unchanged"])

    def test_checkpoint_roundtrip_keeps_reference_and_predictions(self):
        self.initialize()
        with torch.no_grad():
            self.target.C.fill_(.3)
        saved = SavedALRNN(self.target.state_dict(), 2)
        z, phi = torch.randn(4, 6), torch.randn(4, 1)
        torch.testing.assert_close(self.target(z, phi), saved(z, phi), rtol=0, atol=0)
        self.assertEqual(saved.model_config()["phi_reference"], [-.75])
        self.save_base(model=self.target)
        loaded, _, _ = load_saved_model(self.run)
        torch.testing.assert_close(loaded(z, phi), saved(z, phi), rtol=0, atol=0)

    def test_each_adaptation_restarts_from_same_base_and_zero_c(self):
        first = self.initialize()
        with torch.no_grad():
            self.target.C.fill_(4.)
        second = self.initialize()
        self.assertTrue(torch.equal(self.target.C, torch.zeros_like(self.target.C)))
        self.assertEqual(first["base_checkpoint_sha256"], second["base_checkpoint_sha256"])

    def test_invalid_base_reference_or_architecture_is_rejected(self):
        for reference in ([], [-1., -.75], [float("nan")]):
            with self.subTest(reference=reference):
                self.save_base(reference=reference)
                with self.assertRaisesRegex(ValueError, "exactly one finite"):
                    self.initialize()
        self.save_base(model=AL_RNN(7, 2, 3, phi_terms=[]))
        with self.assertRaisesRegex(ValueError, "identical"):
            self.initialize()

    def test_nonvanilla_base_is_rejected(self):
        self.save_base(model=self.target)
        with self.assertRaisesRegex(ValueError, "vanilla"):
            self.initialize()

    def test_frozen_weight_tampering_is_detected(self):
        transfer = self.initialize()
        with torch.no_grad():
            self.target.W[0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "weights changed"):
            self.training["verify_frozen_weights"](self.target, transfer)

    def test_parser_requires_only_additive_and_not_eval_only(self):
        parser = self.training["parse_args"]
        self.assertEqual(parser(["--frozen-base-run", str(self.run), "--phi-terms", "additive"]).phi_terms, ["additive"])
        for flags in (["--phi-terms", "none"], ["--phi-terms", "diagonal"],
                      ["--phi-terms", "additive", "shifted-relu"], ["--phi-terms", "additive", "--eval-only"]):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                parser(["--frozen-base-run", str(self.run), *flags])

    def test_old_checkpoints_do_not_gain_a_reference_buffer(self):
        for model in (self.base, self.target):
            self.assertNotIn("phi_reference", model.state_dict())
            self.assertNotIn("phi_reference", model.model_config())
            SavedALRNN(model.state_dict(), model.P)


if __name__ == "__main__":
    unittest.main()
