import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import compare_model_bifurcation_diagrams as comparison
import rerun_mixed_bifurcations as recovery
import run_dadras_combined_balanced as runner


@dataclass
class Regime:
    name: str
    family: str
    parameter: str
    boundary: float
    start: float
    end: float
    transition_type: str
    description: str
    rhs_key: str
    params: dict
    init_center: tuple
    init_scale: tuple
    observation: str = "identity"


class AdditionalRegimesTests(unittest.TestCase):
    def backend(self, regimes=()):
        return SimpleNamespace(Regime=Regime, RHS={"lorenz": object(), "rossler": object()},
                               default_regimes=lambda: list(regimes))

    def test_old_backend_gets_exact_new_presets_and_original_references(self):
        backend = self.backend()
        comparison.install_additional_regimes(backend)
        presets = {regime.name: regime for regime in backend.default_regimes()}
        lorenz = presets["lorenz_rho_1_to_30"]
        self.assertEqual((lorenz.start, lorenz.end, lorenz.boundary), (1., 30., 24.74))
        self.assertEqual(lorenz.params, {"sigma": 10., "beta": 8. / 3.})
        self.assertEqual(lorenz.init_center, (1., 1., 1.))
        rossler = presets["rossler_c_3_to_6"]
        self.assertEqual((rossler.start, rossler.end, rossler.boundary), (3., 6., 4.3))
        self.assertEqual(rossler.params, {"a": .2, "b": .2})
        self.assertEqual(rossler.init_center, (0., -5., .5))
        comparison.install_additional_regimes(backend)
        self.assertEqual(len(backend.default_regimes()), 2)

    def test_installed_presets_are_not_overwritten(self):
        backend = self.backend()
        comparison.install_additional_regimes(backend)
        original = backend.default_regimes()
        original[0].description = "server's own definition"
        comparison.install_additional_regimes(backend)
        self.assertIs(backend.default_regimes()[0], original[0])
        self.assertEqual(backend.default_regimes()[0].description, "server's own definition")

    def test_backend_without_required_odes_fails_clearly(self):
        backend = self.backend()
        backend.RHS = {}
        with self.assertRaisesRegex(ValueError, "missing ODE"):
            comparison.install_additional_regimes(backend)


class BifurcationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "existing_run"
        self.run.mkdir()
        args = runner.parse_args(["--system", "lorenz_rho", "--training-data-root", str(self.root / "data"),
                                  "--backend-script", str(self.root / "backend.py"), "--gpu-ids", "4", "5"])
        models = runner.variants(args)
        jobs, comparisons = runner.job_definitions(models, runner.EXPERIMENTS[args.system]["datasets"])
        self.plan = dict(args=vars(args), variants={key: vars(model) for key, model in models.items()},
                         training_jobs=jobs, comparison_jobs=comparisons,
                         plot_root=str(self.root / "plots"), bifurcation_dt=.02)
        (self.run / "plan.json").write_text(json.dumps(self.plan, default=str))
        self.argv = ["--run-root", str(self.run), "--gpu-ids", "5"]

    def test_dry_run_replays_only_two_comparisons_with_saved_settings(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(recovery.main([*self.argv, "--dry-run"]), 0)
        text = output.getvalue()
        self.assertIn("2 bifurcation jobs; no training", text)
        self.assertNotIn("train_halvorsen_plain.py", text)
        self.assertEqual(text.count("--model-runs"), 2)
        self.assertEqual(text.count("--gpu-id 5"), 2)
        self.assertEqual(text.count("--dt 0.02"), 2)
        self.assertIn("--parameter-min 1.0 --parameter-max 30.0", text)
        self.assertFalse((self.run / "logs").exists())

    def test_reuse_cache_and_explicit_plot_settings_are_forwarded(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            recovery.main([*self.argv, "--dry-run", "--reuse-cache", "--num-parameters", "12", "--components", "x", "z"])
        text = output.getvalue()
        self.assertNotIn("--force-recompute", text)
        self.assertIn("--num-parameters 12 --components x z", text)

    def test_x_only_uses_original_cache_and_separate_plot_names(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            recovery.main([*self.argv, "--dry-run", "--reuse-cache", "--components", "x"])
        text = output.getvalue()
        self.assertEqual(text.count("--components x"), 2)
        self.assertEqual(text.count("bifurcation_compare_x.png"), 2)
        self.assertEqual(text.count("bifurcation_cache.npz"), 2)
        self.assertNotIn("--force-recompute", text)

    def test_missing_checkpoint_is_rejected_before_cuda_or_scheduling(self):
        with patch.object(runner, "prepare_data", return_value=.02), \
             patch.object(runner, "schedule", side_effect=AssertionError("must not schedule")):
            with self.assertRaisesRegex(FileNotFoundError, "Missing trained model"):
                recovery.main(self.argv)

    def test_retry_preserves_checkpoints_and_original_logs_and_summary(self):
        import torch
        for variant, dataset in self.plan["training_jobs"].values():
            model = self.run / variant / dataset
            model.mkdir(parents=True)
            (model / "model.pt").write_bytes(b"existing trained weights")
            (model / "metrics.json").write_text("{}")
        (self.run / "summary.json").write_text("original summary")
        (self.run / "logs").mkdir()
        (self.run / "logs/cphi__bifurcation.log").write_text("original error")
        (self.run / "gpu_ids.json").write_text("[4, 5]")
        before = {path: path.read_bytes() for path in self.run.rglob("*") if path.is_file()}

        def schedule(gpu_file, gpu_ids, device_count, log_dir, steps, env, job_names, max_gpus):
            self.assertEqual(json.loads(gpu_file.read_text()), [5])
            self.assertEqual(job_names, list(self.plan["comparison_jobs"]))
            for job in job_names:
                stage = steps(job, 5)[0]
                self.assertIn("compare_model_bifurcation_diagrams.py", stage.command[2])
                self.assertNotIn("train_halvorsen_plain.py", " ".join(stage.command))
            return dict.fromkeys(job_names, 0)

        with patch.object(runner, "prepare_data", return_value=.02), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=6), \
             patch.object(runner, "schedule", side_effect=schedule), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recovery.main(self.argv), 0)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(len(list((self.run / "logs").glob("bifurcation_retry_*/summary.json"))), 1)


if __name__ == "__main__":
    unittest.main()
