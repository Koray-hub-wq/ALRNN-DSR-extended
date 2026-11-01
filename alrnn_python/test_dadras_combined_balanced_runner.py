import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_dadras_combined_balanced as runner
import run_term_ablations as shared


class CombinedDadrasRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "argv", ["train_halvorsen_plain.py"]):
            cls.trainer = runpy.run_path(str(Path(__file__).with_name("train_halvorsen_plain.py")))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        for name in ("train_halvorsen_plain.py", "mixed_trajectory_dataset.py", "trajectory_storage.py",
                     "compare_model_bifurcation_diagrams.py"):
            (self.scripts / name).touch()
        self.backend = self.root / "backend.py"
        self.backend.touch()
        self.data_root = self.root / "data"
        for name in runner.DATASETS:
            data = self.data_root / name
            data.mkdir(parents=True)
            metadata = {"format": "alrnn_mixed_parameter_trajectories_v2",
                        "config": {"regime": runner.REGIME, "standardization": "global", "dt": 0.01},
                        "phi_reference": {"boundary": 5.65, "scale": 0.75}}
            (data / "metadata.json").write_text(json.dumps(metadata))
            np.save(data / "trajectories.npy", np.zeros((1002, 3), dtype=np.float32))
            np.save(data / "trajectory_phi.npy", np.zeros((1002, 1), dtype=np.float32))
            np.save(data / "trajectory_lengths.npy", [401, 601])
            np.save(data / "trajectory_train_lengths.npy", [320, 480])
            np.save(data / "trajectory_group_names.npy", ["constant", "sinusoidal"])
            np.save(data / "trajectory_mean.npy", np.zeros((2, 1, 3), dtype=np.float32))
            np.save(data / "trajectory_std.npy", np.ones((2, 1, 3), dtype=np.float32))
        self.argv = ["--training-data-root", str(self.data_root), "--backend-script", str(self.backend)]
        self.args = runner.parse_args(self.argv)

    def test_defaults_and_one_to_six_distinct_gpu_ids(self):
        models = runner.variants(self.args)
        self.assertEqual((models["cphi"].latent_dim, models["cphi"].pwl_units), (300, 150))
        self.assertEqual(models["cphi"].terms, ("additive",))
        self.assertEqual((models["cphi_shifted_relu"].latent_dim, models["cphi_shifted_relu"].pwl_units), (200, 100))
        self.assertEqual(models["cphi_shifted_relu"].terms, ("additive", "shifted-relu"))
        self.assertEqual((self.args.epochs, self.args.time_sampling_bins), (1000, 10))
        for count in range(1, 7):
            values = list(range(count))
            self.assertEqual(runner.parse_args([*self.argv, "--gpu-ids", *map(str, values)]).gpu_ids, values)
        for values in ([0, 0], [-1], list(range(7))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                runner.parse_args([*self.argv, "--gpu-ids", *map(str, values)])

    def test_invalid_training_and_plot_settings(self):
        for flags in (["--cphi-pwl-units", "301"], ["--shifted-latent-dim", "2"],
                      ["--time-sampling-bins", "0"], ["--epochs", "0"],
                      ["--parameter-min", "6.3"], ["--parameter-max", "nan"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args([*self.argv, *flags])

    def test_preflight_accepts_current_data_and_rejects_incomplete_or_incompatible_data(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.prepare_data(self.args, self.scripts), 0.01)
        folder = self.data_root / next(iter(runner.DATASETS))
        marker = folder / ".compact_in_progress.json"
        marker.write_text("{}")
        with self.assertRaisesRegex(ValueError, "conversion is incomplete"):
            runner.prepare_data(self.args, self.scripts)
        marker.unlink()
        np.save(folder / "trajectory_train_lengths.npy", [100, 480])
        with self.assertRaisesRegex(ValueError, "sequence-length"):
            runner.prepare_data(self.args, self.scripts)
        np.save(folder / "trajectory_train_lengths.npy", [320, 480])
        meta_path = folder / "metadata.json"
        metadata = json.loads(meta_path.read_text())
        metadata["config"]["dt"] = 0.02
        meta_path.write_text(json.dumps(metadata))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "same simulation dt"):
            runner.prepare_data(self.args, self.scripts)

    def test_training_commands_use_current_trainer_and_isolated_outputs(self):
        models = runner.variants(self.args)
        jobs, comparisons = runner.job_definitions(models)
        self.assertEqual(len(jobs), 6)
        self.assertEqual(len(comparisons), 2)
        self.assertEqual(list(jobs.values())[:2], [(name, next(iter(runner.DATASETS))) for name in models])
        outputs = set()
        for variant_name, dataset in jobs.values():
            variant = models[variant_name]
            step = runner.training_step(self.args, self.scripts, self.root / "runs", variant_name, variant, dataset, 5)
            parsed = self.trainer["parse_args"](step.command[3:])
            self.assertTrue(parsed.multi_trajectory and parsed.mixed_balanced_sampling)
            self.assertEqual(parsed.trajectory_sampling, "length")
            self.assertEqual(parsed.time_sampling_bins, 10)
            self.assertEqual((parsed.latent_dim, parsed.pwl_units), (variant.latent_dim, variant.pwl_units))
            self.assertEqual(parsed.phi_terms, list(variant.terms))
            self.assertFalse(parsed.use_best_model)
            self.assertEqual(parsed.gpu_id, 5)
            self.assertEqual(Path(parsed.data_dir), self.data_root / dataset)
            self.assertEqual(Path(parsed.output_dir), self.root / "runs" / variant_name / dataset)
            outputs.add(parsed.output_dir)
        self.assertEqual(len(outputs), 6)

    def test_comparisons_include_three_models_ground_truth_coordinates_and_requested_sweep(self):
        for name, variant in runner.variants(self.args).items():
            step = runner.comparison_step(self.args, self.scripts, self.root / "runs", self.root / "plots",
                                          name, variant, 3, 0.01)
            command = step.command
            self.assertIn("--use-training-normalization", command)
            self.assertEqual(command[command.index("--model-type") + 1], "alrnn")
            self.assertEqual(command[command.index("--regime") + 1], runner.REGIME)
            self.assertEqual(command[command.index("--parameter-min") + 1], "4.9")
            self.assertEqual(command[command.index("--parameter-max") + 1], "6.2")
            self.assertEqual(command[command.index("--components") + 1], "z")
            self.assertEqual(command[command.index("--model-runs") + 1:command.index("--training-data-dirs")],
                             [str(self.root / "runs" / name / dataset) for dataset in runner.DATASETS])
            self.assertEqual(command[command.index("--training-data-dirs") + 1:command.index("--model-labels")],
                             [str(self.data_root / dataset) for dataset in runner.DATASETS])
            self.assertEqual(Path(command[command.index("--output") + 1]).parent, self.root / "plots" / name)
            self.assertEqual(Path(command[command.index("--cache") + 1]).parent, self.root / "plots" / name)
            self.assertEqual(step.cwd, self.backend.parent)
            self.assertNotIn("--no-ground-truth", command)

    def test_failed_training_only_skips_its_own_variant_comparison(self):
        jobs, comparisons = runner.job_definitions(runner.variants(self.args))
        results = dict.fromkeys(jobs, 0)
        results[next(iter(jobs))] = 7
        with patch.object(runner, "schedule", side_effect=[results, {"cphi_shifted_relu__bifurcation": 0}]) as scheduler:
            with contextlib.redirect_stdout(io.StringIO()):
                final, skipped = runner.run_phases(self.root / "gpus.json", [0, 1], 6, self.root, None, {}, jobs, comparisons)
        self.assertEqual(skipped, ["cphi__bifurcation"])
        self.assertEqual(final["cphi_shifted_relu__bifurcation"], 0)
        self.assertEqual(scheduler.call_args_list[1].kwargs["job_names"], ["cphi_shifted_relu__bifurcation"])
        self.assertTrue(all(call.kwargs["max_gpus"] == 6 for call in scheduler.call_args_list))

    def test_all_failed_trainings_start_no_comparisons(self):
        jobs, comparisons = runner.job_definitions(runner.variants(self.args))
        with patch.object(runner, "schedule", return_value=dict.fromkeys(jobs, 1)) as scheduler:
            with contextlib.redirect_stdout(io.StringIO()):
                results, skipped = runner.run_phases(self.root / "gpus.json", [0], 6, self.root, None, {}, jobs, comparisons)
        self.assertEqual(scheduler.call_count, 1)
        self.assertEqual(skipped, list(comparisons))
        self.assertEqual(len(results), 8)

    def test_dry_run_needs_no_data_or_cuda_and_writes_nothing(self):
        with patch.object(runner, "__file__", str(self.scripts / "run_dadras_combined_balanced.py")):
            with patch.object(runner, "prepare_data") as prepare, contextlib.redirect_stdout(io.StringIO()) as output:
                code = runner.main(["--training-data-root", str(self.root / "missing"),
                                    "--backend-script", str(self.root / "also_missing.py"),
                                    "--gpu-ids", "0", "1", "2", "3", "4", "5", "--dry-run"])
        self.assertEqual(code, 0)
        prepare.assert_not_called()
        self.assertEqual(output.getvalue().count("--mixed-balanced-sampling"), 6)
        self.assertEqual(output.getvalue().count("--phi-terms additive shifted-relu"), 3)
        self.assertEqual(output.getvalue().count("--model-runs"), 2)
        self.assertFalse((self.scripts / "runs").exists())

    def test_full_queue_on_six_gpus_saves_all_jobs_and_comparisons(self):
        # Exercise real child-process scheduling with tiny CPU stand-ins for
        # expensive GPU training/plots. Plot stand-ins require all model files.
        with contextlib.redirect_stdout(io.StringIO()):
            runner.prepare_data(self.args, self.scripts)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 6,
            get_device_name=lambda gpu: f"mock GPU {gpu}",
        ))
        original_training, original_comparison = runner.training_step, runner.comparison_step

        def short_training(*args, **kwargs):
            step = original_training(*args, **kwargs)
            output = Path(step.command[step.command.index("--output-dir") + 1])
            code = f"from pathlib import Path; p=Path({str(output)!r}); p.mkdir(parents=True); (p/'model.pt').write_bytes(b'mock')"
            step.command = [sys.executable, "-c", code]
            return step

        def short_comparison(*args, **kwargs):
            step = original_comparison(*args, **kwargs)
            command = step.command
            inputs = command[command.index("--model-runs") + 1:command.index("--training-data-dirs")]
            output, cache = [Path(command[command.index(flag) + 1]) for flag in ("--output", "--cache")]
            code = "from pathlib import Path; "
            code += f"assert all((Path(p)/'model.pt').is_file() for p in {inputs!r}); "
            code += f"p=Path({str(output)!r}); p.parent.mkdir(parents=True); p.write_bytes(b'plot'); "
            code += f"Path({str(cache)!r}).write_bytes(b'cache')"
            step.command = [sys.executable, "-c", code]
            return step

        def fast_schedule(*args, **kwargs):
            kwargs["poll_interval"] = 0.01
            return shared.schedule(*args, **kwargs)

        run_root, plot_root = self.root / "runs", self.root / "plots"
        args = [*self.argv, "--gpu-ids", *map(str, range(6)),
                "--run-root", str(run_root), "--plot-root", str(plot_root)]
        with patch.object(runner, "__file__", str(self.scripts / "run_dadras_combined_balanced.py")), \
             patch.dict(sys.modules, {"torch": fake_torch}), \
             patch.object(runner, "training_step", side_effect=short_training), \
             patch.object(runner, "comparison_step", side_effect=short_comparison), \
             patch.object(runner, "schedule", side_effect=fast_schedule), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(args), 0)
        for name in runner.variants(self.args):
            for dataset in runner.DATASETS:
                self.assertTrue((run_root / name / dataset / "model.pt").is_file())
            self.assertTrue((plot_root / name / "bifurcation_compare.png").is_file())
            self.assertTrue((plot_root / name / "bifurcation_cache.npz").is_file())
        plan = json.loads((run_root / "plan.json").read_text())
        self.assertEqual(plan["variants"]["cphi"]["latent_dim"], 300)
        self.assertEqual(plan["variants"]["cphi_shifted_relu"]["terms"], ["additive", "shifted-relu"])
        summary = json.loads((run_root / "summary.json").read_text())
        self.assertEqual(len(summary["exit_codes"]), 8)
        self.assertTrue(all(code == 0 for code in summary["exit_codes"].values()))
        self.assertFalse(summary["skipped_comparisons"])
        self.assertEqual(len(list((run_root / "logs").glob("*.log"))), 8)


if __name__ == "__main__":
    unittest.main()
