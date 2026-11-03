import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_dadras_combined_balanced as runner


class MixedSystemExperimentsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        for name in ("train_halvorsen_plain.py", "mixed_trajectory_dataset.py",
                     "trajectory_storage.py", "compare_model_bifurcation_diagrams.py", "backend.py"):
            (self.scripts / name).touch()

    def args(self, system, *extra):
        return runner.parse_args([
            "--system", system, "--training-data-root", str(self.root / system),
            "--backend-script", str(self.scripts / "backend.py"),
            "--sequence-length", "16", *extra,
        ])

    def test_regimes_ranges_and_model_sizes(self):
        expected = {
            "rossler": ("rossler_a_005_to_035", (0.05, 0.35), [
                "rossler_a_0p05_to_0p14", "rossler_a_0p05_to_0p35", "rossler_a_0p17_to_0p35"]),
            "tsucs1": ("tsucs1_e_20_to_245", (20.0, 24.5), [
                "tsucs1_e_20_to_24p5", "tsucs1_e_20_to_23", "tsucs1_e_23p6_to_24p5"]),
            "lorenz_rho": ("lorenz_rho_1_to_30", (1.0, 30.0), [
                "lorenz_rho_5_to_23", "lorenz_rho_1_to_30"]),
            "rossler_c": ("rossler_c_3_to_6", (3.0, 6.0), [
                "rossler_c_4p5_to_5p5", "rossler_c_3_to_6"]),
        }
        for system, (regime, sweep, datasets) in expected.items():
            args = self.args(system)
            experiment = runner.EXPERIMENTS[system]
            self.assertEqual(experiment["regime"], regime)
            self.assertEqual(list(experiment["datasets"]), datasets)
            self.assertEqual((args.parameter_min, args.parameter_max), sweep)
            models = runner.variants(args)
            self.assertEqual((models["cphi"].latent_dim, models["cphi"].pwl_units), (300, 150))
            self.assertEqual((models["cphi_shifted_relu"].latent_dim, models["cphi_shifted_relu"].pwl_units), (200, 100))
            self.assertEqual(models["cphi_shifted_relu"].terms, ("additive", "shifted-relu"))

    def test_training_and_comparison_commands_stay_with_the_selected_system(self):
        for system in ("rossler", "tsucs1", "lorenz_rho", "rossler_c"):
            args = self.args(system)
            experiment = runner.EXPERIMENTS[system]
            models = runner.variants(args)
            jobs, comparisons = runner.job_definitions(models, experiment["datasets"])
            self.assertEqual(len(jobs), 2 * len(experiment["datasets"]))
            self.assertEqual(len(comparisons), 2)
            outputs = set()
            for variant, dataset in jobs.values():
                step = runner.training_step(args, self.scripts, self.root / "runs", variant, models[variant], dataset, 4)
                cmd = step.command
                self.assertEqual(Path(cmd[cmd.index("--data-dir") + 1]), args.training_data_root / dataset)
                self.assertIn("--mixed-balanced-sampling", cmd)
                self.assertIn("--no-best-model", cmd)
                outputs.add(cmd[cmd.index("--output-dir") + 1])
            self.assertEqual(len(outputs), len(jobs))
            for variant in comparisons.values():
                cmd = runner.comparison_step(args, self.scripts, self.root / "runs", self.root / "plots",
                                             variant, models[variant], 2, 0.01).command
                self.assertEqual(cmd[cmd.index("--regime") + 1], experiment["regime"])
                self.assertIn("--use-training-normalization", cmd)
                self.assertEqual(cmd[cmd.index("--training-data-dirs") + 1:cmd.index("--model-labels")],
                                 [str(args.training_data_root / name) for name in experiment["datasets"]])

    def test_preflight_reads_each_system_and_rejects_a_different_regime(self):
        for system in ("rossler", "tsucs1", "lorenz_rho", "rossler_c"):
            args = self.args(system)
            experiment = runner.EXPERIMENTS[system]
            for name in experiment["datasets"]:
                folder = args.training_data_root / name
                folder.mkdir(parents=True)
                metadata = {"format": "alrnn_mixed_parameter_trajectories_v2",
                            "config": {"regime": experiment["regime"], "standardization": "global", "dt": 0.01},
                            "phi_reference": {"boundary": 1.0, "scale": 0.5}}
                (folder / "metadata.json").write_text(json.dumps(metadata))
                for filename, values in {
                    "trajectories": np.zeros((114, 3)), "trajectory_phi": np.zeros((114, 1)),
                    "trajectory_lengths": [51, 63], "trajectory_train_lengths": [40, 50],
                    "trajectory_group_names": ["constant", "sinusoidal"],
                    "trajectory_mean": np.zeros((2, 1, 3)), "trajectory_std": np.ones((2, 1, 3)),
                }.items():
                    np.save(folder / f"{filename}.npy", values)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.prepare_data(args, self.scripts), 0.01)
            first = args.training_data_root / next(iter(experiment["datasets"])) / "metadata.json"
            metadata = json.loads(first.read_text())
            metadata["config"]["regime"] = "dadras_e_49_to_64"
            first.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, experiment["regime"]):
                runner.prepare_data(args, self.scripts)

    def test_dry_runs_isolate_model_and_plot_folders_for_all_systems(self):
        for system in runner.EXPERIMENTS:
            args = self.args(system)
            argv = ["--system", system, "--training-data-root", str(args.training_data_root),
                    "--backend-script", str(args.backend_script), "--gpu-ids", "0", "1", "2", "3", "4", "5", "--dry-run"]
            with patch.object(runner, "__file__", str(self.scripts / "runner.py")), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(runner.main(argv), 0)
            text = output.getvalue()
            count = 2 * len(runner.EXPERIMENTS[system]["datasets"])
            self.assertEqual(text.count("--mixed-balanced-sampling"), count)
            self.assertIn(f"DRY RUN: {count} trainings, then 2 bifurcation comparisons", text)
            self.assertEqual(text.count("--model-runs"), 2)
            self.assertIn(f"combined_{system}_balanced", text)
            self.assertIn(runner.EXPERIMENTS[system]["regime"], text)
            self.assertFalse((self.scripts / "runs").exists())


if __name__ == "__main__":
    unittest.main()
