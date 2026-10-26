import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_cphi_shifted_regimes as combined
import run_term_ablations as shared


PHI = [-1., -.7142857, -.4285714, -.1428571, .1428571, .4285714, .7142857, 1.]


class CphiShiftedRegimesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        for name in ("train_halvorsen_plain.py", "compare_model_bifurcation_diagrams.py"):
            (self.scripts / name).touch()
        self.backend = self.root / "backend.py"
        self.backend.touch()
        self.data_root = self.root / "data"
        for index, regime in enumerate(combined.REGIMES):
            data = self.data_root / regime
            if index == 1:
                data /= "values_8"
            data.mkdir(parents=True)
            (data / "metadata.json").write_text(json.dumps({"args": {"phi_values": PHI}}))
            np.save(data / "trajectory_phi_constants.npy", np.repeat(PHI, 2).astype(np.float32))
            for name in ("trajectories.npy", "trajectory_phi.npy"):
                (data / name).touch()
        self.argv = [
            "--training-data-root", str(self.data_root), "--backend-script", str(self.backend),
            "--gpu-ids", "0", "1",
        ]
        self.args = combined.parse_args(self.argv)

    def test_defaults_and_existing_parameter_splits(self):
        self.assertEqual((self.args.latent_dim, self.args.pwl_units, self.args.epochs), (200, 20, 750))
        experiments = combined.prepare_experiments(self.args, self.scripts)
        dadras, rossler = (experiments[name][1] for name in combined.REGIMES)
        self.assertEqual(dadras, {"all": PHI[:-1], "negative": PHI[:4], "positive": PHI[4:7]})
        self.assertEqual(rossler, {"all": PHI, "negative": PHI[:3], "positive": PHI[3:]})
        self.assertEqual(experiments[combined.REGIMES[1]][0].data_dir.name, "values_8")

    def test_six_gpu_ids_are_accepted_but_seven_and_duplicates_are_rejected(self):
        prefix = self.argv[:-2]
        args = combined.parse_args([*prefix, "0", "1", "2", "3", "4", "5"])
        self.assertEqual(args.gpu_ids, list(range(6)))
        with self.assertRaises(ValueError):
            combined.parse_args([*prefix, "0", "1", "2", "3", "4", "5", "6"])
        with self.assertRaises(ValueError):
            combined.parse_args([*prefix, "0", "1", "2", "3", "4", "4"])
        with self.assertRaises(ValueError):
            shared.validate_gpu_ids(list(range(6)), device_count=5, max_gpus=6)

    def test_six_gpu_queue_launches_all_six_jobs_before_polling(self):
        gpu_file = self.root / "gpu_ids.json"
        gpu_file.write_text(json.dumps(list(range(6))))
        busy = set()
        peak = 0
        launches = []

        def fake_popen(command, **kwargs):
            nonlocal peak
            gpu = int(command[-1])
            self.assertNotIn(gpu, busy)
            busy.add(gpu)
            launches.append(gpu)
            peak = max(peak, len(busy))

            def finished():
                busy.discard(gpu)
                return 0

            return SimpleNamespace(poll=finished)

        output = io.StringIO()
        jobs = [f"job_{index}" for index in range(6)]
        with patch.object(shared.subprocess, "Popen", side_effect=fake_popen), patch.object(shared.time, "sleep"):
            with contextlib.redirect_stdout(output):
                results = shared.schedule(
                    gpu_file, list(range(6)), 6, self.root,
                    lambda job, gpu: [shared.Step(job, ["fake", str(gpu)], self.root)],
                    {}, job_names=jobs, max_gpus=combined.MAX_GPUS,
                )
        self.assertEqual(results, dict.fromkeys(jobs, 0))
        self.assertEqual(launches, list(range(6)))
        self.assertEqual(peak, 6)
        self.assertNotIn("Ignoring invalid GPU file", output.getvalue())

    def test_preflight_requires_both_systems(self):
        (self.data_root / combined.REGIMES[1] / "values_8/trajectories.npy").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "trajectories.npy"):
            combined.prepare_experiments(self.args, self.scripts)

    def test_terms_output_isolation_and_bifurcation_ranges(self):
        experiments = combined.prepare_experiments(self.args, self.scripts)
        outputs = []
        for regime, (config, groups) in experiments.items():
            batch_id = "test_run_m200_p20"
            model_root = self.scripts / "runs/term_ablations" / regime / "cphi_shifted_relu" / batch_id
            plot_root = self.scripts / "bifurcation_diagrams/term_ablations" / regime / "cphi_shifted_relu" / batch_id
            steps = shared.build_steps(
                config, self.scripts, groups, model_root, plot_root,
                combined.VARIANT, 3, terms=combined.TERMS, variant_subdir=False,
            )
            for step in steps[:3]:
                command = step.command
                start = command.index("--phi-terms") + 1
                self.assertEqual(command[start:command.index("--gpu-id")], ["additive", "shifted-relu"])
                self.assertNotIn("--use-phi", command)
                outputs.append(command[-1])
                self.assertEqual(Path(command[-1]).parent, model_root)
            command = steps[-1].command
            lower, upper = shared.PARAMETER_RANGES[regime]
            self.assertEqual(command[command.index("--parameter-min") + 1], str(lower))
            self.assertEqual(command[command.index("--parameter-max") + 1], str(upper))
            self.assertEqual(command[command.index("--model-runs") + 1:command.index("--model-labels")], outputs[-3:])
            self.assertEqual(Path(command[command.index("--output") + 1]).parent, plot_root)
            self.assertEqual(Path(command[command.index("--cache") + 1]).parent, plot_root)
        self.assertEqual(len(set(outputs)), 6)
        self.assertNotIn(combined.VARIANT, shared.VARIANTS)

    def test_six_jobs_interleave_the_two_systems(self):
        trainings, comparisons = combined.job_definitions()
        self.assertEqual(list(trainings), ["dadras_all", "rossler_all", "dadras_negative", "rossler_negative", "dadras_positive", "rossler_positive"])
        self.assertEqual(len(comparisons), 2)

    def test_failed_training_skips_only_its_system_comparison(self):
        trainings, comparisons = combined.job_definitions()
        results = dict.fromkeys(trainings, 0)
        results["dadras_positive"] = 7
        with patch.object(combined, "schedule", side_effect=[results, {"rossler_bifurcation": 0}]) as scheduler:
            with contextlib.redirect_stdout(io.StringIO()):
                final, skipped = combined.run_job_phases(
                    self.root / "gpu_ids.json", [0, 1], 2, self.root, None, {}, trainings, comparisons,
                )
        self.assertEqual(scheduler.call_args_list[0].kwargs["job_names"], list(trainings))
        self.assertEqual(scheduler.call_args_list[1].kwargs["job_names"], ["rossler_bifurcation"])
        self.assertTrue(all(call.kwargs["max_gpus"] == 6 for call in scheduler.call_args_list))
        self.assertEqual(skipped, ["dadras_bifurcation"])
        self.assertNotEqual(final["dadras_bifurcation"], 0)
        self.assertEqual(final["rossler_bifurcation"], 0)

    def test_all_failed_trainings_launch_no_comparisons(self):
        trainings, comparisons = combined.job_definitions()
        with patch.object(combined, "schedule", return_value=dict.fromkeys(trainings, 1)) as scheduler:
            with contextlib.redirect_stdout(io.StringIO()):
                final, skipped = combined.run_job_phases(
                    self.root / "gpu_ids.json", [0], 2, self.root, None, {}, trainings, comparisons,
                )
        self.assertEqual(scheduler.call_count, 1)
        self.assertEqual(skipped, list(comparisons))
        self.assertEqual(len(final), 8)

    def test_dry_run_has_exactly_six_trainings_and_two_comparisons(self):
        output = io.StringIO()
        with patch.object(combined, "__file__", str(self.scripts / "run_cphi_shifted_regimes.py")):
            with contextlib.redirect_stdout(output):
                self.assertEqual(combined.main([*self.argv, "--dry-run"]), 0)
        self.assertEqual(output.getvalue().count("--phi-terms additive shifted-relu"), 6)
        self.assertEqual(output.getvalue().count("--model-runs"), 2)
        self.assertFalse((self.scripts / "runs").exists())

    def test_full_queue_writes_plan_logs_and_summary_without_real_training(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 6, get_device_name=lambda gpu: f"mock GPU {gpu}",
        ))
        original_builder = combined.build_steps

        def short_steps(*args, **kwargs):
            steps = original_builder(*args, **kwargs)
            for step in steps:
                original = step.command
                if "--output-dir" in original:
                    directory = original[original.index("--output-dir") + 1]
                    targets = [Path(directory) / "model.pt"]
                else:
                    targets = [Path(original[original.index(flag) + 1]) for flag in ("--output", "--cache")]
                code = "from pathlib import Path; "
                for target in targets:
                    code += f"p = Path({str(target)!r}); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'mock'); "
                code += f"print({step.label!r}, flush=True)"
                step.command = [sys.executable, "-u", "-c", code]
            return steps

        def fast_schedule(*args, **kwargs):
            kwargs["poll_interval"] = 0.01
            return shared.schedule(*args, **kwargs)

        with patch.object(combined, "__file__", str(self.scripts / "run_cphi_shifted_regimes.py")):
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(combined, "build_steps", side_effect=short_steps):
                with patch.object(combined, "schedule", side_effect=fast_schedule), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(combined.main([*self.argv[:-2], "0", "1", "2", "3", "4", "5"]), 0)
        roots = list((self.scripts / "runs/term_ablations/_queues/cphi_shifted_relu").iterdir())
        self.assertEqual(len(roots), 1)
        root = roots[0]
        self.assertTrue(root.name.endswith("_m200_p20"))
        plan = json.loads((root / "plan.json").read_text())
        self.assertEqual(plan["phi_terms"], ["additive", "shifted-relu"])
        self.assertEqual(json.loads((root / "gpu_ids.json").read_text()), list(range(6)))
        for regime in combined.REGIMES:
            model_root = self.scripts / "runs/term_ablations" / regime / "cphi_shifted_relu" / root.name
            plot_root = self.scripts / "bifurcation_diagrams/term_ablations" / regime / "cphi_shifted_relu" / root.name
            self.assertEqual(plan["run_roots"][regime], str(model_root))
            self.assertEqual(plan["plot_roots"][regime], str(plot_root))
            for subset in combined.SUBSETS:
                self.assertTrue((model_root / subset / "model.pt").is_file())
            self.assertTrue((plot_root / "bifurcation_compare.png").is_file())
            self.assertTrue((plot_root / "bifurcation_cache.npz").is_file())
        self.assertFalse((self.scripts / "runs/cphi_shifted_relu").exists())
        summary = json.loads((root / "summary.json").read_text())
        self.assertEqual(len(summary["exit_codes"]), 8)
        self.assertTrue(all(code == 0 for code in summary["exit_codes"].values()))
        self.assertFalse(summary["skipped_comparisons"])
        logs = list((root / "logs").glob("*.log"))
        self.assertEqual(len(logs), 8)
        for log in logs:
            self.assertEqual(log.read_text().count("### "), 1)


if __name__ == "__main__":
    unittest.main()
