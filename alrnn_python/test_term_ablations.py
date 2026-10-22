import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_term_ablations as runner


PHI = [-1., -0.7142857, -0.4285714, -0.1428571, 0.1428571, 0.4285714, 0.7142857, 1.]


class TermAblationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "metadata.json").write_text(json.dumps({"args": {"phi_values": PHI}}))
        np.save(self.root / "trajectory_phi_constants.npy", np.repeat(PHI, 2).astype(np.float32))

    def test_dadras_groups_exclude_phi1_in_every_training(self):
        groups = runner.load_groups(self.root, "dadras_e_49_to_64")
        self.assertEqual([len(v) for v in groups.values()], [7, 4, 3])
        self.assertTrue(all(1. not in values for values in groups.values()))
        self.assertEqual(groups["negative"], PHI[:4])
        self.assertEqual(groups["positive"], PHI[4:7])

    def test_rossler_moves_phi_without_changing_its_value(self):
        groups = runner.load_groups(self.root, "rossler_a_005_to_035")
        self.assertEqual(groups["all"], PHI)
        self.assertEqual(groups["negative"], PHI[:3])
        self.assertEqual(groups["positive"], PHI[3:])

    def test_metadata_fallback_and_mismatch(self):
        (self.root / "metadata.json").write_text(json.dumps({"requested_phi_values": PHI}))
        self.assertEqual(len(runner.load_groups(self.root, "dadras_e_49_to_64")["all"]), 7)
        np.save(self.root / "trajectory_phi_constants.npy", [0.])
        with self.assertRaisesRegex(ValueError, "do not match"):
            runner.load_groups(self.root, "dadras_e_49_to_64")

    def test_gpu_validation(self):
        for values in ([], [0, 0], [-1], [True], ["0"], [0, 1, 2, 3, 4], {"gpu": 0}, [None]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                runner.validate_gpu_ids(values)
        self.assertEqual(runner.validate_gpu_ids([2, 3], 4), [2, 3])
        with self.assertRaises(ValueError):
            runner.validate_gpu_ids([4], 4)

    def test_default_training_settings(self):
        args = runner.parse_args([
            "--regime", "dadras_e_49_to_64",
            "--training-data-root", str(self.root),
            "--backend-script", str(self.root / "backend.py"),
        ])
        self.assertEqual((args.latent_dim, args.pwl_units, args.epochs), (200, 20, 750))
        for name in ("run_dadras_cphi_without_phi1.sh", "run_rossler_cphi_shifted_split.sh"):
            source = Path(__file__).with_name(name).read_text()
            self.assertIn(f"latents_dim={args.latent_dim}\n", source)
            self.assertIn(f"pwl_units={args.pwl_units}\n", source)
            self.assertIn(f"epochs={args.epochs}\n", source)

    def test_commands_select_exact_terms_and_three_models(self):
        for regime, limits in runner.PARAMETER_RANGES.items():
            groups = runner.load_groups(self.root, regime)
            args = SimpleNamespace(
                regime=regime, data_dir=self.root, backend_script=self.root / "backend.py",
                cpu_threads=4, latent_dim=200, pwl_units=20, batch_size=80,
                epochs=750, batches_per_epoch=100,
            )
            outputs = []
            for variant, terms in runner.VARIANTS.items():
                steps = runner.build_steps(args, self.root, groups, self.root / "runs", self.root / "plots", variant, 3)
                self.assertEqual(len(steps), 4)
                for (subset, values), step in zip(groups.items(), steps[:3]):
                    command = step.command
                    start = command.index("--phi-terms") + 1
                    self.assertEqual(command[start:command.index("--gpu-id")], list(terms))
                    self.assertNotIn("--use-phi", command)
                    start = command.index("--train-phi-values") + 1
                    self.assertEqual([float(v) for v in command[start:command.index("--output-dir")]], values)
                    outputs.append(command[-1])
                    self.assertEqual(Path(command[-1]).name, subset)
                comparison = steps[-1].command
                self.assertEqual(comparison[2], str(self.root / "compare_model_bifurcation_diagrams.py"))
                self.assertEqual(comparison[comparison.index("--model-runs") + 1:comparison.index("--model-labels")], outputs[-3:])
                self.assertEqual(float(comparison[comparison.index("--parameter-min") + 1]), limits[0])
                self.assertEqual(float(comparison[comparison.index("--parameter-max") + 1]), limits[1])
            self.assertEqual(len(set(outputs)), 12)

    def fake_steps(self, variant, gpu, duration=0.12, failure=False):
        steps = []
        for index in range(4):
            start = {"event": "start", "variant": variant, "gpu": gpu, "step": index}
            end = {**start, "event": "end"}
            code = (
                "import json,time,sys; "
                f"print(json.dumps(dict({start!r}, timestamp=time.time())), flush=True); "
                f"time.sleep({duration}); "
                f"print(json.dumps(dict({end!r}, timestamp=time.time())), flush=True); "
                f"sys.exit({7 if failure and index == 1 else 0})"
            )
            steps.append(runner.Step(f"stage {index}", [sys.executable, "-u", "-c", code], self.root))
        return steps

    def events(self):
        events = []
        for log in self.root.glob("*.log"):
            events.extend(json.loads(line) for line in log.read_text().splitlines() if line.startswith("{"))
        return sorted(events, key=lambda event: event["timestamp"])

    def assert_parallelism(self, expected):
        busy = set()
        maximum = 0
        for event in self.events():
            gpu = event["gpu"]
            if event["event"] == "start":
                self.assertNotIn(gpu, busy, "Two stages overlapped on the same GPU")
                busy.add(gpu)
                maximum = max(maximum, len(busy))
            else:
                busy.remove(gpu)
        self.assertFalse(busy)
        self.assertEqual(maximum, expected)

    def run_schedule(self, gpu_ids, make_steps=None):
        gpu_file = self.root / "gpu_ids.json"
        gpu_file.write_text(json.dumps(gpu_ids))
        return runner.schedule(
            gpu_file, gpu_ids, 4, self.root, make_steps or self.fake_steps,
            os.environ.copy(), poll_interval=0.01,
        )

    def test_one_two_and_four_gpu_slots(self):
        for count in (1, 2, 4):
            with self.subTest(count=count), contextlib.redirect_stdout(io.StringIO()):
                results = self.run_schedule(list(range(count)))
                self.assertEqual(results, dict.fromkeys(runner.VARIANTS, 0))
                self.assert_parallelism(count)
                for variant in runner.VARIANTS:
                    stages = [e["step"] for e in self.events() if e["variant"] == variant and e["event"] == "start"]
                    self.assertEqual(stages, [0, 1, 2, 3])

    def test_hot_added_gpu_and_partial_json_write(self):
        added = threading.Event()
        errors = []

        def add_gpu():
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    log = self.root / "diagonal.log"
                    if log.exists() and '"event": "start"' in log.read_text():
                        break
                    time.sleep(0.01)
                else:
                    raise AssertionError("Initial pipeline did not start")
                gpu_file = self.root / "gpu_ids.json"
                gpu_file.write_text("[")
                time.sleep(0.08)
                gpu_file.write_text("[0, 1]")
                added.set()
            except Exception as exc:
                errors.append(exc)

        editor = threading.Thread(target=add_gpu)
        editor.start()
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                results = self.run_schedule([0], lambda v, g: self.fake_steps(v, g, duration=0.2))
        finally:
            editor.join(timeout=6)
        self.assertFalse(editor.is_alive())
        self.assertFalse(errors)
        self.assertTrue(added.is_set())
        self.assertEqual(results, dict.fromkeys(runner.VARIANTS, 0))
        self.assertIn("Ignoring invalid GPU file", output.getvalue())
        self.assertIn("GPU queue updated: [0, 1]", output.getvalue())
        self.assert_parallelism(2)
        starts = [e for e in self.events() if e["event"] == "start" and e["step"] == 0]
        self.assertEqual(starts[1]["gpu"], 1)

    def test_failed_pipeline_does_not_stop_other_variants(self):
        with contextlib.redirect_stdout(io.StringIO()):
            results = self.run_schedule([0, 1], lambda v, g: self.fake_steps(v, g, failure=(v == "diagonal")))
        self.assertEqual(results["diagonal"], 7)
        self.assertTrue(all(code == 0 for variant, code in results.items() if variant != "diagonal"))
        failed_steps = [e["step"] for e in self.events() if e["variant"] == "diagonal" and e["event"] == "start"]
        self.assertEqual(failed_steps, [0, 1])

    def test_launch_failure_does_not_stop_other_variants(self):
        def make_steps(variant, gpu):
            if variant == "diagonal":
                return [runner.Step("missing executable", [str(self.root / "missing_program")], self.root)]
            return self.fake_steps(variant, gpu, duration=0.01)

        with contextlib.redirect_stdout(io.StringIO()):
            results = self.run_schedule([0], make_steps)
        self.assertEqual(results["diagonal"], 1)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(code == 0 for variant, code in results.items() if variant != "diagonal"))

    def test_interrupt_reaps_running_process(self):
        children = []
        real_popen = runner.subprocess.Popen

        def capture(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            children.append(process)
            return process

        with contextlib.redirect_stdout(io.StringIO()), patch.object(runner.subprocess, "Popen", side_effect=capture):
            with patch.object(runner.time, "sleep", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_schedule([0], lambda v, g: self.fake_steps(v, g, duration=10))
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())


if __name__ == "__main__":
    unittest.main()
