"""Server-only simulator layout, explicit paths and local synchronization checks."""
from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from halvorsen_ab import common, generate_data, sync_simulator


class SimulatorLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = sync_simulator.simulator_paths(common.DEFAULT_ROOT)[0].read_bytes()

    def write(self, relative, content=None):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.source if content is None else content)
        return path

    def test_generation_with_only_diffrentbutconst_prams_and_explicit_file(self):
        path = self.write("diffrentbutconst_prams/" + sync_simulator.SIMULATOR_NAME)
        expected_hash = common.file_hash(path)
        for location in (path.parent, path, self.root):
            module, digest = common.load_simulator(location)
            self.assertEqual(digest, expected_hash)
            self.assertEqual(Path(module.__file__), path)
        config = replace(common.load_config(), grid_a=2, grid_b=2, trajectories_per_pair=1,
                         n_steps=40, burn_in_steps=2)
        folder = self.root / "data"
        metadata = generate_data.generate(config, folder, path.parent, progress=False)
        self.assertEqual(metadata["shapes"]["trajectory_phi"], [160, 2])
        self.assertTrue(np.isfinite(np.load(folder / "trajectories.npy")).all())
        self.assertFalse((self.root / "time_varying_bifurcation_data").exists())
        self.assertFalse((self.root / "mixed_parameter_trajectories").exists())
        self.assertFalse((self.root / "generate_trainingdata").exists())

    def test_default_discovery_finds_sibling_server_folder(self):
        path = self.write("diffrentbutconst_prams/" + sync_simulator.SIMULATOR_NAME)
        fake_script = self.root / "ALRNN-DSR-extended/alrnn_python/halvorsen_ab/sync_simulator.py"
        with patch.dict(os.environ), patch.object(sync_simulator, "__file__", str(fake_script)):
            os.environ.pop("GENERATE_TRAININGDATA_ROOT", None)
            self.assertEqual(sync_simulator.default_simulator_root(), path.parent)

    def test_explicit_environment_location_is_not_overridden(self):
        path = self.write("custom/" + sync_simulator.SIMULATOR_NAME)
        with patch.dict(os.environ, {"GENERATE_TRAININGDATA_ROOT": str(path)}):
            self.assertEqual(sync_simulator.default_simulator_root(), path)

    def test_two_local_copies_still_require_bytewise_agreement(self):
        first = self.write(sync_simulator.RELATIVE_FILES[0])
        second = self.write(sync_simulator.RELATIVE_FILES[1], self.source + b"\n# different copy\n")
        with self.assertRaisesRegex(ValueError, "not synchronized"):
            common.load_simulator(self.root)
        with self.assertRaisesRegex(ValueError, "copies differ"):
            sync_simulator.synchronize(self.root, apply=True)
        self.assertEqual(first.read_bytes(), self.source)
        self.assertEqual(second.read_bytes(), self.source + b"\n# different copy\n")

    def test_single_legacy_server_copy_can_be_updated(self):
        source = ("import numpy as np\nArray = np.ndarray\n\n\n" + sync_simulator.OLD_SIGNATURE + "\n"
                  "    a = p\n"
                  "    return np.array([\n"
                  "        -a*x[0] - 4.0 * x[1] - 4.0 * x[2] - x[1]*x[1] + drive,\n"
                  "        -a*x[1] - 4.0 * x[2] - 4.0 * x[0] - x[2]*x[2] + drive,\n"
                  "        -a*x[2] - 4.0 * x[0] - 4.0 * x[1] - x[0]*x[0] + drive,\n"
                  "    ], dtype=np.float64)\n\n\nRHS: dict = {'halvorsen': halvorsen_rhs}\n").encode()
        path = self.write("diffrentbutconst_prams/" + sync_simulator.SIMULATOR_NAME, source)
        with self.assertRaisesRegex(ValueError, "lacks b/c"):
            common.load_simulator(path.parent)
        with self.assertRaisesRegex(ValueError, "extension is missing"):
            sync_simulator.synchronize(path.parent)
        self.assertEqual(path.read_bytes(), source)
        digest = sync_simulator.synchronize(path.parent, apply=True)
        module, loaded_hash = common.load_simulator(path.parent)
        self.assertEqual(digest, loaded_hash)
        np.testing.assert_allclose(module.halvorsen_rhs(0, np.array([1.,2.,3.]), 1.5, b=2., c=5.),
                                   [-24.5, -23., -17.5])
        self.assertEqual(sync_simulator.synchronize(path), digest)
        self.assertFalse((self.root / "time_varying_bifurcation_data").exists())

    def test_explicit_missing_path_does_not_fall_back_to_another_simulator(self):
        self.write("diffrentbutconst_prams/" + sync_simulator.SIMULATOR_NAME)
        with self.assertRaisesRegex(FileNotFoundError, "--simulator-root"):
            common.load_simulator(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
