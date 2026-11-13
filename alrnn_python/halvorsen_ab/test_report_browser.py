"""Optional browser check: actual Plotly rendering, both slices and CSV export."""
from dataclasses import asdict, replace
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from halvorsen_ab.common import DataConfig
from halvorsen_ab.report import write_report


def browser_executable():
    candidates = [os.environ.get("HALVORSEN_BROWSER"), shutil.which("google-chrome"), shutil.which("chromium"),
                  "C:/Program Files/Google/Chrome/Application/chrome.exe",
                  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"]
    return next((str(path) for path in candidates if path and Path(path).is_file()), None)


@unittest.skipUnless(importlib.util.find_spec("playwright") and browser_executable(),
                     "Optional browser test needs playwright and Chrome/Chromium/Edge (HALVORSEN_BROWSER).")
class PlotlyBrowserTests(unittest.TestCase):
    def test_offline_cloud_slices_numeric_input_orbits_and_csv(self):
        from playwright.sync_api import sync_playwright
        config = replace(DataConfig(), grid_a=2, grid_b=2)
        pairs = config.pairs()
        truth_peaks = 10*pairs[:, 0]+pairs[:, 1]
        orbit = np.tile(np.arange(6).reshape(1, 6, 1), (4, 1, 3)).astype(float)
        arrays = dict(pairs=pairs, truth_peaks=truth_peaks, truth_offsets=np.arange(5),
                      truth_equilibria=np.full(4, np.nan), model_peaks=truth_peaks+.1,
                      model_offsets=np.arange(5), model_equilibria=np.full(4,np.nan),
                      model_failed_at=np.full(4,-1), orbit_pairs=pairs, orbit_truth=orbit, orbit_model=orbit+.2)
        metadata = dict(label="Browser fixture", ground_truth_only=False, data_config=asdict(config),
                        ground_truth=dict(failures=[], settings=dict(warmup_steps=1, observation_steps=5, initial_states=1)),
                        model_sha256="fixture", model_failures=0,
                        metrics=[dict(status="completed", dstsp=.1, dh=.2, rmse=.3)]*4)
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "report.html"
            write_report(arrays, metadata, report)
            errors, remote_requests = [], []
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(executable_path=browser_executable(), headless=True,
                    args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
                page = browser.new_page(viewport=dict(width=1400, height=1000), accept_downloads=True)
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda request: remote_requests.append(request.url) if request.url.startswith("http") else None)
                page.goto(report.as_uri())
                page.wait_for_function("window.halvorsenReady === true", timeout=60000)
                self.assertEqual(page.evaluate("document.getElementById('cloud').data[0].z.length"), 4)
                self.assertEqual(page.evaluate("document.getElementById('slice').data[0].x"), [2.,4.])
                page.select_option("#fixed-axis", "b")
                page.wait_for_function("document.getElementById('slice-label').textContent.startsWith('b =')")
                page.wait_for_function("document.getElementById('slice').data[0].x[0] === 1.4")
                self.assertEqual(page.evaluate("document.getElementById('slice').data[0].x"), [1.4,2.1])
                self.assertEqual(page.evaluate("document.getElementById('slice').layout.xaxis.range"), [1.4,2.1])
                page.locator("#slice-index").evaluate("el => {el.value=0;el.dispatchEvent(new Event('input'));}")
                page.wait_for_function("document.getElementById('slice-label').textContent.startsWith('b = 2;')")
                page.wait_for_function("document.getElementById('slice').data[0].y[0] === 16")
                self.assertEqual(page.evaluate("document.getElementById('slice').data[0].y"), [16.,23.])
                page.select_option("#fixed-axis", "a")
                page.fill("#fixed-value", "1.5")
                page.locator("#fixed-value").dispatch_event("change")
                page.wait_for_function("document.getElementById('slice-label').textContent.startsWith('a = 1.4;')")
                self.assertEqual(page.input_value("#fixed-value"), "1.4")
                with page.expect_download() as download:
                    page.click("#download-slice")
                csv_path = Path(temp) / "slice.csv"
                download.value.save_as(csv_path)
                rows = csv_path.read_text().splitlines()
                self.assertEqual(rows[0], "a,b,system,kind,z")
                self.assertEqual(len(rows), 5)
                self.assertTrue(all(row.startswith("1.4,") for row in rows[1:]))
                page.select_option("#orbit-pair", "0")
                page.wait_for_function("document.getElementById('orbit').layout.title.text === 'a=1.4, b=2'")
                self.assertEqual(page.evaluate("document.getElementById('orbit').data.length"), 2)
                self.assertEqual(errors, [])
                self.assertEqual(remote_requests, [])
                browser.close()


if __name__ == "__main__":
    unittest.main()
