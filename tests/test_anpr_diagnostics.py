import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from anpr_diagnostics import (
    Diagnostics,
    describe_plate,
    read_diagnostics,
    set_confirmation,
)


def plate(**changes):
    data = dict(
        box=[100, 200, 400, 260],
        text="A123AA77",
        confidence=0.95,
        detection=0.8,
        readable=True,
        valid=True,
        subtype="type1",
        patch_png="data:image/png;base64,eA==",
        raw_id=4,
    )
    data.update(changes)
    return data


class DiagnosticsTest(unittest.TestCase):
    def test_reports_all_gate_failures_without_changing_thresholds(self):
        item = describe_plate(
            plate(readable=False, confidence=0.3, text="BAD"), 0.85, 3840, 640
        )
        self.assertEqual(
            item["reasons"],
            ["unreadable_subtype", "ocr_confidence", "rover_format"],
        )
        self.assertEqual(item["status"], "rejected")
        good = describe_plate(plate(), 0.85, 3840, 640)
        self.assertEqual(good["reasons"], [])
        self.assertEqual(good["detector_size"], [50, 10])
        self.assertEqual(good["status"], "waiting_confirmation")

    def test_ocr_threshold_boundary_and_model_format_are_distinct(self):
        good = describe_plate(
            plate(confidence=0.85, valid=False), 0.85, 3840, 640
        )
        self.assertNotIn("ocr_confidence", good["reasons"])
        self.assertFalse(good["model_format_valid"])
        self.assertTrue(good["rover_format_valid"])
        low = describe_plate(plate(confidence=0.849), 0.85, 3840, 640)
        self.assertIn("ocr_confidence", low["reasons"])

    def test_confirmation_states(self):
        item = describe_plate(plate(), 0.85, 3840, 640)
        set_confirmation(item, 2, 2, 3, None)
        self.assertEqual(item["reasons"], ["insufficient_confirmations"])
        set_confirmation(item, 2, 3, 3, "A123AA77")
        self.assertEqual(item["status"], "confirmed")
        set_confirmation(item, 2, 3, 3, "B123BB77", history=["A123AA77"])
        self.assertEqual(item["status"], "waiting_number_change")
        self.assertEqual(item["confirmations"], 1)
        self.assertEqual(item["required"], 7)

    def test_disabled_sampling_expiration_and_safe_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample"
            diagnostics = Diagnostics({}, {}, path)
            self.assertFalse(diagnostics.due())
            self.assertFalse(path.exists())
            diagnostics = Diagnostics(
                {"ANPR_DIAGNOSTICS": "true"}, {"ocr": 0.85}, path
            )
            self.assertTrue(diagnostics.due())
            self.assertFalse(diagnostics.due())
            output = dict(
                source=dict(width=3840, height=2160),
                timing_ms={},
                diagnostics=dict(
                    raw_proposals=[
                        dict(
                            id=0, confidence=0.1, reason="detector_confidence"
                        )
                    ]
                ),
            )
            diagnostics.finish(
                output,
                [describe_plate(plate(), 0.85, 3840, 640)],
                42,
                time.monotonic(),
                b"jpeg",
            )
            sample = read_diagnostics(path)
            self.assertEqual(sample["state"], "ok")
            self.assertEqual(sample["frame_id"], 42)
            self.assertEqual(sample["plates"][0]["raw_id"], 4)
            content = json.loads(path.read_text())
            content["VIDEO_SOURCE"] = "secret"
            path.write_text(json.dumps(content))
            self.assertNotIn("secret", json.dumps(read_diagnostics(path)))
            with patch(
                "anpr_diagnostics.time.time", return_value=time.time() + 20
            ):
                self.assertEqual(read_diagnostics(path)["state"], "stale")
                self.assertNotIn("plates", read_diagnostics(path))
            diagnostics.error("worker_failed")
            self.assertEqual(
                read_diagnostics(path)["error_code"], "worker_failed"
            )
            Diagnostics({}, {}, path)
            self.assertFalse(path.exists())


class NativeRegressionTest(unittest.TestCase):
    @unittest.skipUnless(
        all(
            importlib.util.find_spec(name)
            for name in ("cv2", "torch", "ultralytics")
        ),
        "Camera runtime dependencies unavailable",
    )
    def test_native_camera_regressions_in_isolated_process(self):
        script = Path(__file__).with_name("check_native_anpr.py")
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
