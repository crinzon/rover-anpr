import base64
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from live_preview import PreviewPublisher, PreviewReader, SystemMetrics


class LiveReadingsTests(unittest.TestCase):
    def test_counts_crop_and_retention_are_per_processed_box(self):
        with tempfile.TemporaryDirectory() as directory:
            publisher = PreviewPublisher({}, path=Path(directory) / "preview")
            try:
                frame = np.zeros((20, 30, 3), dtype=np.uint8)
                frame[3:7, 2:9] = (20, 80, 200)
                raw = [
                    dict(
                        box=[2, 3, 9, 7],
                        text="K4#7",
                        readable=True,
                        confidence=0.8,
                    )
                ]
                live = [
                    dict(
                        box=[2, 3, 9, 7],
                        text="K4#7",
                        status="PARTIAL",
                        track_id=1,
                    )
                ]
                publisher.record_readings(
                    frame, raw, live, time.time(), "front-session"
                )
                crop = publisher.readings["last_reading"]["crop"]
                image = cv2.imdecode(
                    np.frombuffer(
                        base64.b64decode(crop.split(",")[1]), np.uint8
                    ),
                    cv2.IMREAD_COLOR,
                )
                self.assertEqual(image.shape[:2], (4, 7))
                self.assertTrue(
                    np.allclose(image.mean(axis=(0, 1)), (20, 80, 200), atol=3)
                )
                publisher.record_readings(
                    frame, [], [], time.time(), "front-session"
                )
                self.assertEqual(
                    publisher.readings["last_reading"]["text"], "K4#7"
                )
                publisher.record_readings(
                    frame, raw, live, time.time(), "front-session"
                )
                unreadable = [dict(box=[2, 3, 9, 7], text="", readable=False)]
                publisher.record_readings(
                    frame, unreadable, [], time.time(), "front-session"
                )
                self.assertEqual(publisher.readings["detections"], 3)
                self.assertEqual(publisher.readings["ocr_readings"], 2)
                self.assertEqual(publisher.readings["partial_readings"], 2)
                self.assertIsNone(publisher.readings["accuracy_percent"])
            finally:
                publisher.close()
            self.assertEqual(
                json.loads(publisher.readings_path.read_text())["detections"],
                3,
            )

    @unittest.skipUnless(
        importlib.util.find_spec("fastapi"), "FastAPI unavailable"
    )
    def test_api_keeps_camera_crop_and_counters_after_signal_loss(self):
        import api

        with tempfile.TemporaryDirectory() as directory:
            paths = {
                name: PreviewReader(Path(directory) / name)
                for name in ("Front", "Back")
            }
            for name in paths:
                Path(str(paths[name].path) + ".readings.json").write_text(
                    json.dumps(
                        dict(detections=1, last_reading=dict(text=name))
                    ),
                    encoding="utf-8",
                )
            with patch.object(api, "CAMERA_READERS", paths):
                for name in paths:
                    result = api.live_status(name)
                    self.assertFalse(result["available"])
                    self.assertEqual(
                        result["readings"]["last_reading"]["text"], name
                    )

    def test_host_metrics_first_cpu_sample_is_unknown(self):
        metrics = SystemMetrics()
        first = metrics._sample_host()
        self.assertIsNone(first["cpu_percent"])
        self.assertGreater(first["ram_total_bytes"], 0)
        second = metrics._sample_host()
        if second["cpu_percent"] is not None:
            self.assertTrue(0 <= second["cpu_percent"] <= 100)

    def test_missing_gpu_and_unavailable_host_return_unknown(self):
        metrics = SystemMetrics()
        with (
            patch.object(metrics, "_sample_host", side_effect=OSError),
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("time.sleep", side_effect=StopIteration),
        ):
            with self.assertRaises(StopIteration):
                metrics._run()
        self.assertIsNone(metrics.value["gpu_percent"])
        self.assertIsNone(metrics.value["cpu_percent"])

    def test_demo_prefers_confirmed_crop_without_double_count(self):
        with tempfile.TemporaryDirectory() as directory:
            publisher = PreviewPublisher({}, path=Path(directory) / "preview")
            try:
                publisher.prefer_confirmed = True
                frame = np.zeros((20, 30, 3), np.uint8)
                raw = [
                    dict(box=[1, 1, 5, 5], text="A111AA77", readable=True),
                    dict(box=[10, 1, 15, 5], text="B222BB77", readable=True),
                ]
                live = [
                    dict(
                        box=raw[0]["box"], text="A111AA77", status="CONFIRMED"
                    ),
                    dict(box=raw[1]["box"], text="B222BB77", status="PENDING"),
                ]
                publisher.record_readings(
                    frame, raw, live, time.time(), "demo"
                )
                self.assertEqual(publisher.readings["detections"], 2)
                self.assertEqual(
                    publisher.readings["last_reading"]["text"], "A111AA77"
                )
                self.assertEqual(
                    publisher.readings["last_reading"]["box"], [1, 1, 5, 5]
                )
            finally:
                publisher.close()


if __name__ == "__main__":
    unittest.main()
