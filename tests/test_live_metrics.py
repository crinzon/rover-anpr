import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from test_process_video import FakeCV, metadata, runner

from live_preview import RunStatistics
from process_video import VideoReplay, process_one


class LiveMetricsTest(unittest.TestCase):
    def test_aggregates_are_measured_and_unknowns_remain_unknown(self):
        stats = RunStatistics()
        stats.started = 10
        stats.add(
            dict(detect=2, recognize=4, worker_cycle=9, invalid=float("nan"))
        )
        stats.add(dict(detect=6, recognize=8, worker_cycle=15))
        with patch("live_preview.time.monotonic", return_value=12):
            result = stats.summary()
        self.assertEqual(result["processed_frames"], 2)
        self.assertEqual(result["average_anpr_fps"], 1)
        self.assertEqual(result["latency_ms"]["detect"]["mean_ms"], 4)
        self.assertEqual(result["latency_ms"]["worker_cycle"]["max_ms"], 15)
        self.assertNotIn("invalid", result["latency_ms"])
        self.assertIsNone(result["capture_fps_last"])
        self.assertIsNone(result["dropped_capture"])

    def test_capture_rate_is_read_cadence_not_container_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay = VideoReplay(
                Path("fixture.mp4"), Path(tmp), metadata(), FakeCV()
            )
            with patch(
                "process_video.time.monotonic", side_effect=[100.0, 100.5]
            ):
                replay.read()
                replay.read()
            self.assertEqual(replay.capture_fps, 2)
            self.assertEqual(replay.source_fps, 10)
            replay.release()

    def test_file_report_contains_latency_and_drops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.mp4"
            path.touch()
            destination = Path(tmp) / "out"
            with patch("process_video.mov_timing", return_value=metadata()):
                report = process_one(
                    path, destination, "cpu", None, FakeCV(), runner
                )
            stats = json.loads((destination / "statistics.json").read_text())
            self.assertEqual(report["statistics"]["processed_frames"], 3)
            self.assertEqual(stats["dropped_capture"], 0)
            self.assertEqual(stats["dropped_processing"], 0)
            self.assertEqual(
                stats["statistics"]["latency_ms"]["total"]["count"], 3
            )
            self.assertFalse(stats["api_delivery"]["enabled"])

    def test_live_publisher_is_opt_in_and_updates_shared_results(self):
        class Publisher:
            dropped = 0

            def __init__(self, *a, **k):
                self.updates = []

            def update_results(self, *a, **k):
                self.updates.append(a)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("live_preview.PreviewPublisher", Publisher),
        ):
            replay = VideoReplay(
                Path("fixture.mp4"),
                Path(tmp),
                metadata(),
                FakeCV(),
                live=True,
                api_url="http://127.0.0.1:1/plates",
            )
            replay.preview.update_results([], 3.0, 5.0, time.monotonic())
            self.assertEqual(replay.anpr_fps, 3.0)
            self.assertEqual(replay.publisher.updates[0][1], 3.0)
            self.assertTrue(
                replay.image_prefix.startswith("/captures/video-runs/")
            )
            self.assertTrue(replay.delivery["enabled"])
            replay.release()
