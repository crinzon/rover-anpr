import importlib.util
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from rover_event_receiver import Receiver


class ReceiverTest(unittest.TestCase):
    def post(self, base, body):
        request = urllib.request.Request(
            base + "/plates",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.load(response)

    def event(self):
        return dict(
            session_id="unit-fixture",
            plates=[
                dict(
                    event_id="same-event",
                    track_id=1,
                    plate="K477##797",
                    subtype="type1",
                    status="PARTIAL",
                    confidence=0.9,
                    quality=0.8,
                    position=dict(
                        x=0.1, y=-0.2, offset_x_px=64, offset_y_px=-96
                    ),
                )
            ],
        )

    def run_server(self, path):
        server = Receiver(("127.0.0.1", 0), path)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(
            lambda: (server.shutdown(), server.server_close(), thread.join(2))
        )
        return server, f"http://127.0.0.1:{server.server_port}"

    def test_existing_fields_and_duplicate_event_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            server, base = self.run_server(tmp)
            try:
                body = self.event()
                self.post(base, body)
                self.assertEqual(self.post(base, body)["duplicate_events"], 1)
                rows = [
                    json.loads(record)
                    for record in (Path(tmp) / "events.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["event"], body["plates"][0])
            finally:
                server.shutdown()
                server.server_close()

    def test_invalid_existing_payload_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            server, base = self.run_server(tmp)
            try:
                with self.assertRaises(urllib.error.HTTPError) as error:
                    self.post(base, {"plates": []})
                self.assertEqual(error.exception.code, 422)
            finally:
                server.shutdown()
                server.server_close()

    def test_receiver_restart_retains_event_id_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            server, base = self.run_server(tmp)
            self.post(base, self.event())
            server.shutdown()
            server.server_close()
            second, base = self.run_server(tmp)
            try:
                self.assertEqual(
                    self.post(base, self.event())["duplicate_events"], 1
                )
            finally:
                second.shutdown()
                second.server_close()


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required")
class FrameIdentityTest(unittest.TestCase):
    def test_mismatched_ids_never_draw_boxes_even_at_same_timestamp(self):
        import numpy as np

        from live_preview import PreviewPublisher, PreviewReader

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preview"
            publisher = PreviewPublisher({"LIVE_FPS": "30"}, path)
            reader = PreviewReader(path)
            frame = np.zeros((100, 200, 3), np.uint8)
            plates = [
                dict(
                    box=[10, 20, 80, 50],
                    text="K477TK797",
                    confidence=0.96,
                    confirmed=True,
                )
            ]
            try:
                captured = time.monotonic()
                publisher.update_results(plates, 5, 30, captured, frame_id=10)
                publisher.submit(frame, captured, frame_id=11)
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    _, meta = reader.read()
                    if meta.get("frame_id") == 11:
                        break
                    time.sleep(0.01)
                self.assertFalse(meta["recognition"])
                self.assertEqual(meta["plates"], [])
                captured = time.monotonic()
                publisher.update_results(plates, 5, 30, captured, frame_id=12)
                publisher.submit(frame, captured, frame_id=12)
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    _, meta = reader.read()
                    if meta.get("frame_id") == 12:
                        break
                    time.sleep(0.01)
                self.assertTrue(meta["recognition"])
                self.assertEqual(
                    [
                        {k: v for k, v in plate.items() if k != "crop"}
                        for plate in meta["plates"]
                    ],
                    plates,
                )
                self.assertTrue(
                    all(
                        plate.get("crop", "").startswith(
                            "data:image/jpeg;base64,"
                        )
                        for plate in meta["plates"]
                    )
                )
            finally:
                publisher.close()

    def test_bridge_uses_measured_fps_and_counts_skips_after_first_frame(self):
        import numpy as np

        from webcam_anpr import WebcamReplay

        sequence = iter(
            [
                (
                    np.zeros((100, 200, 3), np.uint8),
                    dict(
                        frame_id=n,
                        capture_session="real-format-test",
                        mode="NORMAL",
                        captured_at=time.time(),
                        media_time=i * 0.04,
                        capture_fps=25.0,
                        capture_read_ms=32.0,
                    ),
                )
                for i, n in enumerate((10, 13))
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            replay = WebcamReplay(
                "",
                lambda p: None,
                Path(tmp) / "capture",
                loader=lambda p: next(sequence),
            )
            self.assertTrue(replay.read()[0])
            self.assertTrue(replay.read()[0])
            self.assertEqual(replay.capture_fps, 25.0)
            self.assertEqual(replay.capture_read_ms, 32.0)
            self.assertEqual(replay.dropped_frames, 2)
            replay.release()

    def test_newer_completion_cannot_replace_metadata_of_pending_matched_frame(
        self,
    ):
        import numpy as np

        from live_preview import PreviewPublisher, PreviewReader

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preview"
            publisher = PreviewPublisher({"LIVE_FPS": "5"}, path)
            reader = PreviewReader(path)
            frame = np.zeros((100, 200, 3), np.uint8)
            plates = [
                dict(
                    box=[10, 20, 80, 50],
                    text="K477TK797",
                    confidence=0.96,
                    confirmed=True,
                )
            ]
            try:
                captured = time.monotonic()
                publisher.update_results([], 1, 30, captured, frame_id=1)
                publisher.submit(frame, captured, frame_id=1)
                deadline = time.monotonic() + 2
                while not path.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                captured = time.monotonic()
                publisher.update_results(plates, 5, 30, captured, frame_id=2)
                publisher.submit(frame, captured, frame_id=2)
                publisher.update_results([], 5, 30, captured, frame_id=3)
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    _, meta = reader.read()
                    if meta.get("frame_id") == 2:
                        break
                    time.sleep(0.005)
                self.assertEqual(meta["result_frame_id"], 2)
                self.assertEqual(
                    [
                        {k: v for k, v in plate.items() if k != "crop"}
                        for plate in meta["plates"]
                    ],
                    plates,
                )
                self.assertTrue(
                    all(
                        plate.get("crop", "").startswith(
                            "data:image/jpeg;base64,"
                        )
                        for plate in meta["plates"]
                    )
                )
                self.assertTrue(meta["recognition"])
            finally:
                publisher.close()

    def test_webcam_device_selection_is_explicit_and_cpu_default_is_preserved(
        self,
    ):
        import sys
        import types
        from unittest.mock import patch

        import webcam_anpr as webcam

        for requested in (None, "cuda"):
            with (
                self.subTest(device=requested),
                tempfile.TemporaryDirectory() as tmp,
            ):
                settings = []
                fake = types.SimpleNamespace(
                    main=lambda config, **kw: settings.append(config)
                )
                with (
                    patch.dict(sys.modules, {"camera": fake}),
                    patch.object(
                        webcam, "__file__", str(Path(tmp) / "webcam_anpr.py")
                    ),
                ):
                    if requested is None:
                        webcam.run_wsl("http://127.0.0.1:8765/video")
                    else:
                        webcam.run_wsl(
                            "http://127.0.0.1:8765/video", device=requested
                        )
                self.assertEqual(settings[0]["DEVICE"], requested or "cpu")
                self.assertEqual(settings[0]["SHOW_WINDOW"], "false")
