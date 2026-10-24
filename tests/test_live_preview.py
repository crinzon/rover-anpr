import importlib.util
import json
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from live_preview import PreviewPublisher, PreviewReader, read_preview

JPEG = b"\xff\xd8test\xff\xd9"


def metadata(**changes):
    result = dict(
        timestamp=time.time(),
        fps=19.0,
        processing_ms=40.0,
        plates=[],
        width=1280,
        height=720,
        recognition=True,
    )
    result.update(changes)
    return result


class PreviewTest(unittest.TestCase):
    def test_read_stale_corrupt_missing_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview"
            self.assertEqual(read_preview(path), (None, {}))
            for data in (
                b"broken",
                b"{}\nnot jpeg",
                json.dumps(metadata(timestamp=time.time() - 10)).encode()
                + b"\n"
                + JPEG,
            ):
                path.write_bytes(data)
                self.assertEqual(read_preview(path), (None, {}))
            path.write_bytes(json.dumps(metadata()).encode() + b"\n" + JPEG)
            jpeg, meta = read_preview(path)
            self.assertEqual(jpeg, JPEG)
            self.assertEqual(meta["plates"], [])
            self.assertEqual(meta["fps"], 19.0)

    def test_metadata_does_not_forward_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview"
            path.write_bytes(
                json.dumps(
                    metadata(VIDEO_SOURCE="secret", password="secret")
                ).encode()
                + b"\n"
                + JPEG
            )
            self.assertNotIn("secret", json.dumps(read_preview(path)[1]))

    def test_publish_resizes_draws_and_rate_limits(self):
        ready = threading.Event()
        rectangles = []
        sizes = []

        class Frame:
            shape = (2160, 3840, 3)

            def copy(self):
                return self

        def resize(frame, size):
            sizes.append(size)
            return types.SimpleNamespace(shape=(size[1], size[0], 3))

        def encode(*args):
            ready.set()
            return True, types.SimpleNamespace(tobytes=lambda: JPEG)

        cv2 = types.SimpleNamespace(
            resize=resize,
            rectangle=lambda *args: rectangles.append(args),
            putText=lambda *args: None,
            imencode=encode,
            FONT_HERSHEY_SIMPLEX=0,
            IMWRITE_JPEG_QUALITY=1,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("sys.modules", cv2=cv2),
        ):
            path = Path(directory) / "preview"
            publisher = PreviewPublisher(
                {"LIVE_FPS": "5", "LIVE_MAX_WIDTH": "1280"}, path
            )
            try:
                plates = [
                    dict(
                        box=[300, 300, 600, 600],
                        text="A123AA77",
                        confidence=0.95,
                        confirmed=True,
                    )
                ]
                captured = time.monotonic()
                publisher.update_results(plates, 18.9, 40, captured)
                publisher.submit(Frame(), captured)
                self.assertTrue(ready.wait(1))
                for _ in range(100):
                    if path.exists():
                        break
                    time.sleep(0.005)
                self.assertEqual(read_preview(path)[1]["plates"], plates)
                self.assertEqual(sizes, [(1280, 720)])
                self.assertEqual(rectangles[0][1:3], ((100, 100), (200, 200)))
                publisher.submit(Frame(), time.monotonic())
                publisher.submit(Frame(), time.monotonic())
                self.assertLessEqual(publisher.dropped, 2)
            finally:
                publisher.close()
            self.assertFalse(path.exists())
            self.assertFalse(publisher.thread.is_alive())

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV unavailable")
    def test_real_jpeg_without_models_or_capture(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview"
            publisher = PreviewPublisher(
                {"LIVE_FPS": "5", "LIVE_MAX_WIDTH": "1280"}, path
            )
            try:
                frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
                captured = time.monotonic()
                publisher.update_results(
                    [
                        dict(
                            box=[300, 300, 600, 500],
                            text="A123AA77",
                            confidence=0.95,
                            confirmed=True,
                        )
                    ],
                    18.9,
                    40,
                    captured,
                )
                publisher.submit(frame, captured)
                deadline = time.monotonic() + 3
                while not path.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                jpeg, meta = read_preview(path)
                self.assertIsNotNone(jpeg)
                image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), 1)
                self.assertEqual(image.shape[:2], (720, 1280))
                self.assertTrue(image.any())
                self.assertFalse(
                    frame.any(), "Preview modified the ANPR input"
                )
                self.assertEqual(meta["fps"], 18.9)
                first_timestamp = meta["timestamp"]
                publisher.update_results([], 0, 0, time.monotonic() - 1)
                publisher.set_inference_error("worker_failed")
                publisher.submit(frame, time.monotonic())
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    jpeg, meta = read_preview(path)
                    if jpeg and meta["timestamp"] != first_timestamp:
                        break
                    time.sleep(0.01)
                self.assertNotEqual(meta["timestamp"], first_timestamp)
                self.assertFalse(meta["recognition"])
                self.assertEqual(meta["plates"], [])
                self.assertEqual(meta["inference_state"], "error")
                self.assertEqual(meta["inference_error"], "worker_failed")
            finally:
                publisher.close()

    def test_reader_caches_version_without_reusing_stale_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview"
            path.write_bytes(json.dumps(metadata()).encode() + b"\n" + JPEG)
            reader = PreviewReader(path)
            first = reader.read()
            with patch(
                "live_preview.json.loads", side_effect=AssertionError("reread")
            ):
                self.assertEqual(reader.read()[0], first[0])
            with patch(
                "live_preview.time.time", return_value=time.time() + 10
            ):
                self.assertEqual(reader.read(), (None, {}))

    def test_latest_slot_replaces_pending_without_frame_copy(self):
        with patch.object(PreviewPublisher, "_run", return_value=None):
            with tempfile.TemporaryDirectory() as directory:
                publisher = PreviewPublisher({}, Path(directory) / "preview")
                try:
                    frame = object()
                    for i in range(100):
                        publisher.submit(frame, i)
                    self.assertIs(publisher.pending[0], frame)
                    self.assertEqual(publisher.pending[1], 99)
                    self.assertEqual(publisher.dropped, 99)
                    self.assertEqual(publisher.fps, 20)
                    self.assertEqual(publisher.quality, 90)
                finally:
                    publisher.close()

    def test_reject_invalid_limits(self):
        for settings in (
            {"LIVE_FPS": "nan"},
            {"LIVE_FPS": "0"},
            {"LIVE_MAX_WIDTH": "99999"},
            {"LIVE_JPEG_QUALITY": "101"},
        ):
            with self.assertRaises(ValueError):
                PreviewPublisher(settings)


@unittest.skipUnless(
    importlib.util.find_spec("fastapi"), "FastAPI unavailable"
)
class LiveRoutesTest(unittest.IsolatedAsyncioTestCase):
    async def test_page_status_multipart_and_cancellation(self):
        import api

        self.assertTrue(api.live_page().path.is_file())
        with patch.object(api, "read_preview", return_value=(None, {})):
            self.assertEqual(api.live_status(), {"available": False})
            self.assertEqual(api.live_frame().status_code, 503)
        with patch.object(
            api, "read_diagnostics", return_value={"enabled": False}
        ):
            response = api.live_diagnostics()
            self.assertEqual(json.loads(response.body), {"enabled": False})
            self.assertEqual(response.headers["cache-control"], "no-store")
        meta = metadata()
        with patch.object(api, "read_preview", return_value=(JPEG, meta)):
            self.assertTrue(api.live_status()["available"])
            snapshot = api.live_frame()
            self.assertEqual(snapshot.body, JPEG)
            self.assertIn("no-store", snapshot.headers["cache-control"])
            self.assertEqual(
                api.live_frame(after=meta["timestamp"]).status_code, 204
            )
            response = await api.live_stream()
            self.assertIn("boundary=frame", response.media_type)
            self.assertEqual(response.headers["cache-control"], "no-store")
            iterator = response.body_iterator
            part = await anext(iterator)
            self.assertIn(b"Content-Type: image/jpeg", part)
            self.assertIn(JPEG, part)
            await iterator.aclose()
