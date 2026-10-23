"""Capture tests without OpenCV, network or model imports."""

import time
import unittest

from video_stream import VideoStream, safe_source


class FakeCapture:
    def __init__(self, source):
        self.source = source
        self.reads = 0

    def isOpened(self):
        return self.source != "closed"

    def read(self):
        self.reads += 1
        if self.source == "hang":
            time.sleep(30)
        if self.source == "error":
            raise OSError("secret")
        return (True, b"fresh") if self.reads == 1 else (False, None)

    def release(self):
        pass


class VideoStreamTest(unittest.TestCase):
    def stream(self, source):
        stream = VideoStream(
            source,
            initial=0.01,
            maximum=0.02,
            timeout=0.5,
            factory=FakeCapture,
            logger=lambda message: None,
        )
        self.addCleanup(stream.release)
        return stream

    def test_reconnect_does_not_reuse_failed_frame(self):
        stream = self.stream(0)
        self.assertEqual(stream.read(), (True, b"fresh"))
        self.assertEqual(stream.read(), (False, None))
        self.assertIsNone(stream.process)
        self.assertEqual(stream.read(), (True, b"fresh"))
        self.assertEqual(stream.delay, stream.initial)

    def test_backoff_is_capped(self):
        stream = self.stream("closed")
        for _ in range(4):
            self.assertEqual(stream.read(), (False, None))
        self.assertEqual(stream.delay, 0.02)

    def test_hung_read_is_terminated(self):
        stream = self.stream("hang")
        started = time.monotonic()
        self.assertEqual(stream.read(), (False, None))
        self.assertLess(time.monotonic() - started, 3)
        self.assertIsNone(stream.process)

    def test_exception_is_not_logged(self):
        stream = self.stream("error")
        messages = []
        stream.logger = messages.append
        self.assertEqual(stream.read(), (False, None))
        self.assertNotIn("secret", "".join(messages))

    def test_close_interrupts_backoff(self):
        stream = self.stream("closed")
        stream.read()
        stream.release()
        self.assertEqual(stream.read(), (False, None))

    def test_url_masking(self):
        masked = safe_source("rtsp://user:password@camera/path?token=secret")
        for secret in ("user", "password", "token", "secret"):
            self.assertNotIn(secret, masked)

    def test_invalid_settings(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                VideoStream(0, initial=value)
