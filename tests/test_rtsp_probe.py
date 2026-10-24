import importlib.util
import signal
import subprocess
import sys
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from check_rtsp import probe
from plate_validator import confirmed_confidence
from video_stream import VideoStream


class ProbeTest(unittest.TestCase):
    def test_confirmed_number_leaves_history(self):
        history = deque([dict(plate="A123AA77", confidence=0.9)], maxlen=10)
        for i in range(10):
            history.append(dict(plate=f"B{i % 2}23BB77", confidence=0.95))
        self.assertIsNone(confirmed_confidence(history, "A123AA77"))
        self.assertAlmostEqual(confirmed_confidence(history, "B023BB77"), 0.95)

    def test_bounded_probe_and_summary(self):
        class Stream:
            stop = threading.Event()
            released = False

            def read(self):
                time.sleep(0.005)
                return True, SimpleNamespace(shape=(720, 1280, 3))

            def release(self):
                self.released = True

        stream = Stream()
        messages = []
        self.assertEqual(probe(stream, 0.04, messages.append), 0)
        self.assertTrue(stream.released)
        for field in (
            "1280x720",
            "frames=",
            "FPS=",
            "first_frame=",
            "read_wait=",
        ):
            self.assertIn(field, messages[0])

    def test_signals_interrupt_probe_without_models(self):
        root = Path(__file__).resolve().parents[1]
        script = """
import signal
from check_rtsp import probe
from video_stream import VideoStream
stream = VideoStream("unused", initial=10, maximum=10)
for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, lambda *_: stream.stop.set())
print("ready", flush=True)
probe(stream, 30)
"""
        for sig in (signal.SIGTERM, signal.SIGINT):
            process = subprocess.Popen(
                [sys.executable, "-c", script],
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(process.stdout.readline().strip(), "ready")
                process.send_signal(sig)
                out, err = process.communicate(timeout=3)
                self.assertEqual(process.returncode, 0, err)
                self.assertIn("Stream:", out)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_settings_validation(self):
        for kwargs in (dict(backend="bad"), dict(transport="bad")):
            with self.assertRaises(ValueError):
                VideoStream(0, **kwargs)

    def test_probe_has_no_model_imports(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("full_stand.pipeline", sys.modules)
        self.assertIsNotNone(importlib.util.find_spec("check_rtsp"))
