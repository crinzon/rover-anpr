import importlib.util
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluate_videos import mov_timing, percentile


def box(kind, payload):
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


class EvaluationTest(unittest.TestCase):
    def test_original_mov_sample_timing(self):
        mdhd = box(
            b"mdhd", bytes(12) + struct.pack(">II", 600, 1200) + bytes(4)
        )
        hdlr = box(b"hdlr", bytes(8) + b"vide" + bytes(12))
        stts = box(
            b"stts",
            bytes(4)
            + struct.pack(">I", 2)
            + struct.pack(">IIII", 15, 20, 15, 60),
        )
        video = box(
            b"moov",
            box(
                b"trak",
                box(b"mdia", mdhd + hdlr + box(b"minf", box(b"stbl", stts))),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.MOV"
            path.write_bytes(video)
            info = mov_timing(path)
        self.assertEqual(info["frames"], 30)
        self.assertEqual(info["duration_s"], 2)
        self.assertEqual(info["fps"], 15)
        self.assertTrue(info["variable_frame_rate"])

    def test_percentiles(self):
        self.assertEqual(percentile([10, 20, 30], 50), 20)
        self.assertEqual(percentile([10, 20, 30], 95), 29)
        self.assertIsNone(percentile([], 50))

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(name)
            for name in ("cv2", "torch", "ultralytics")
        ),
        "Camera runtime dependencies unavailable",
    )
    def test_actual_camera_worker_replay(self):
        script = Path(__file__).with_name("check_replay.py")
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
