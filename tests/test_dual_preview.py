import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from live_preview import PreviewReader
from video_settings import load_env


class DualPreviewTest(unittest.TestCase):
    def test_two_files_keep_frames_and_tracks_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            readers = []
            for name, number in [("Front", 11), ("Back", 22)]:
                path = Path(directory) / name
                path.write_bytes(
                    json.dumps(
                        dict(
                            timestamp=time.time(),
                            frame_id=number,
                            result_frame_id=number,
                            plates=[
                                dict(track_id=1, text=name, status="PARTIAL")
                            ],
                        )
                    ).encode()
                    + b"\n\xff\xd8frame\xff\xd9"
                )
                readers.append(PreviewReader(path))
            first, second = [r.read()[1] for r in readers]
            self.assertEqual((first["frame_id"], second["frame_id"]), (11, 22))
            self.assertNotEqual(
                first["plates"][0]["text"], second["plates"][0]["text"]
            )

    def test_lost_camera_does_not_hide_other_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Back"
            path.write_bytes(
                json.dumps(dict(timestamp=time.time(), frame_id=2)).encode()
                + b"\n\xff\xd8frame\xff\xd9"
            )
            self.assertEqual(
                PreviewReader(Path(directory) / "Front").read(), (None, {})
            )
            self.assertEqual(PreviewReader(path).read()[1]["frame_id"], 2)

    def test_windows_bom_is_accepted_without_exposing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / ".env").write_text(
                "CAMERA_USER=test\nCAMERA_PASSWORD=local-only",
                encoding="utf-8-sig",
            )
            with (
                patch(
                    "video_settings.__file__",
                    str(Path(directory) / "video_settings.py"),
                ),
                patch.dict("os.environ", {}, clear=True),
            ):
                settings = load_env()
                self.assertEqual(settings["CAMERA_USER"], "test")
                self.assertEqual(settings["CAMERA_PASSWORD"], "local-only")


if __name__ == "__main__":
    unittest.main()
