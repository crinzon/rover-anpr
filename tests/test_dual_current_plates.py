import base64
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from live_preview import PreviewPublisher, PreviewReader


class CurrentPlatesTests(unittest.TestCase):
    def wait_frame(self, reader, frame_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            jpeg, meta = reader.read()
            if meta.get("frame_id") == frame_id:
                return jpeg, meta
            time.sleep(0.01)
        self.fail("Preview frame not published")

    def test_all_crops_match_source_including_unreadable_and_disappear(self):
        with tempfile.TemporaryDirectory() as directory:
            publisher = PreviewPublisher(
                {"LIVE_FPS": "30"}, Path(directory) / "preview"
            )
            reader = PreviewReader(publisher.path)
            frame = np.zeros((120, 200, 3), np.uint8)
            frame[10:30, 10:60] = (10, 70, 200)
            frame[60:80, 100:170] = (200, 70, 10)
            original = frame.copy()
            rows = [
                dict(
                    box=[10, 10, 60, 30],
                    text="A111AA77",
                    ocr_text="A111AA77",
                    confidence=0.95,
                    confirmed=True,
                    status="CONFIRMED",
                    track_id=1,
                ),
                dict(
                    box=[100, 60, 170, 80],
                    text="B2#2BB77",
                    ocr_text="B222BB77",
                    confidence=0.8,
                    confirmed=False,
                    status="PARTIAL",
                    track_id=2,
                ),
                dict(
                    box=[5, 90, 40, 110],
                    text="",
                    ocr_text="",
                    confidence=0.1,
                    confirmed=False,
                    status="UNREADABLE",
                ),
            ]
            try:
                stamp = time.monotonic()
                publisher.update_results(rows, 10, 50, stamp, frame_id=7)
                publisher.submit(frame, stamp, frame_id=7)
                _, meta = self.wait_frame(reader, 7)
                self.assertEqual(meta["frame_id"], meta["result_frame_id"])
                self.assertEqual(len(meta["plates"]), 3)
                for index, color in enumerate(((10, 70, 200), (200, 70, 10))):
                    raw = base64.b64decode(
                        meta["plates"][index]["crop"].split(",")[1]
                    )
                    image = cv2.imdecode(
                        np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR
                    )
                    self.assertTrue(
                        np.allclose(image.mean(axis=(0, 1)), color, atol=3)
                    )
                self.assertEqual(meta["plates"][2]["text"], "")
                self.assertEqual(meta["plates"][2]["status"], "UNREADABLE")
                self.assertTrue(np.array_equal(frame, original))
                stamp = time.monotonic()
                publisher.update_results([], 10, 50, stamp, frame_id=8)
                publisher.submit(frame, stamp, frame_id=8)
                self.assertEqual(self.wait_frame(reader, 8)[1]["plates"], [])
            finally:
                publisher.close()

    def test_api_bundle_keeps_frame_and_all_results_in_one_response(self):
        import api

        jpeg = b"\xff\xd8example\xff\xd9"
        meta = dict(
            timestamp=time.time(),
            frame_id=9,
            result_frame_id=9,
            session_id="Back-run",
            plates=[
                dict(
                    text="",
                    status="UNREADABLE",
                    crop="data:image/jpeg;base64,test",
                )
            ],
        )
        with patch.object(api, "camera_preview", return_value=(jpeg, meta)):
            bundle = json.loads(
                api.live_frame(camera="Back", bundle=True).body
            )
            self.assertEqual(
                base64.b64decode(bundle["image"].split(",")[1]), jpeg
            )
            self.assertEqual(bundle["frame_id"], bundle["result_frame_id"])
            self.assertEqual(bundle["session_id"], "Back-run")
            self.assertEqual(len(bundle["plates"]), 1)
            self.assertEqual(
                api.live_frame(camera="Back").media_type, "image/jpeg"
            )


if __name__ == "__main__":
    unittest.main()
