"""Shared webcam worker and Windows presentation contracts, without
hardware."""

import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import webcam_anpr as webcam


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Camera Torch runtime required"
)
class WebcamConfirmationTest(unittest.TestCase):
    def test_production_worker_auto_confirms_masks_and_recovers(self):
        from camera import main

        words = ["K477TK797"] * 3 + ["K477##797"] * 10 + ["K477TK797"] * 3
        packets = []
        frame = np.zeros((240, 400, 3), np.uint8)
        index = 0

        def load(previous):
            nonlocal index
            if index == len(words):
                return None
            index += 1
            return frame, dict(
                frame_id=100 + index * 7,
                capture_session="windows-session",
                mode="OCCLUDED" if 3 < index <= 13 else "NORMAL",
                captured_at=time.time(),
                media_time=index * 0.05,
            )

        class Pipeline:
            calls = 0

            def __call__(self, payload):
                text = words[self.calls]
                self.calls += 1
                return dict(
                    plates=[
                        dict(
                            text=text,
                            raw_text=text,
                            subtype="type1",
                            readable=True,
                            valid=True,
                            confidence=0.96,
                            detection=0.99,
                            box=[20.0, 40.0, 340.0, 95.0],
                            characters=[
                                dict(position=i, char=c, probability=0.99)
                                for i, c in enumerate(text)
                            ],
                        )
                    ],
                    timing_ms={"total": 1},
                )

        with tempfile.TemporaryDirectory() as temp:
            replay = webcam.WebcamReplay(
                "", packets.append, Path(temp) / "capture", loader=load
            )

            def forbidden(*a, **kw):
                raise AssertionError("WSL worker attempted GUI")

            with contextlib.ExitStack() as stack:
                for name in (
                    "imshow",
                    "waitKey",
                    "namedWindow",
                    "destroyAllWindows",
                ):
                    stack.enter_context(patch.object(cv2, name, forbidden))
                with contextlib.redirect_stdout(io.StringIO()):
                    main(
                        dict(DEVICE="cpu", SHOW_WINDOW="false"),
                        replay=replay,
                        pipeline_override=Pipeline(),
                    )
            self.assertFalse(replay.worker_error)
            self.assertEqual(len(packets), 16)
            states = [p["plates"][0]["status"] for p in packets]
            self.assertEqual(states[:3], ["PENDING", "PENDING", "CONFIRMED"])
            self.assertEqual(states[12], "PARTIAL")
            self.assertEqual(packets[12]["plates"][0]["text"], "K477##797")
            self.assertEqual(states[-1], "CONFIRMED")
            self.assertEqual(
                {p["plates"][0]["track_id"] for p in packets}, {1}
            )
            self.assertEqual(
                [p["frame_id"] for p in packets],
                [100 + i * 7 for i in range(1, 17)],
            )
            logger = webcam.CsvObservations(Path(temp) / "csv")
            for packet in packets:
                logger.add(packet)
            logger.close()
            for mode in ("NORMAL", "OCCLUDED"):
                with (Path(temp) / "csv" / f"{mode}.csv").open(
                    encoding="utf-8-sig"
                ) as handle:
                    rows = list(csv.DictReader(handle))
                self.assertTrue(rows)
                self.assertTrue(
                    all(
                        row["track_id"] == "1" and row["mode"] == mode
                        for row in rows
                    )
                )

    def test_windows_q_connection_reset_finishes_live_source(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        meta = dict(
            frame_id=1,
            capture_session="session",
            mode="NORMAL",
            captured_at=time.time(),
            media_time=0,
        )
        with tempfile.TemporaryDirectory() as temp:
            replay = webcam.WebcamReplay(
                "http://localhost:8765/video",
                lambda p: None,
                Path(temp) / "capture",
                loader=lambda p: (frame, meta),
            )
            self.assertTrue(replay.read()[0])
            replay.loader = None
            with patch.object(
                webcam.urllib.request,
                "urlopen",
                side_effect=ConnectionResetError("Windows Q"),
            ):
                self.assertEqual(replay.read(), (False, None))

    def test_source_restart_requires_a_new_tracker_session(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        sequence = iter(
            [
                (
                    frame,
                    dict(
                        frame_id=1,
                        capture_session="first",
                        mode="NORMAL",
                        captured_at=time.time(),
                        media_time=0,
                    ),
                ),
                (
                    frame,
                    dict(
                        frame_id=2,
                        capture_session="second",
                        mode="NORMAL",
                        captured_at=time.time(),
                        media_time=0.1,
                    ),
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            replay = webcam.WebcamReplay(
                "",
                lambda p: None,
                Path(temp) / "capture",
                loader=lambda p: next(sequence),
            )
            self.assertTrue(replay.read()[0])
            with self.assertRaisesRegex(RuntimeError, "capture restarted"):
                replay.read()


@unittest.skipUnless(sys.platform == "win32", "Windows viewer test")
class WindowsPresentationTest(unittest.TestCase):
    def test_delayed_result_is_drawn_on_its_source_frame_and_q_flushes(self):
        class Camera:
            index = 0
            released = False

            def isOpened(self):
                return True

            def read(self):
                self.index += 1
                return True, np.full((100, 200, 3), self.index * 11, np.uint8)

            def release(self):
                self.released = True

        class Server:
            def __init__(self, *a):
                pass

            def serve_forever(self):
                pass

            def shutdown(self):
                pass

            def server_close(self):
                pass

        plate = dict(
            text="K477TK797",
            ocr_text="K477TK797",
            raw_text="K477TK797",
            subtype="type1",
            status="CONFIRMED",
            confidence=0.96,
            track_id=7,
            box=[20, 40, 80, 60],
            filter_result="K477TK797",
            filter_accepted=True,
        )
        packet = dict(
            capture_session="capture",
            source_session="engine",
            frame_id=1,
            observed_at="now",
            mode="NORMAL",
            width=200,
            height=100,
            plates=[plate],
        )

        class Receiver:
            def __init__(self):
                self.items = iter([None, json.dumps(packet).encode(), None])

            def bind(self, *a):
                pass

            def setblocking(self, *a):
                pass

            def recvfrom(self, *a):
                value = next(self.items)
                if value is None:
                    raise BlockingIOError
                return value, ("127.0.0.1", 1)

            def close(self):
                pass

        camera = Camera()
        shown = []
        with (
            tempfile.TemporaryDirectory() as temp,
            contextlib.ExitStack() as stack,
        ):
            stack.enter_context(
                patch.object(
                    webcam, "__file__", str(Path(temp) / "webcam_anpr.py")
                )
            )
            stack.enter_context(
                patch.object(
                    webcam.uuid,
                    "uuid4",
                    return_value=type("ID", (), {"hex": "capture"})(),
                )
            )
            stack.enter_context(
                patch.object(cv2, "VideoCapture", return_value=camera)
            )
            stack.enter_context(
                patch.object(webcam.socket, "socket", return_value=Receiver())
            )
            stack.enter_context(
                patch("http.server.ThreadingHTTPServer", Server)
            )
            stack.enter_context(
                patch.object(
                    cv2,
                    "imshow",
                    side_effect=lambda title, img: shown.append(img.copy()),
                )
            )
            stack.enter_context(
                patch.object(cv2, "waitKey", side_effect=[-1, ord("q")])
            )
            stack.enter_context(patch.object(cv2, "destroyAllWindows"))
            with contextlib.redirect_stdout(io.StringIO()):
                webcam.run_windows()
            self.assertEqual(
                int(shown[-1][10, 10, 0]), 11
            )  # Frame 1, not live frame 2 (=22).
            self.assertTrue(camera.released)
            paths = list(Path(temp).rglob("NORMAL.csv"))
            self.assertEqual(len(paths), 1)
            with paths[0].open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["track_id"], "7")
            self.assertEqual(rows[0]["status"], "CONFIRMED")
            self.assertEqual(rows[0]["first_frame"], "1")


if __name__ == "__main__":
    unittest.main()
