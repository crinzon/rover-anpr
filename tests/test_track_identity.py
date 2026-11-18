"""Association regressions through the shared worker, without camera/models."""

import contextlib
import importlib.util
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from plate_evidence import identity_compatible, registry_supported, resolve


class IdentityGuardTest(unittest.TestCase):
    def test_required_car_change_is_not_compatible(self):
        self.assertFalse(
            identity_compatible(
                {"plate": "H372PH777"}, {"confirmed_number": "Y624TK777"}
            )
        )

    def test_unknown_positions_and_single_ocr_error_preserve_identity(self):
        for text in ("K477##797", "K477TX797", "#########"):
            self.assertTrue(
                identity_compatible(
                    {"plate": text}, {"confirmed_number": "K477TK797"}
                )
            )


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "WSL Torch runtime required"
)
class WorkerIdentityTest(unittest.TestCase):
    def run_sequence(self, words):
        import cv2
        import numpy as np

        from camera import main
        from webcam_anpr import WebcamReplay

        index = 0
        packets = []
        self.events = []
        frame = np.zeros((240, 400, 3), np.uint8)

        def load(previous):
            nonlocal index
            if index == len(words):
                return None
            index += 1
            return frame, dict(
                frame_id=index,
                capture_session="identity-test",
                mode="NORMAL",
                captured_at=time.time(),
                media_time=index * 0.05,
            )

        class Pipeline:
            calls = 0

            def __call__(self, payload):
                word = words[self.calls]
                self.calls += 1
                return dict(
                    plates=[
                        dict(
                            text=word,
                            raw_text=word,
                            subtype="type1",
                            readable=True,
                            valid=True,
                            confidence=0.96,
                            detection=0.99,
                            box=[20.0, 40.0, 340.0, 95.0],
                            characters=[
                                dict(position=i, char=c, probability=0.99)
                                for i, c in enumerate(word)
                            ],
                        )
                    ],
                    timing_ms={"total": 1},
                )

        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.ExitStack() as stack,
        ):
            replay = WebcamReplay(
                "", packets.append, Path(tmp) / "capture", loader=load
            )
            events = self.events

            class Sender:
                def __init__(self, *a, **kw):
                    pass

                def submit(self, session, event):
                    events.append((session, dict(event)))
                    return True

                def close(self):
                    pass

            replay.sender_factory = Sender
            for name in (
                "imshow",
                "waitKey",
                "namedWindow",
                "destroyAllWindows",
            ):
                stack.enter_context(
                    patch.object(
                        cv2, name, side_effect=AssertionError("GUI forbidden")
                    )
                )
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    dict(DEVICE="cpu", SHOW_WINDOW="false"),
                    replay=replay,
                    pipeline_override=Pipeline(),
                )
            self.assertFalse(replay.worker_error)
        self.assertEqual(len(packets), len(words))
        return [packet["plates"][0] for packet in packets]

    def test_y624_to_h372_starts_independent_pending_track(self):
        rows = self.run_sequence(["Y624TK777"] * 8 + ["H372PH777"] * 3)
        self.assertEqual(rows[7]["status"], "CONFIRMED")
        self.assertNotEqual(rows[7]["track_id"], rows[8]["track_id"])
        self.assertEqual(rows[8]["text"], "H372PH777")
        self.assertEqual(rows[8]["status"], "PENDING")
        self.assertEqual(rows[-1]["text"], "H372PH777")
        self.assertEqual(rows[-1]["status"], "CONFIRMED")

    def test_isolated_substitution_keeps_track_and_full_result(self):
        rows = self.run_sequence(
            ["K477TK797"] * 8 + ["K477TX797"] + ["K477TK797"]
        )
        self.assertEqual({r["track_id"] for r in rows}, {1})
        self.assertEqual(rows[8]["text"], "K477TK797")
        self.assertEqual(rows[8]["status"], "CONFIRMED")

    def test_partial_and_recovery_keep_identity(self):
        rows = self.run_sequence(
            ["K477TK797"] * 3 + ["K477##797"] * 10 + ["K477TK797"] * 3
        )
        self.assertEqual({r["track_id"] for r in rows}, {1})
        self.assertEqual(rows[12]["text"], "K477##797")
        self.assertEqual(rows[12]["status"], "PARTIAL")
        self.assertEqual(rows[-1]["status"], "CONFIRMED")

    def test_two_frame_false_object_is_live_but_not_persisted(self):
        rows = self.run_sequence(["B3#######"] * 2)
        self.assertEqual(rows[-1]["status"], "PARTIAL")
        self.assertEqual(self.events, [])

    def test_partial_confirmed_and_reocclusion_update_one_database_row(self):
        import database
        from current_plates import CurrentPlates

        self.run_sequence(
            ["K477##797"] * 3
            + ["K477TK797"] * 3
            + ["K477##797"] * 10
            + ["K477TK797"] * 3
        )
        events = self.events
        self.assertEqual(events[0][1]["plate"], "K477##797")
        self.assertEqual(events[0][1]["status"], "PARTIAL")
        self.assertEqual(events[-1][1]["status"], "CONFIRMED")
        self.assertEqual({e["track_id"] for _, e in events}, {1})
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(database, "DB_PATH", Path(tmp) / "test.db"):
                database.init_db()
                state = CurrentPlates()
                for index, (session, event) in enumerate(events):
                    event["event_id"] = "event-" + str(index)
                    state.update(session, [event], database.apply_event)
                rows = database.get_detections()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["plate"], "K477TK797")
                self.assertEqual(rows[0]["status"], "CONFIRMED")


class RegistrySupportTest(unittest.TestCase):
    def result(self, text, frames):
        return resolve(
            [
                dict(
                    frame_id=i,
                    plate=text,
                    subtype="type1",
                    confidence=0.96,
                    quality=0.8,
                )
                for i in frames
            ]
        )

    def test_existing_minimum_not_new_confidence_threshold(self):
        self.assertFalse(registry_supported(self.result("B3#######", [1, 2])))
        self.assertTrue(
            registry_supported(self.result("K477##797", [1, 2, 3]))
        )
        self.assertFalse(
            registry_supported(self.result("#########", [1, 2, 3]))
        )
        self.assertFalse(
            registry_supported(self.result("K477##797", [1, 1, 1]))
        )

    def test_no_claim_to_classify_persistent_false_objects(self):
        # Explicit limitation: temporal support is not visual object validity.
        self.assertTrue(
            registry_supported(self.result("B3#######", [1, 2, 3]))
        )
