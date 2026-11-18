"""Reacquisition uses the production worker and existing SQLite upsert."""

import contextlib
import importlib.util
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from plate_evidence import reid_match, resolve


def candidate(text, center=(0.3, 0.5), width=0.12):
    x, y = center
    return dict(
        plate=text,
        subtype="type1",
        center=center,
        normalized_box=[x - width / 2, y - 0.04, x + width / 2, y + 0.04],
    )


def supported_track(text="K477##797", moving=False):
    history = [
        dict(
            frame_id=i + 1,
            track_id=1,
            plate=text,
            subtype="type1",
            confidence=0.96,
            quality=0.8,
            tracking_time=i * 0.1,
            center=(0.2 + i * 0.01, 0.5) if moving else (0.3, 0.5),
        )
        for i in range(3)
    ]
    result = resolve(history)
    box = candidate(text, history[-1]["center"])["normalized_box"]
    return dict(
        history=history,
        last_seen=0.2,
        center=history[-1]["center"],
        normalized_box=box,
        subtype="type1",
        association_number=result["plate"],
        association_evidence=result["character_evidence"],
    )


class ReacquisitionRulesTest(unittest.TestCase):
    def test_supported_partial_returns_at_multiple_gaps_not_just_fixture(self):
        for gap in (0.2, 0.8, 1.3, 2.0, 2.9):
            with self.subTest(gap=gap):
                self.assertIsNotNone(
                    reid_match(
                        candidate("K477TK797"), supported_track(), 0.2 + gap
                    )
                )

    def test_expired_or_unknown_or_conflicting_evidence_never_recovers(self):
        track = supported_track()
        self.assertIsNone(reid_match(candidate("K477TK797"), track, 3.3))
        for text in (
            "#########",
            "K#####797",
            "Y624TK777",
            "H372PH777",
            "K478##797",
        ):
            self.assertIsNone(reid_match(candidate(text), track, 1.5))

    def test_same_text_elsewhere_and_wrong_motion_are_not_identity(self):
        self.assertIsNone(
            reid_match(
                candidate("K477TK797", (0.8, 0.5)), supported_track(), 1.5
            )
        )
        moving = supported_track(moving=True)
        self.assertIsNotNone(
            reid_match(candidate("K477TK797", (0.35, 0.5)), moving, 1.5)
        )
        self.assertIsNone(
            reid_match(candidate("K477TK797", (0.22, 0.5)), moving, 1.5)
        )

    def test_no_history_geometry_or_support_means_no_reid(self):
        track = supported_track()
        track["association_evidence"] = []
        self.assertIsNone(reid_match(candidate("K477TK797"), track, 1.5))
        track = supported_track()
        track.pop("normalized_box")
        self.assertIsNone(reid_match(candidate("K477TK797"), track, 1.5))


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "WSL camera runtime required"
)
class WorkerReacquisitionTest(unittest.TestCase):
    def run_case(self, sequence):
        import cv2
        import numpy as np

        from camera import main
        from webcam_anpr import WebcamReplay

        frame = np.zeros((240, 400, 3), np.uint8)
        packets = []
        events = []
        index = 0

        def load(previous):
            nonlocal index
            if index == len(sequence):
                return None
            media, _ = sequence[index]
            index += 1
            return frame, dict(
                frame_id=index,
                capture_session="reacquisition",
                mode="NORMAL",
                media_time=media,
                captured_at=time.time(),
            )

        class Pipeline:
            calls = 0

            def __call__(self, payload):
                _, items = sequence[self.calls]
                self.calls += 1
                return dict(
                    plates=[
                        dict(
                            text=text,
                            raw_text=text,
                            subtype=item[2] if len(item) > 2 else "type1",
                            readable=True,
                            valid=True,
                            confidence=0.96,
                            detection=0.99,
                            box=box,
                            characters=[
                                dict(position=i, char=c, probability=0.99)
                                for i, c in enumerate(text)
                            ],
                        )
                        for item in items
                        for text, box in [item[:2]]
                    ],
                    timing_ms={"total": 1},
                )

        class Sender:
            def __init__(self, *a, **k):
                pass

            def submit(self, session, event):
                events.append((session, dict(event)))
                return True

            def close(self):
                pass

        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.ExitStack() as stack,
        ):
            replay = WebcamReplay(
                "", packets.append, Path(tmp) / "capture", loader=load
            )
            replay.sender_factory = Sender
            for name in (
                "imshow",
                "waitKey",
                "namedWindow",
                "destroyAllWindows",
            ):
                stack.enter_context(
                    patch.object(
                        cv2, name, side_effect=AssertionError("No WSL GUI")
                    )
                )
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    dict(DEVICE="cpu", SHOW_WINDOW="false"),
                    replay=replay,
                    pipeline_override=Pipeline(),
                )
            self.assertFalse(replay.worker_error)
        self.assertEqual(len(packets), len(sequence))
        self.events = events
        return packets

    def row(self, text, box=None):
        return [(text, box or [80.0, 100.0, 200.0, 150.0])]

    def stable(self, text, times=(0, 0.05, 0.1), box=None):
        return [(t, self.row(text, box)) for t in times]

    def test_lost_partial_recovers_to_full_same_track_and_database_record(
        self,
    ):
        import database
        from current_plates import CurrentPlates

        seq = (
            self.stable("K477##797")
            + [(0.5, []), (1.0, [])]
            + self.stable("K477TK797", (1.4, 1.45, 1.5))
        )
        packets = self.run_case(seq)
        self.assertEqual(
            {p["plates"][0]["track_id"] for p in packets if p["plates"]}, {1}
        )
        self.assertEqual(packets[-1]["plates"][0]["status"], "CONFIRMED")
        self.assertEqual(self.events[0][1]["status"], "PARTIAL")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(database, "DB_PATH", Path(tmp) / "test.sqlite"),
        ):
            database.init_db()
            store = CurrentPlates()
            for i, (session, event) in enumerate(self.events):
                event["event_id"] = str(i)
                store.update(session, [event], database.apply_event)
            rows = database.get_detections()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["plate"], "K477TK797")
            self.assertEqual(rows[0]["status"], "CONFIRMED")

    def test_old_identity_cannot_return_after_retention_deadline(self):
        packets = self.run_case(
            self.stable("K477TK797")
            + [(1.0, [])]
            + self.stable("K477TK797", (3.3, 3.35, 3.4))
        )
        self.assertNotEqual(
            packets[2]["plates"][0]["track_id"],
            packets[-1]["plates"][0]["track_id"],
        )

    def test_required_p0_different_car_after_loss_starts_fresh(self):
        packets = self.run_case(
            self.stable("Y624TK777")
            + [(0.3, [])]
            + self.stable("H372PH777", (1.4, 1.45, 1.5))
        )
        self.assertNotEqual(
            packets[2]["plates"][0]["track_id"],
            packets[4]["plates"][0]["track_id"],
        )
        self.assertEqual(packets[4]["plates"][0]["status"], "PENDING")
        self.assertEqual(packets[-1]["plates"][0]["text"], "H372PH777")

    def test_one_symbol_different_plate_after_short_loss_is_separate(self):
        packets = self.run_case(
            self.stable("K477TK797")
            + [(0.15, [])]
            + self.stable("K477TX797", (0.2, 0.25, 0.3))
        )
        self.assertNotEqual(
            packets[2]["plates"][0]["track_id"],
            packets[4]["plates"][0]["track_id"],
        )
        self.assertEqual(packets[-1]["plates"][0]["text"], "K477TX797")

    def test_same_plate_at_distant_location_is_not_reidentified(self):
        packets = self.run_case(
            self.stable("K477TK797")
            + [(1.0, [])]
            + self.stable(
                "K477TK797", (1.4, 1.45, 1.5), [280.0, 100.0, 380.0, 150.0]
            )
        )
        self.assertNotEqual(
            packets[2]["plates"][0]["track_id"],
            packets[-1]["plates"][0]["track_id"],
        )

    def test_ambiguous_two_lost_tracks_do_not_force_a_match(self):
        old = [
            ("K477TK797", [80.0, 100.0, 160.0, 150.0]),
            ("K477TK797", [140.0, 100.0, 220.0, 150.0]),
        ]
        sequence = (
            [(t, old) for t in (0, 0.05, 0.1)]
            + [(1.0, [])]
            + [(1.4, self.row("K477TK797", [110.0, 100.0, 190.0, 150.0]))]
        )
        packets = self.run_case(sequence)
        oldids = {p["track_id"] for p in packets[2]["plates"]}
        self.assertNotIn(packets[-1]["plates"][0]["track_id"], oldids)

    def test_two_candidates_for_one_lost_track_are_both_left_separate(self):
        new = [
            ("K477TK797", [70.0, 100.0, 190.0, 150.0]),
            ("K477TK797", [90.0, 100.0, 210.0, 150.0]),
        ]
        packets = self.run_case(
            self.stable("K477TK797") + [(1.0, []), (1.4, new)]
        )
        self.assertNotIn(1, {p["track_id"] for p in packets[-1]["plates"]})

    def test_full_occlusion_after_loss_is_not_enough_for_reid(self):
        packets = self.run_case(
            self.stable("K477TK797")
            + [(1.0, []), (1.4, self.row("#########"))]
        )
        self.assertNotEqual(packets[-1]["plates"][0]["track_id"], 1)

    def test_noisy_temporary_track_cannot_steal_return_after_camera_motion(
        self,
    ):
        seq = self.stable("K477TK797", box=[80.0, 100.0, 160.0, 150.0])
        seq += [
            (0.15, self.row("Y624TK777", [100.0, 90.0, 180.0, 140.0])),
            (0.2, self.row("Y624TK777", [125.0, 80.0, 205.0, 130.0])),
        ]
        seq += self.stable(
            "K477TK797", (0.25, 0.3, 0.35), [150.0, 70.0, 230.0, 120.0]
        )
        packets = self.run_case(seq)
        self.assertEqual(packets[-1]["plates"][0]["track_id"], 1)
        self.assertEqual({e["track_id"] for _, e in self.events}, {1})
        self.assertEqual(packets[-1]["plates"][0]["text"], "K477TK797")

    def test_rejected_ocr_geometry_can_bridge_but_never_vote(self):
        seq = self.stable("K477TK797", box=[80.0, 100.0, 160.0, 150.0])
        seq += [
            (0.15, self.row("", [100.0, 90.0, 180.0, 140.0])),
            (0.2, self.row("", [125.0, 80.0, 205.0, 130.0])),
        ]
        seq += self.stable(
            "K477TK797", (0.25, 0.3, 0.35), [150.0, 70.0, 230.0, 120.0]
        )
        packets = self.run_case(seq)
        self.assertEqual(packets[-1]["plates"][0]["track_id"], 1)
        self.assertEqual({e["track_id"] for _, e in self.events}, {1})
        self.assertEqual(
            len(self.events), 4
        )  # Only actual accepted OCR frames.

    def test_geometry_without_ocr_cannot_keep_identity_alive_forever(self):
        seq = self.stable("K477TK797") + [
            (1.0, self.row("")),
            (2.0, self.row("")),
            (3.1, self.row("")),
        ]
        seq += self.stable("K477TK797", (3.4, 3.45, 3.5))
        packets = self.run_case(seq)
        self.assertNotEqual(packets[-1]["plates"][0]["track_id"], 1)

    def test_ambiguous_short_loss_is_not_resolved_by_nearest_center(self):
        old = [
            ("K477TK797", [80.0, 100.0, 160.0, 150.0]),
            ("K477TK797", [140.0, 100.0, 220.0, 150.0]),
        ]
        seq = [(t, old) for t in (0, 0.05, 0.1)] + [
            (0.15, []),
            (0.2, self.row("K477TK797", [110.0, 100.0, 190.0, 150.0])),
        ]
        packets = self.run_case(seq)
        self.assertNotIn(packets[-1]["plates"][0]["track_id"], {1, 2})

    def test_two_short_loss_candidates_cannot_borrow_one_identity(self):
        new = [
            ("K477TK797", [70.0, 100.0, 190.0, 150.0]),
            ("K477TK797", [90.0, 100.0, 210.0, 150.0]),
        ]
        packets = self.run_case(
            self.stable("K477TK797") + [(0.15, []), (0.2, new)]
        )
        self.assertNotIn(1, {p["track_id"] for p in packets[-1]["plates"]})

    def test_similar_new_car_without_missing_frame_needs_own_spatial_track(
        self,
    ):
        packets = self.run_case(
            self.stable("K477TK797")
            + self.stable(
                "K477TX797", (0.15, 0.2, 0.25), [210.0, 100.0, 290.0, 150.0]
            )
        )
        self.assertNotEqual(
            packets[2]["plates"][0]["track_id"],
            packets[3]["plates"][0]["track_id"],
        )
        self.assertEqual(packets[3]["plates"][0]["status"], "PENDING")
        self.assertEqual(packets[-1]["plates"][0]["text"], "K477TX797")
