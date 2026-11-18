"""Short isolated regressions: no models, cameras or production database."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_track_reacquisition import WorkerReacquisitionTest

import database
from plate_evidence import resolve


def item(i, text):
    return dict(
        frame_id=i,
        track_id=1,
        plate=text,
        subtype="type1",
        confidence=0.96,
        quality=0.8,
    )


class PartialUpdatesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.override = patch.object(
            database, "DB_PATH", Path(self.tmp.name) / "isolated.sqlite"
        )
        self.override.start()
        database.init_db()

    def tearDown(self):
        self.override.stop()
        self.tmp.cleanup()

    def event(
        self,
        i,
        text="H588#O797",
        track=1,
        status="PARTIAL",
        at=None,
        camera="Front",
    ):
        return dict(
            track_id=track,
            camera_id=camera,
            plate=text,
            status=status,
            subtype="type1",
            quality=0.8,
            confidence=0.96,
            observation_id="o" + str(i),
            event_id="e" + str(i),
            captured_at=float(i if at is None else at),
        )

    def test_repeat_partial_is_one_row_and_history_is_preserved(self):
        for i in (1, 2, 3):
            database.apply_event("test", self.event(i))
        rows = database.get_detections()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["observation_count"], 3)
        self.assertEqual(rows[0]["last_seen"], 3)
        self.assertFalse(database.apply_event("test", self.event(3)))
        self.assertEqual(database.get_detections()[0]["observation_count"], 3)

    def test_fill_unknown_only_confirms_after_three_independent_frames(self):
        history = [item(i, "H588#O797") for i in (1, 2, 3)]
        for i in (4, 5, 6):
            history.append(item(i, "H588HO797"))
            result = resolve(history)
            self.assertEqual(result["plate"], "H588HO797")
            self.assertEqual(
                result["status"], "CONFIRMED" if i == 6 else "PENDING"
            )
            database.apply_event(
                "test", self.event(i, result["plate"], status=result["status"])
            )
        self.assertEqual(len(database.get_detections()), 1)
        self.assertEqual(database.get_detections()[0]["status"], "CONFIRMED")

    def test_conflict_stays_unknown(self):
        result = resolve([item(1, "H588HO797"), item(2, "H588KO797")])
        self.assertEqual(result["plate"], "H588#O797")
        self.assertEqual(result["status"], "PARTIAL")

    def test_different_tracks_sessions_and_cameras_remain_separate(self):
        database.apply_event("s1", self.event(1))
        database.apply_event("s1", self.event(2, track=2))
        database.apply_event("s2", self.event(3))
        database.apply_event("s1", self.event(4, camera="Back"))
        self.assertEqual(len(database.get_detections()), 4)

    def test_out_of_order_observation_cannot_move_last_seen_back(self):
        database.apply_event("test", self.event(1, at=10))
        database.apply_event("test", self.event(2, at=9))
        self.assertEqual(database.get_detections()[0]["last_seen"], 10)
        self.assertEqual(database.get_detections()[0]["observation_count"], 2)

    def test_different_lengths_never_splice(self):
        result = resolve([item(1, "H588#O797"), item(2, "H588HO77")])
        self.assertNotEqual(result["status"], "CONFIRMED")


class OCRBridgeTest(WorkerReacquisitionTest):
    def test_supported_track_survives_rejected_ocr_with_sparse_recent_votes(
        self,
    ):
        seq = self.stable("H588#O###")
        seq += [(i * 0.05, self.row("#588#O###")) for i in range(3, 12)]
        seq += [
            (0.6, self.row("M588#O797")),
            (0.65, self.row("")),
            (0.7, self.row("M588#O797")),
        ]
        rows = self.run_case(seq)
        self.assertEqual(
            rows[2]["plates"][0]["track_id"], rows[-1]["plates"][0]["track_id"]
        )
        self.assertNotEqual(rows[-1]["plates"][0]["status"], "CONFIRMED")


if __name__ == "__main__":
    unittest.main()
