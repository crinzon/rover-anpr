"""Fragmentation and separate-identity regressions; no real camera required."""

import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import database
from plate_evidence import reid_match, remember_identity, spatial_match
from vehicle_registry import (
    init_schema,
    link_reviewed,
    record_observation,
    registry_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


class RegistryIdentityTest(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        init_schema(self.connection)
        self.addCleanup(self.connection.close)

    def event(
        self,
        oid,
        track=1,
        camera="Front",
        plate="K477TK797",
        status="CONFIRMED",
    ):
        return dict(
            observation_id=oid,
            track_id=track,
            camera_id=camera,
            plate=plate,
            status=status,
            subtype="type1",
            confidence=0.95,
            quality=0.8,
            captured_at=100.0,
        )

    def add(self, oid, track=1, camera="Front", session="s", **kwargs):
        return record_observation(
            self.connection, session, self.event(oid, track, camera, **kwargs)
        )

    def proof(self, a, b, relation="cross_camera_same_observation"):
        return dict(
            reviewed_same_vehicle=True,
            relation=relation,
            evidence_refs=["source/video#frame=1", "other/video#frame=2"],
            anchor_observation_ids=[a, b],
        )

    def test_same_track_number_different_camera_or_session_is_not_identity(
        self,
    ):
        a = self.add("a")
        b = self.add("b", camera="Back")
        c = self.add("c", session="next_file")
        self.assertEqual(len({a, b, c}), 3)
        rows = registry_snapshot(self.connection)
        self.assertEqual(len({r["vehicle_id"] for r in rows}), 3)

    def test_identical_adjacent_file_stays_candidate_without_review(self):
        self.add("a", session="file_1")
        self.add("b", session="file_2")
        self.assertEqual(len(registry_snapshot(self.connection)), 2)

    def test_reviewed_adjacent_file_continuation_joins_exact_anchors(self):
        a = self.add("a", session="file_1")
        b = self.add("b", session="file_2")
        link_reviewed(
            self.connection,
            a,
            b,
            self.proof("a", "b", "same_passage_continuation"),
        )
        rows = registry_snapshot(self.connection)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["local_tracks"], 2)

    def test_reviewed_front_back_join_without_destroying_observations(self):
        a = self.add("a")
        b = self.add("b", camera="Back")
        link_reviewed(self.connection, a, b, self.proof("a", "b"))
        self.assertEqual(len(registry_snapshot(self.connection)), 1)
        self.assertEqual(
            self.connection.execute(
                "select count(*) from vehicle_observations"
            ).fetchone()[0],
            2,
        )

    def test_matching_number_without_proof_cannot_join(self):
        a = self.add("a")
        b = self.add("b", camera="Back")
        with self.assertRaises(ValueError):
            link_reviewed(
                self.connection,
                a,
                b,
                dict(relation="cross_camera_same_observation"),
            )
        self.assertEqual(len(registry_snapshot(self.connection)), 2)

    def test_reviewed_merge_preserves_previous_vehicle_id_aliases(self):
        a = self.add("a")
        b = self.add("b", camera="Back")
        old_ids = {
            row["vehicle_id"] for row in registry_snapshot(self.connection)
        }
        link_reviewed(self.connection, a, b, self.proof("a", "b"))
        row = registry_snapshot(self.connection)[0]
        self.assertEqual(
            {row["vehicle_id"], *row["vehicle_id_aliases"]}, old_ids
        )

    def test_review_proof_id_cannot_be_reused_for_another_pair(self):
        a = self.add("a")
        b = self.add("b", camera="Back")
        first = dict(self.proof("a", "b"), proof_id="review1")
        link_reviewed(self.connection, a, b, first)
        c = self.add("c", track=2)
        d = self.add("d", track=3)
        with self.assertRaises(ValueError):
            link_reviewed(
                self.connection,
                c,
                d,
                dict(self.proof("c", "d"), proof_id="review1"),
            )
        self.assertEqual(len(registry_snapshot(self.connection)), 3)

    def test_repeat_passage_same_plate_remains_separate(self):
        a = self.add("a")
        b = self.add("b", track=2)
        with self.assertRaises(ValueError):
            link_reviewed(
                self.connection,
                a,
                b,
                self.proof("a", "b", "same_vehicle_new_passage"),
            )
        self.assertEqual(len(registry_snapshot(self.connection)), 2)

    def test_similar_real_plates_and_required_p0_never_merge(self):
        for left, right in [
            ("Y624TK777", "H372PH777"),
            ("K477TK797", "K477TX797"),
        ]:
            a = self.add("a" + left, plate=left)
            b = self.add("b" + right, camera="Back", plate=right)
            with self.assertRaises(ValueError):
                link_reviewed(
                    self.connection, a, b, self.proof("a" + left, "b" + right)
                )

    def test_false_object_cannot_supply_merge_proof_or_vehicle(self):
        event = self.event("false", plate="K477##797", status="PARTIAL")
        event["visual_assessment"] = "FALSE_OBJECT"
        a = record_observation(self.connection, "s", event)
        b = self.add("real")
        with self.assertRaises(ValueError):
            link_reviewed(self.connection, a, b, self.proof("false", "real"))
        self.assertIsNone(
            self.connection.execute(
                "select vehicle_id from vehicle_passages where passage_id=?",
                (a,),
            ).fetchone()[0]
        )

    def test_partial_confirmed_reoccluded_is_one_encounter(self):
        self.add("p", plate="K477##797", status="PARTIAL")
        self.assertIsNone(registry_snapshot(self.connection)[0]["vehicle_id"])
        self.add("full")
        vehicle = registry_snapshot(self.connection)[0]["vehicle_id"]
        self.add("p2", plate="K477##797", status="PARTIAL")
        rows = registry_snapshot(self.connection)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["vehicle_id"], vehicle)
        self.assertEqual(
            self.connection.execute(
                "select count(*) from vehicle_observations"
            ).fetchone()[0],
            3,
        )

    def test_observation_identity_cannot_cross_scope(self):
        self.add("same")
        with self.assertRaises(ValueError):
            self.add("same", camera="Back")

    def test_scoped_camera_persistence_and_legacy_upsert(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(database, "DB_PATH", Path(temp) / "test.db"),
        ):
            database.init_db()
            a = self.event("a", plate="K477##797", status="PARTIAL")
            database.apply_event("s", a)
            database.apply_event("s", self.event("b", camera="Back"))
            database.apply_event("s", self.event("c"))
            self.assertEqual(len(database.get_detections()), 2)
            self.assertEqual(len(database.get_vehicle_registry()), 2)
            self.assertFalse(database.apply_event("s", self.event("c")))
            self.assertEqual(len(database.get_vehicle_observations()), 3)


@unittest.skipUnless(importlib.util.find_spec("torch"), "WSL runtime required")
class FragmentedWorkerTest(unittest.TestCase):
    def setUp(self):
        from test_track_reacquisition import WorkerReacquisitionTest

        self.helper = WorkerReacquisitionTest()

    def test_supported_identity_survives_sparse_occlusion_then_loss(self):
        helper = self.helper
        seq = helper.stable("K477TK797")
        seq += [(i * 0.05, helper.row("K477#####")) for i in range(3, 13)]
        seq += [(0.8, [])] + helper.stable("K477TK797", (1.0, 1.05, 1.1))
        packets = helper.run_case(seq)
        self.assertEqual({e["track_id"] for _, e in helper.events}, {1})
        self.assertEqual(packets[-1]["plates"][0]["status"], "CONFIRMED")

    def test_partial_wrong_layout_over_continuous_plate_is_quarantined(self):
        helper = self.helper
        seq = helper.stable("E131TP797")
        seq += [
            (t, [("##1311##", [80.0, 100.0, 200.0, 150.0], "type2")])
            for t in (0.15, 0.2, 0.25)
        ]
        seq += helper.stable("E131TP797", (0.3, 0.35, 0.4))
        helper.run_case(seq)
        self.assertEqual(
            {event["track_id"] for _, event in helper.events}, {1}
        )

    def test_simultaneous_duplicate_layout_does_not_create_second_record(self):
        helper = self.helper
        seq = helper.stable("E131TP797")
        seq += [
            (
                t,
                [
                    ("E131TP797", [80.0, 100.0, 200.0, 150.0]),
                    ("##1311##", [80.0, 100.0, 200.0, 150.0], "type2"),
                ],
            )
            for t in (0.15, 0.2, 0.25)
        ]
        helper.run_case(seq)
        self.assertEqual(
            {event["track_id"] for _, event in helper.events}, {1}
        )

    def test_other_partial_object_at_different_position_is_not_suppressed(
        self,
    ):
        helper = self.helper
        seq = helper.stable("E131TP797")
        seq += [
            (t, [("##1311##", [260.0, 100.0, 390.0, 150.0], "type2")])
            for t in (0.15, 0.2, 0.25)
        ]
        helper.run_case(seq)
        self.assertEqual(
            len({event["track_id"] for _, event in helper.events}), 2
        )

    def test_expired_anchor_does_not_revive_a_later_passage(self):
        helper = self.helper
        seq = helper.stable("K477TK797")
        seq += [(i * 0.05, helper.row("K477#####")) for i in range(3, 13)]
        seq += [(2.0, [])] + helper.stable("K477TK797", (3.3, 3.35, 3.4))
        packets = helper.run_case(seq)
        self.assertNotEqual(
            packets[2]["plates"][0]["track_id"],
            packets[-1]["plates"][0]["track_id"],
        )


class BoundedAnchorTest(unittest.TestCase):
    def test_scope_and_deadline_are_not_bypassed(self):
        from test_track_reacquisition import candidate, supported_track

        from plate_evidence import resolve

        track = supported_track("K477TK797")
        track["confirmed_number"] = "K477TK797"
        remember_identity(track, resolve(track["history"]), 0.2)
        track["association_evidence"] = []
        track["camera_id"] = "Front"
        track["session_id"] = "s"
        same = candidate("K477TK797")
        same.update(camera_id="Front", session_id="s")
        self.assertIsNotNone(reid_match(same, track, 1.0))
        for camera, session in [("Back", "s"), ("Front", "other")]:
            with self.subTest(camera=camera, session=session):
                different = dict(same, camera_id=camera, session_id=session)
                self.assertIsNone(reid_match(different, track, 1.0))
        track["last_seen"] = 3.0
        self.assertIsNone(reid_match(same, track, 3.3))

    def test_last_measured_box_only_for_consecutive_detection(self):
        track = dict(
            last_seen=0.2,
            motion_seen=0.2,
            motion_frame_id=3,
            normalized_box=[0.4, 0.4, 0.5, 0.5],
            center=(0.45, 0.45),
            motion_history=[
                dict(tracking_time=0.0, center=(0.05, 0.45)),
                dict(tracking_time=0.1, center=(0.25, 0.45)),
                dict(tracking_time=0.2, center=(0.45, 0.45)),
            ],
        )
        candidate = dict(
            frame_id=4,
            center=(0.45, 0.45),
            normalized_box=[0.4, 0.4, 0.5, 0.5],
        )
        self.assertIsNotNone(spatial_match(candidate, track, 0.3))
        candidate["frame_id"] = 5
        self.assertIsNone(spatial_match(candidate, track, 0.3))


if __name__ == "__main__":
    unittest.main()
