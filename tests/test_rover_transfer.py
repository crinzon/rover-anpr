import unittest
from pathlib import Path
from unittest.mock import patch

from adapters.ros2_humble import Bridge, message, poll_api
from combine_six_videos import episodes

ROOT = Path(__file__).resolve().parents[1]


class ConsolidationTest(unittest.TestCase):
    def row(self, oid, track, text, status="CONFIRMED", video="Front/a.mp4"):
        return dict(
            observation_id=str(oid),
            frame=str(oid),
            track_id=str(track),
            text=text,
            status=status,
            video=video,
        )

    def test_same_text_not_car_identity(self):
        rows = [
            self.row(1, 1, "K477TK797"),
            self.row(2, 2, "K477TK797"),
            self.row(3, 1, "K477TK797", video="Back/a.mp4"),
        ]
        self.assertEqual(len(episodes(rows)), 3)

    def test_old_p0_track_with_two_cars_is_partitioned(self):
        rows = [self.row(1, 6, "Y624TK777"), self.row(2, 6, "H372PH777")]
        self.assertEqual(len(episodes(rows)), 2)

    def test_partial_to_confirmed_same_local_episode(self):
        rows = [
            self.row(1, 1, "K477##797", "PARTIAL"),
            self.row(2, 1, "K477TK797"),
        ]
        parts = episodes(rows)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0][0]["text"], "K477##797")

    def test_partial_different_tracks_not_wildcard_merge(self):
        self.assertEqual(
            len(
                episodes(
                    [
                        self.row(1, 1, "K477##797", "PARTIAL"),
                        self.row(2, 2, "K477TK797"),
                    ]
                )
            ),
            2,
        )

    def test_purge_preserves_user_data(self):
        make = (ROOT / "deploy/rover/Makefile").read_text(encoding="utf-8")
        purge = make.split("purge:\n")[1].split("\n\n")[0]
        self.assertIn("down --remove-orphans", purge)
        for forbidden in ["--volumes", "--rmi", "prune", "rm -rf", "unlink"]:
            self.assertNotIn(forbidden, purge)


class RosAdapterTest(unittest.TestCase):
    def event(self):
        return dict(
            plate="K477##797",
            status="PARTIAL",
            track_id=3,
            captured_at=1234.0,
            event_id="evt",
            confidence=0.71,
            position=dict(x=0.1, y=-0.2),
        )

    def test_actual_fields_and_unknowns_preserved(self):
        event = self.event()
        result = message("session", event, "Front")
        self.assertEqual(result["event"], event)
        self.assertEqual(result["camera_id"], "Front")
        self.assertEqual(result["event"]["plate"], "K477##797")
        self.assertEqual(result["event"]["position"]["y"], -0.2)
        self.assertIsNone(message("session", event)["camera_id"])

    def test_retry_deduplicates_but_sessions_independent(self):
        rows = []
        bridge = Bridge(rows.append)
        self.assertTrue(bridge.accept("a", self.event()))
        self.assertFalse(bridge.accept("a", self.event()))
        self.assertTrue(bridge.accept("b", self.event()))
        self.assertEqual(len(rows), 2)

    def test_failed_publish_can_retry(self):
        failures = [True]

        def publish(row):
            if failures.pop():
                raise OSError("not published")

        bridge = Bridge(publish)
        with self.assertRaises(OSError):
            bridge.accept("a", self.event())
        self.assertEqual(len(bridge.seen), 0)

    def test_no_missing_fields_fabricated(self):
        result = message("s", dict(track_id=1, plate="#", status="PARTIAL"))
        self.assertNotIn("confidence", result["event"])
        self.assertNotIn("position", result["event"])
        self.assertNotIn("captured_at", result["event"])

    def test_invalid_time_rejected(self):
        event = self.event()
        event["captured_at"] = float("nan")
        with self.assertRaises(ValueError):
            message("s", event)

    def test_transport_loss_then_recovery(self):
        rows = []
        bridge = Bridge(rows.append)
        with patch(
            "adapters.ros2_humble.urlopen", side_effect=OSError("disconnected")
        ):
            with self.assertRaises(OSError):
                poll_api(bridge, "http://localhost/plates")
        self.assertFalse(bridge.seen)
        bridge.accept("s", self.event())
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
