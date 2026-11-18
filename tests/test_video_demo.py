import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import video_demo as demo


class VideoDemoTests(unittest.TestCase):
    def test_partial_confirmed_repeated_frames_count_one_event(self):
        events = demo.DemoEvents()
        for state in (
            "PARTIAL",
            "PARTIAL",
            "CONFIRMED",
            "CONFIRMED",
            "PARTIAL",
            "CONFIRMED",
        ):
            events.submit(
                "session", dict(track_id=1, status=state, plate="K477TK797")
            )
        self.assertEqual(len(events.events), 1)
        self.assertEqual(len(events.confirmed), 1)

    def test_similar_cars_and_sessions_are_not_text_deduplicated(self):
        events = demo.DemoEvents()
        for session, track, plate in [
            ("a", 1, "Y624TK777"),
            ("a", 2, "H372PH777"),
            ("b", 1, "Y624TK777"),
        ]:
            events.submit(
                session, dict(track_id=track, status="CONFIRMED", plate=plate)
            )
        self.assertEqual(len(events.confirmed), 3)

    def test_start_pause_resume_restart_select_reset_scope(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(demo, "STATE", Path(directory)),
            patch.object(
                demo,
                "videos",
                return_value=[
                    dict(id="0", name="Back/real.mp4"),
                    dict(id="1", name="Front/real.mp4"),
                ],
            ),
            patch.object(demo, "_ensure_worker"),
            patch.object(
                demo, "_processes", {"Front": Mock(poll=lambda: None)}
            ),
        ):
            first = demo.control("start", "0")["control"]["run_id"]
            self.assertEqual(
                demo.control("pause")["control"]["state"], "paused"
            )
            self.assertEqual(
                demo.control("start", "0")["control"]["run_id"], first
            )
            restarted = demo.control("restart", "0")["control"]["run_id"]
            self.assertNotEqual(restarted, first)
            selected = demo.control("select", "1")["control"]
            self.assertNotEqual(selected["run_id"], restarted)
            self.assertEqual(selected["state"], "idle")
            self.assertFalse(demo.snapshot()["status"])

    def test_eof_start_creates_new_run_not_duplicate_events(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(demo, "STATE", Path(directory)),
            patch.object(
                demo, "videos", return_value=[dict(id="0", name="real.mp4")]
            ),
            patch.object(demo, "_ensure_worker"),
            patch.object(
                demo, "_processes", {"Front": Mock(poll=lambda: None)}
            ),
        ):
            first = demo.control("start", "0")["control"]["run_id"]
            demo.atomic_json(
                Path(directory) / "Front" / "status.json",
                dict(run_id=first, phase="ended"),
            )
            self.assertNotEqual(
                demo.control("start", "0")["control"]["run_id"], first
            )
            self.assertFalse(demo.DemoEvents().confirmed)

    def test_arbitrary_path_cannot_be_selected(self):
        with patch.object(
            demo, "videos", return_value=[dict(id="0", name="real.mp4")]
        ):
            with self.assertRaises(ValueError):
                demo.control("start", "../../outside.mp4")

    def test_timestamps_keep_order_without_accelerated_catchup_and_eof(self):
        clock = [0.0]

        class Stop:
            def is_set(self):
                return False

            def wait(self, seconds):
                clock[0] += seconds

        class Base:
            def read(self):
                if not self.points:
                    return False, None
                self.media_time = self.points.pop(0)
                return True, self.media_time

        with (
            patch("process_video.VideoReplay", Base),
            patch("video_demo.time.monotonic", side_effect=lambda: clock[0]),
        ):
            replay_class = demo.build_replay_class()
            replay = replay_class.__new__(replay_class)
            replay.points = [0.0, 0.05, 0.10]
            replay.stop = Stop()
            replay.wait_playing = lambda: True
            replay.anchor = None
            replay.previous_start = None
            replay.previous_start_pts = None
            self.assertEqual(replay.read(), (True, 0.0))
            clock[0] = 0.2  # a slow real inference; next read is already late
            self.assertEqual(replay.read(), (True, 0.05))
            clock[0] = 0.21
            self.assertEqual(replay.read(), (True, 0.10))
            self.assertAlmostEqual(clock[0], 0.25)  # no 100 FPS catch-up burst
            self.assertEqual(replay.read(), (False, None))

    def test_atomic_command_retries_windows_sharing_violation(self):
        original = demo.os.replace
        attempts = []

        def replace(source, target):
            attempts.append(1)
            if len(attempts) < 3:
                raise PermissionError("sharing violation")
            original(source, target)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("video_demo.os.replace", side_effect=replace),
            patch("video_demo.time.sleep"),
        ):
            path = Path(directory) / "control.json"
            demo.atomic_json(path, dict(state="paused"))
            self.assertEqual(demo.read_json(path)["state"], "paused")
            self.assertEqual(len(attempts), 3)
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_front_back_pause_restart_are_independent(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(demo, "STATE", Path(directory)),
            patch.object(
                demo,
                "videos",
                return_value=[
                    dict(id="0", name="Back/real.mp4"),
                    dict(id="1", name="Front/real.mp4"),
                ],
            ),
            patch.object(demo, "_ensure_worker"),
            patch.object(
                demo,
                "_processes",
                {
                    "Front": Mock(poll=lambda: None),
                    "Back": Mock(poll=lambda: None),
                },
            ),
        ):
            front = demo.control("start", "1", "Front")["control"]["run_id"]
            back = demo.control("start", "0", "Back")["control"]["run_id"]
            self.assertNotEqual(front, back)
            demo.control("pause", channel="Front")
            self.assertEqual(
                demo.snapshot("Back")["control"]["state"], "playing"
            )
            demo.control("restart", "1", "Front")
            self.assertEqual(demo.snapshot("Back")["control"]["run_id"], back)


if __name__ == "__main__":
    unittest.main()
