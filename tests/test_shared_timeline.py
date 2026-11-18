import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import video_demo as demo


class SharedTimelineTests(unittest.TestCase):
    def test_start_both_shared_group_pause_resume_and_restart_reset(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(demo, "STATE", Path(d)),
            patch.object(
                demo,
                "videos",
                return_value=[
                    dict(id="0", name="Back/v.mp4"),
                    dict(id="1", name="Front/v.mp4"),
                ],
            ),
            patch.object(demo, "_ensure_worker"),
        ):
            first = demo.control_both("start", {"Front": "1", "Back": "0"})
            controls = {c: first[c]["control"] for c in demo.CHANNELS}
            self.assertEqual(
                controls["Front"]["sync_group"], controls["Back"]["sync_group"]
            )
            self.assertNotEqual(
                controls["Front"]["run_id"], controls["Back"]["run_id"]
            )
            demo.control_both("pause")
            resumed = demo.control_both("start")
            self.assertEqual(
                resumed["Front"]["control"]["run_id"],
                controls["Front"]["run_id"],
            )
            restarted = demo.control_both("restart")
            self.assertNotEqual(
                restarted["Front"]["control"]["run_id"],
                controls["Front"]["run_id"],
            )
            self.assertNotEqual(
                restarted["Back"]["control"]["run_id"],
                controls["Back"]["run_id"],
            )
            demo.control("pause", channel="Front")
            self.assertNotIn("sync_group", demo.snapshot("Front")["control"])
            self.assertEqual(
                demo.snapshot("Back")["control"]["state"], "playing"
            )

    def make_replay(self):
        replay_class = demo.build_replay_class()
        replay = replay_class.__new__(replay_class)
        replay.stop = Mock()
        replay.stop.is_set.return_value = False
        replay.wait_playing = lambda: True
        replay.source_fps = 20.0
        replay.initialized_mono = 10.0
        replay.sync_wait_ms = 0.0
        replay.media_time = 0.15
        replay.previous_start_pts = 0.10
        return replay

    def test_common_epoch_uses_ready_time_of_both_not_start_button_timing(
        self,
    ):
        a = self.make_replay()
        b = self.make_replay()
        b.initialized_mono = 12.0
        a.sync_peer = lambda: dict(initialized_mono=12.0)
        b.sync_peer = lambda: dict(initialized_mono=10.0)
        self.assertTrue(a.wait_sync(startup=True))
        self.assertTrue(b.wait_sync(startup=True))
        self.assertEqual(a.anchor, b.anchor)
        self.assertEqual(a.anchor, 12.0)

    def test_media_pts_gate_waits_for_peer_even_with_unrelated_frame_counts(
        self,
    ):
        r = self.make_replay()
        r.sync_peer = Mock(
            side_effect=[
                dict(
                    initialized_mono=12.0,
                    completed_media_time_s=0.03,
                    source_fps=30.0,
                    frame_id=999,
                ),
                dict(
                    initialized_mono=12.0,
                    completed_media_time_s=0.10,
                    source_fps=30.0,
                    frame_id=1000,
                ),
            ]
        )
        self.assertTrue(r.wait_sync())
        r.stop.wait.assert_called_once_with(0.01)

    def test_peer_eof_does_not_stop_remaining_video(self):
        r = self.make_replay()
        r.media_time = 120.0
        r.sync_peer = lambda: dict(
            initialized_mono=12.0, completed_media_time_s=90.0, phase="ended"
        )
        self.assertTrue(r.wait_sync())
        r.stop.wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
