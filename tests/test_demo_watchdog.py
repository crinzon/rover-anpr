import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import video_demo as d

try:
    from . import test_demo_sequence as base
except ImportError:
    import test_demo_sequence as base


class WatchdogTests(unittest.TestCase):
    setUp = base.SequenceTests.setUp
    finish = base.SequenceTests.finish

    def test_transient_missing_status_after_two_minutes_is_not_start_failure(
        self,
    ):
        d.sequence_control("start")
        self.finish(frames=1, phase="playing")
        d._sequence_tick()
        back = d._pair_status(d._sequence)["Back"]
        with (
            patch.object(
                d, "_pair_status", return_value={"Front": {}, "Back": back}
            ),
            patch.object(
                d.time,
                "monotonic",
                return_value=d._sequence["pair_started_mono"] + 121,
            ),
        ):
            d._sequence_tick()
            d._sequence_tick()
            self.assertEqual(d._sequence["state"], "playing")
            self.assertEqual(
                d.sequence_snapshot()["current_pair_totals"][
                    "processed_frames"
                ],
                2,
            )

    def test_true_never_started_stream_still_times_out(self):
        d.sequence_control("start")
        with patch.object(
            d.time,
            "monotonic",
            return_value=d._sequence["pair_started_mono"] + 121,
        ):
            d._sequence_tick()
        self.assertEqual(d._sequence["state"], "error")

    def test_readiness_reset_on_new_pair(self):
        d.sequence_control("start")
        self.finish()
        d._sequence_tick()
        self.assertEqual(d._sequence["ready_runs"], {})
        with patch.object(
            d.time,
            "monotonic",
            return_value=d._sequence["pair_started_mono"] + 121,
        ):
            d._sequence_tick()
        self.assertEqual(d._sequence["state"], "error")

    def test_real_worker_error_not_ignored_after_ready(self):
        d.sequence_control("start")
        self.finish(frames=1, phase="playing")
        d._sequence_tick()
        self.finish(frames=1, phase="error")
        d._sequence_tick()
        self.assertEqual(d._sequence["state"], "error")

    def test_retry_preserves_pair_without_counting_failure_twice(
        self,
    ):
        d.sequence_control("start")
        self.finish()
        d._sequence_tick()
        first = d._sequence["completed_pairs"][0]
        self.finish(frames=1, phase="error")
        d._sequence_tick()
        old = d._sequence_view()
        self.assertTrue((d.ROOT / old["report_path"]).exists())
        recovered = d.sequence_control("retry")
        self.assertNotEqual(recovered["sequence_id"], old["sequence_id"])
        self.assertEqual(recovered["pair_index"], 1)
        self.assertEqual(recovered["completed_pairs"][0], first)
        self.assertNotEqual(recovered["run_ids"], old["run_ids"])
        self.assertEqual(recovered["totals"]["processed_frames"], 4)
        self.finish()
        d._sequence_tick()
        self.finish()
        d._sequence_tick()
        self.assertEqual(
            d.sequence_snapshot()["totals"]["processed_frames"], 12
        )
        self.assertEqual(len(d._sequence["failed_attempts"]), 1)

    def test_control_temporary_read_failure_does_not_stop_worker(self):
        replay_class = d.build_replay_class()
        replay = replay_class.__new__(replay_class)
        replay.session_id = "run"
        replay.stop = Mock()
        replay.stop.is_set.return_value = False
        with patch.object(
            d,
            "read_json",
            side_effect=[{}, dict(run_id="run", state="playing")],
        ):
            self.assertEqual(replay.active_control()["state"], "playing")
        replay.stop.set.assert_not_called()

    def test_read_json_retries_sharing_violation(self):
        with (
            patch.object(
                Path,
                "read_text",
                side_effect=[
                    PermissionError("busy"),
                    json.dumps(dict(ok=True)),
                ],
            ),
            patch.object(d.time, "sleep"),
        ):
            self.assertEqual(d.read_json(Path("status.json")), dict(ok=True))


if __name__ == "__main__":
    unittest.main()
