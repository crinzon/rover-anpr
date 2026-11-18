import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import video_demo as d


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.choices = [
            dict(id=str(i * 2 + j), name=f"{c}/{slot}[R]_ch1.mp4")
            for i, slot in enumerate(
                ("12.30.00-12.33.00", "12.33.00-12.36.00", "12.36.00-12.39.00")
            )
            for j, c in enumerate(d.CHANNELS)
        ]
        for p in (
            patch.object(d, "STATE", self.root / "state"),
            patch.object(d, "ROOT", self.root),
            patch.object(d, "videos", return_value=self.choices),
            patch.object(d, "_sequence", None),
            patch.object(d, "_sequence_thread", None),
            patch.object(d, "_ensure_worker"),
            patch.object(d.threading.Thread, "start"),
        ):
            p.start()
            self.addCleanup(p.stop)

    def finish(self, cameras=d.CHANNELS, frames=2, phase="ended"):
        for c in cameras:
            run = d._sequence["run_ids"][c]
            s = dict(
                run_id=run,
                phase=phase,
                source_frames=2,
                processed_frames=frames,
                skipped_processing_frames=0,
                detection_count=3,
                ocr_readings=2,
                unique_events=1,
                unique_confirmed_events=1,
                updated_at=d.time.time(),
            )
            d.atomic_json(d.STATE / c / "status.json", s)
            d.atomic_json(
                d.STATE / c / "runs" / (run + ".json"), dict(state=s)
            )

    def test_all_three_pairs_exactly_once_totals_and_final_report(self):
        d.sequence_control("start")
        groups = []
        runs = []
        for i in range(3):
            self.assertEqual(d.sequence_snapshot()["pair_number"], i + 1)
            a, b = (d.snapshot(c)["control"] for c in d.CHANNELS)
            self.assertEqual(a["sync_group"], b["sync_group"])
            groups.append(a["sync_group"])
            runs.extend([a["run_id"], b["run_id"]])
            self.finish()
            d._sequence_tick()
        view = d.sequence_snapshot()
        self.assertEqual(view["state"], "completed")
        self.assertEqual(view["progress_percent"], 100)
        self.assertEqual(view["totals"]["processed_frames"], 12)
        self.assertEqual(view["totals"]["unique_confirmed_events"], 6)
        self.assertEqual(len(set(groups)), 3)
        self.assertEqual(len(set(runs)), 6)
        d._sequence_tick()
        d._sequence_tick()
        self.assertEqual(len(d._sequence["completed_pairs"]), 3)
        self.assertTrue((self.root / view["report_path"]).is_file())

    def test_pause_at_eof_blocks_transition_until_resume_without_restart(self):
        d.sequence_control("start")
        before = dict(d._sequence["run_ids"])
        d.sequence_control("pause")
        self.finish()
        d._sequence_tick()
        self.assertEqual(d._sequence["run_ids"], before)
        self.assertEqual(d._sequence["pair_index"], 0)
        d.sequence_control("resume")
        self.assertEqual(d._sequence["pair_index"], 1)
        self.assertEqual(len(d._sequence["completed_pairs"]), 1)

    def test_resume_paused_processing_keeps_ids(self):
        d.sequence_control("start")
        before = dict(d._sequence["run_ids"])
        d.sequence_control("pause")
        d.sequence_control("resume")
        self.assertEqual(d._sequence["run_ids"], before)

    def test_stop_cancels_even_when_eof_arrives_later(self):
        d.sequence_control("start")
        before = dict(d._sequence["run_ids"])
        d.sequence_control("stop")
        self.finish()
        d._sequence_tick()
        self.assertEqual(d._sequence["run_ids"], before)
        self.assertEqual(d._sequence["state"], "stopped")
        self.assertTrue(
            all(
                d.snapshot(c)["control"]["state"] == "idle" for c in d.CHANNELS
            )
        )

    def test_one_eof_and_stale_eof_do_not_advance(self):
        d.sequence_control("start")
        self.finish(("Front",))
        d._sequence_tick()
        self.assertEqual(d._sequence["pair_index"], 0)
        self.finish(("Back",))
        d._sequence_tick()
        self.assertEqual(d._sequence["pair_index"], 1)
        d._sequence_tick()
        self.assertEqual(d._sequence["pair_index"], 1)

    def test_archives_must_be_saved_before_switching(self):
        d.sequence_control("start")
        self.finish()
        path = (
            d.STATE
            / "Back"
            / "runs"
            / (d._sequence["run_ids"]["Back"] + ".json")
        )
        path.unlink()
        d._sequence_tick()
        self.assertEqual(d._sequence["pair_index"], 0)
        d.atomic_json(path, {})
        d._sequence_tick()
        self.assertEqual(d._sequence["pair_index"], 1)

    def test_double_click_does_not_launch_again(self):
        first = d.sequence_control("start")
        second = d.sequence_control("start")
        self.assertEqual(first["sequence_id"], second["sequence_id"])
        self.assertEqual(first["run_ids"], second["run_ids"])

    def test_incomplete_or_failed_pair_never_advances(self):
        for phase, frames in [("ended", 1), ("error", 2)]:
            d.sequence_control("start")
            self.finish(phase=phase, frames=frames)
            d._sequence_tick()
            self.assertEqual(d._sequence["state"], "error")
            self.assertEqual(d._sequence["pair_index"], 0)

    def test_manual_selection_cancels_queue(self):
        d.sequence_control("start")
        d.control("select", "2", "Front")
        self.finish()
        d._sequence_tick()
        self.assertEqual(d._sequence["state"], "stopped")
        self.assertEqual(d._sequence["pair_index"], 0)

    def test_stop_statistics_survive_later_manual_run(self):
        d.sequence_control("start")
        self.finish(frames=1, phase="playing")
        d.sequence_control("stop")
        total = d.sequence_snapshot()["totals"]["processed_frames"]
        d.control("restart", "0", "Front")
        self.assertEqual(
            d.sequence_snapshot()["totals"]["processed_frames"], total
        )

    def test_excel_is_preserved(self):
        excel = self.root / "results/anpr_final/unique_plates.xlsx"
        excel.parent.mkdir(parents=True)
        excel.write_bytes(b"existing approved export")
        d.sequence_control("start")
        for _ in range(3):
            self.finish()
            d._sequence_tick()
        self.assertEqual(excel.read_bytes(), b"existing approved export")


if __name__ == "__main__":
    unittest.main()
