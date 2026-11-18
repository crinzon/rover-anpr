"""Offline CLI contracts without OpenCV, Torch, models or network."""

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from process_video import (
    CFRWriter,
    VideoReplay,
    discover,
    main,
    process_one,
    render_frame,
)


class Frame:
    shape = (100, 200, 3)

    def __init__(self, index):
        self.index = index
        self.annotations = []

    def copy(self):
        return Frame(self.index)


class FakeCV:
    CAP_FFMPEG = 1
    CAP_PROP_POS_MSEC = 2
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self, count=3, pts=None, codec=True):
        self.count, self.pts, self.codec = count, pts, codec
        self.exports = {}

    def VideoCapture(self, path, backend):
        cv = self
        frames = self.exports.get(path, [Frame(i) for i in range(self.count)])

        class Capture:
            position = 0

            def isOpened(self):
                return True

            def read(self):
                if self.position == len(frames):
                    return False, None
                frame = frames[self.position]
                self.position += 1
                return True, frame

            def get(self, prop):
                if cv.pts is not None:
                    return cv.pts[self.position - 1] * 1000
                return (self.position - 1) * 100

            def release(self):
                pass

        return Capture()

    def VideoWriter_fourcc(self, *args):
        return 0

    def VideoWriter(self, path, codec, fps, size):
        cv = self
        cv.exports[path] = []

        class Writer:
            def isOpened(self):
                return cv.codec

            def write(self, frame):
                cv.exports[path].append(frame)
                Path(path).write_bytes(b"fake mp4")

            def release(self):
                pass

        return Writer()

    def rectangle(self, image, start, end, color, thickness):
        image.annotations.append(("box", start, end))

    def putText(self, image, text, *args):
        image.annotations.append(("text", text))


def metadata(count=3, vfr=False):
    return dict(
        frames=count,
        fps=10.0,
        duration_s=count / 10,
        width=200,
        height=100,
        variable_frame_rate=vfr,
    )


def runner(settings, replay, pipeline_override):
    while True:
        ok, frame = replay.read()
        if not ok:
            break
        plates = (
            []
            if frame.index == 1
            else [
                dict(
                    box=[frame.index + 1, 30, 100, 60],
                    text="A123AA77",
                    readable=True,
                    ocr_confidence=0.95,
                    status="confirmed",
                    track_id=1,
                )
            ]
        )
        replay.record_frame(
            replay.frame_count,
            replay.media_time,
            {"timing_ms": {"total": 1}},
            plates,
            [],
            1,
        )
        replay.wait_frame(replay.frame_count)
    replay.release()


class ProcessVideoTest(unittest.TestCase):
    def test_single_file_and_literal_brackets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "12[R][0@0].MP4"
            path.touch()
            root, files = discover(path)
            self.assertEqual(root, path.parent)
            self.assertEqual(files, [path])

    def test_recursive_front_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ("Front/a.mp4", "Back/nested/b.MOV", "Back/ignore.txt")
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            base, files = discover(root)
            self.assertEqual(
                [p.relative_to(base).as_posix() for p in files],
                ["Back/nested/b.MOV", "Front/a.mp4"],
            )

    def test_cli_batch_summary_and_preserved_directories(self):
        original = process_one

        def process(*args):
            return original(*args, camera_runner=runner)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "videos"
            output = Path(tmp) / "results"
            for camera in ("Front", "Back"):
                path = root / camera / "a[0@0].mp4"
                path.parent.mkdir(parents=True)
                path.touch()
            modules = {
                "cv2": FakeCV(),
                "full_stand.pipeline": SimpleNamespace(
                    PlatePipeline=lambda config: None
                ),
            }
            args = [
                "process_video.py",
                "--input",
                str(root),
                "--output",
                str(output),
            ]
            with (
                patch.dict(sys.modules, modules),
                patch.object(sys, "argv", args),
                patch("full_stand.config.StandConfig.validate"),
                patch("process_video.process_one", side_effect=process),
                patch("process_video.mov_timing", return_value=metadata()),
            ):
                self.assertEqual(main(), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary), 2)
            self.assertEqual({r["camera"] for r in summary}, {"Front", "Back"})
            for camera in ("Front", "Back"):
                self.assertTrue(
                    (
                        output / camera / "a[0@0].mp4" / "annotated.mp4"
                    ).is_file()
                )

    def test_literal_exclude_in_probe_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "videos", Path(tmp) / "probe"
            for camera in ("Front", "Back"):
                path = root / camera / "a[0@0].mp4"
                path.parent.mkdir(parents=True)
                path.touch()
            args = [
                "process_video.py",
                "--input",
                str(root),
                "--output",
                str(output),
                "--probe-only",
                "--exclude",
                "Front/a[0@0].mp4",
            ]
            with (
                patch.dict(sys.modules, {"cv2": FakeCV()}),
                patch.object(sys, "argv", args),
                patch("process_video.mov_timing", return_value=metadata()),
            ):
                self.assertEqual(main(), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary), 1)
            self.assertEqual(Path(summary[0]["source"]).parent.name, "Back")

    def test_eof_all_frames_reports_and_exact_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "Front" / "a[0@0].mp4"
            source.parent.mkdir()
            source.write_bytes(b"unchanged")
            cv = FakeCV()
            destination = Path(tmp) / "results" / source.relative_to(Path(tmp))
            with patch("process_video.mov_timing", return_value=metadata()):
                result = process_one(
                    source, destination, "cpu", None, cv, runner
                )
            self.assertEqual(source.read_bytes(), b"unchanged")
            self.assertEqual(result["processed_frames"], 3)
            self.assertEqual(result["detection_count"], 2)
            self.assertIsNone(result["unique_vehicle_count"])
            self.assertEqual(result["camera"], "Front")
            self.assertTrue((destination / "annotated.mp4").is_file())
            report = json.loads((destination / "report.json").read_text())
            self.assertEqual(
                [f["video_time_s"] for f in report["frames"]], [0, 0.1, 0.2]
            )
            with (destination / "detections.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r["frame"] for r in rows], ["1", "3"])
            frames = next(iter(cv.exports.values()))
            self.assertEqual([f.index for f in frames], [0, 1, 2])
            boxes = [
                [a for a in f.annotations if a[0] == "box"] for f in frames
            ]
            self.assertEqual(boxes[0][0][1], (1, 30))
            self.assertEqual(boxes[1], [])
            self.assertEqual(boxes[2][0][1], (3, 30))

    def test_empty_detections(self):
        frame = Frame(7)
        image = render_frame(FakeCV(), frame, [], 12)
        self.assertEqual(frame.annotations, [])
        self.assertEqual(image.index, 7)
        self.assertEqual(len(image.annotations), 1)

    def test_empty_video_detections_export_and_csv_header(self):
        def empty_runner(settings, replay, pipeline_override):
            while replay.read()[0]:
                replay.record_frame(
                    replay.frame_count,
                    replay.media_time,
                    {"timing_ms": {"total": 1}},
                    [],
                    [],
                    1,
                )
                replay.wait_frame(replay.frame_count)
            replay.release()

        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "a.mp4", Path(tmp) / "out"
            source.touch()
            with patch("process_video.mov_timing", return_value=metadata()):
                report = process_one(
                    source, output, "cpu", None, FakeCV(), empty_runner
                )
            self.assertEqual(report["detection_count"], 0)
            self.assertEqual(report["confirmed_numbers"], [])
            self.assertEqual(report["processed_frames"], 3)
            with (output / "detections.csv").open() as file:
                reader = csv.DictReader(file)
                self.assertIn("video_time_s", reader.fieldnames)
                self.assertEqual(list(reader), [])

    def test_silent_encoder_failure_not_published(self):
        class BrokenCV(FakeCV):
            def VideoWriter(self, *args):
                writer = super().VideoWriter(*args)
                writer.write = lambda frame: None
                return writer

        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "a.mp4", Path(tmp) / "out"
            source.touch()
            with patch("process_video.mov_timing", return_value=metadata()):
                with self.assertRaisesRegex(RuntimeError, "verification"):
                    process_one(
                        source, output, "cpu", None, BrokenCV(), runner
                    )
            self.assertFalse((output / "annotated.mp4").exists())

    def test_codec_error_is_failed_and_no_success_mp4(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, destination = Path(tmp) / "a.mp4", Path(tmp) / "out"
            path.touch()
            with patch("process_video.mov_timing", return_value=metadata()):
                with self.assertRaisesRegex(RuntimeError, "encoder"):
                    process_one(
                        path,
                        destination,
                        "cpu",
                        None,
                        FakeCV(codec=False),
                        runner,
                    )
            self.assertFalse((destination / "annotated.mp4").exists())
            report = json.loads((destination / "report.json").read_text())
            self.assertEqual(report["status"], "failed")

    def test_early_decode_end_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, destination = Path(tmp) / "a.mp4", Path(tmp) / "out"
            path.touch()
            with patch("process_video.mov_timing", return_value=metadata(4)):
                with self.assertRaises(RuntimeError):
                    process_one(
                        path, destination, "cpu", None, FakeCV(), runner
                    )
            self.assertFalse((destination / "annotated.partial.mp4").exists())
            self.assertFalse((destination / "annotated.mp4").exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                process_one(
                    Path("input.mp4"), Path(tmp), "cpu", None, FakeCV(), runner
                )

    def test_vfr_resampling_uses_previous_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            cv = FakeCV()
            writer = CFRWriter(Path(tmp) / "out.mp4", metadata(), cv)
            writer.add(Frame(0), 0)
            writer.add(Frame(1), 0.04)
            writer.add(Frame(2), 0.21)
            with patch("process_video.mov_timing", return_value=metadata()):
                writer.finish()
            frames = next(iter(cv.exports.values()))
            self.assertEqual([f.index for f in frames], [0, 1, 1])
            writer.close()

    def test_vfr_invalid_pts_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay = VideoReplay(
                Path("input.mp4"),
                Path(tmp),
                metadata(vfr=True),
                FakeCV(pts=[0, 0, 0.2]),
            )
            self.assertTrue(replay.read()[0])
            with self.assertRaisesRegex(ValueError, "VFR"):
                replay.read()
            replay.release()

    def test_result_frame_identity_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay = VideoReplay(
                Path("input.mp4"), Path(tmp), metadata(), FakeCV()
            )
            replay.read()
            with self.assertRaisesRegex(RuntimeError, "current frame"):
                replay.record_frame(2, 0, {}, [], [], 0)
            replay.release()

    def test_interruption_removes_partial_export(self):
        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as tmp:
            path, destination = Path(tmp) / "a.mp4", Path(tmp) / "out"
            path.touch()
            with patch("process_video.mov_timing", return_value=metadata()):
                with self.assertRaises(KeyboardInterrupt):
                    process_one(
                        path, destination, "cpu", None, FakeCV(), interrupt
                    )
            self.assertFalse((destination / "annotated.mp4").exists())
            report = json.loads((destination / "report.json").read_text())
            self.assertEqual(report["status"], "failed")

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(n)
            for n in ("cv2", "numpy", "torch", "ultralytics")
        ),
        "Camera runtime dependencies unavailable",
    )
    def test_real_worker_and_mp4_codec(self):
        script = Path(__file__).with_name("check_process_video.py")
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
