"""Sequential MP4/MOV processing using the production camera worker.

Offline by default; --live and --api-url opt into existing Preview/API.
"""

import argparse
import contextlib
import csv
import json
import math
import os
import signal
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from evaluate_videos import FileReplay, mov_timing


class LocalSender:
    """Offline sink; the replay callback records confirmations."""

    def __init__(self, *args, **kwargs):
        pass

    def submit(self, session_id, plate):
        return True

    def close(self):
        pass


def discover(source):
    source = Path(source).resolve()
    if source.is_file() and source.suffix.lower() in (".mp4", ".mov"):
        return source.parent, [source]
    if source.is_dir():
        paths = sorted(
            p
            for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in (".mp4", ".mov")
        )
        if paths:
            return source, paths
    raise ValueError(f"No MP4/MOV files found: {source}")


def probe_video(path, cv):
    """Use container timing and dimensions from a decoded frame."""
    metadata = mov_timing(path)
    capture = cv.VideoCapture(str(path), cv.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot decode first frame: {path}")
        height, width = frame.shape[:2]
        metadata.update(width=width, height=height, first_frame_decodable=True)
    finally:
        capture.release()
    if not (
        math.isfinite(metadata["fps"])
        and metadata["fps"] > 0
        and metadata["duration_s"] > 0
        and metadata["frames"] > 0
    ):
        raise ValueError("Invalid container timing")
    return metadata


def render_frame(cv, frame, plates, fps, capture_fps=None):
    """Only results belonging to this input frame are accepted here."""
    image = frame.copy()
    height, width = image.shape[:2]
    for plate in plates:
        x1, y1, x2, y2 = map(int, plate["box"])
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        text = plate.get("merged_text", plate.get("text")) or ""
        state = plate.get("recognition_status") or (
            "CONFIRMED"
            if plate.get("status") == "confirmed"
            else "PENDING"
            if plate.get("track_id") is not None
            else "UNREADABLE"
            if not text or not plate.get("readable")
            else "UNTRACKED"
        )
        color = (60, 220, 90) if state == "CONFIRMED" else (0, 190, 255)
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{text or 'plate'} | {state}"
        confidence = plate.get("ocr_confidence")
        if confidence is not None:
            label += f" {confidence:.0%}"
        if plate.get("track_id") is not None:
            label += f" ID:{plate['track_id']}"
        # Put labels outside the box where there is room; clamp to image.
        label_y = y1 - 8 if y1 >= 28 else min(height - 8, y2 + 22)
        cv.putText(
            image,
            label,
            (x1, max(16, label_y)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    cv.putText(
        image,
        f"ANPR: {fps:.2f} FPS | Capture: "
        + (f"{capture_fps:.2f} FPS" if capture_fps is not None else "n/a"),
        (12, 24),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return image


class CFRWriter:
    """Sample annotated frames on a CFR grid, preserving elapsed source time.

    Hold the preceding frame until the next source PTS. Some source frames
    may be duplicated or omitted in the export, but all undergo inference.
    Output duration differs by at most half an output frame. Audio is omitted.
    """

    def __init__(self, path, metadata, cv):
        self.cv, self.path, self.metadata = cv, path, metadata
        self.fps = metadata["fps"]
        self.count = 0
        self.previous = None
        self.previous_pts = None
        self.writer = cv.VideoWriter(
            str(path),
            cv.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (metadata["width"], metadata["height"]),
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError("MP4 mp4v encoder unavailable")

    def add(self, frame, pts):
        if not math.isfinite(pts) or pts < 0:
            raise ValueError("Invalid video timestamp")
        if self.previous_pts is not None and pts <= self.previous_pts:
            raise ValueError("Non-increasing video timestamps")
        if pts >= self.metadata["duration_s"]:
            raise ValueError("Video timestamp exceeds container duration")
        if self.previous is not None:
            self._until(pts)
        self.previous, self.previous_pts = frame, pts

    def _until(self, end, limit=None):
        while self.count / self.fps < end - 1e-9:
            if limit is not None and self.count >= limit:
                break
            self.writer.write(self.previous)
            self.count += 1

    def finish(self):
        if self.previous is None:
            raise RuntimeError("No frames to export")
        target = round(self.metadata["duration_s"] * self.fps)
        self._until(self.metadata["duration_s"], target)
        self.writer.release()
        if self.count != target:
            raise RuntimeError("Incorrect export frame count")
        # OpenCV write() has no reliable success return. Decode the entire
        # export before publishing it as successful, including silent failures.
        capture = self.cv.VideoCapture(str(self.path), self.cv.CAP_FFMPEG)
        decoded = 0
        try:
            if not capture.isOpened():
                raise RuntimeError("Export cannot be opened")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None or frame.shape[:2] != (
                    self.metadata["height"],
                    self.metadata["width"],
                ):
                    raise RuntimeError("Invalid export dimensions")
                decoded += 1
        finally:
            capture.release()
        if decoded != target:
            raise RuntimeError(
                f"Export verification failed: {decoded}/{target}"
            )
        exported = mov_timing(self.path)
        difference = abs(exported["duration_s"] - self.metadata["duration_s"])
        if difference > 0.5 / self.fps + 0.002:
            raise RuntimeError("Export duration mismatch")
        return exported

    def close(self):
        self.writer.release()


class VideoReplay(FileReplay):
    """Reuse FileReplay's sequential handshake with camera.main unchanged."""

    sender_factory = LocalSender

    def __init__(
        self,
        path,
        directory,
        metadata,
        cv,
        live=False,
        api_url=None,
        preview_path=None,
    ):
        self.cv2, self.path, self.directory = cv, path, directory
        self.metadata = metadata
        self.camera_id = next(
            (part for part in path.parts if part in ("Front", "Back")),
            "file_unspecified",
        )
        self.capture = cv.VideoCapture(str(path), cv.CAP_FFMPEG)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open video: {path}")
        self.capture_dir = directory / "captures"
        self.image_prefix = "captures"
        self.api_url = api_url
        self.delivery = dict(
            enabled=bool(api_url),
            submitted=0,
            queued=0,
            acknowledged=0,
            failed_attempts=0,
            rejected=0,
            pending_at_shutdown=0,
        )
        if api_url:
            from event_sender import EventSender

            self.sender_factory = EventSender
        self.session_id = uuid.uuid4().hex
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.last_completed = self.frame_count = 0
        self.worker_error = False
        self.media_time = self.last_frame_timestamp = 0.0
        self.capture_read_ms = 0.0
        self.source_fps = metadata["fps"]
        self.capture_fps, self.dropped_frames = None, 0
        self.capture_times = deque(maxlen=30)
        self.anpr_fps = 0.0
        from live_preview import RunStatistics

        self.statistics = RunStatistics()
        self.first_decoded = None
        self.preview = SimpleNamespace(
            submit=lambda *a: None,
            update_results=lambda *a: None,
            set_inference_error=lambda *a: None,
            close=lambda: None,
        )
        self.publisher = None
        if live:
            from live_preview import DEFAULT_PATH, PreviewPublisher

            self.publisher = PreviewPublisher(
                {"STATS_PATH": str(directory / "live_stats.json")},
                path=preview_path or DEFAULT_PATH,
            )
            self.preview = self.publisher
            self.capture_dir = (
                Path(__file__).resolve().parent
                / "captures"
                / "video-runs"
                / self.session_id
            )
            self.image_prefix = "/captures/video-runs/" + self.session_id
        self._publish_results = self.preview.update_results
        self.preview.update_results = self._update_results
        self.diagnostics = SimpleNamespace(
            due=lambda: False, finish=lambda *a: None, error=lambda *a: None
        )
        self.errors, self.frames, self.detections = [], [], []
        self.confirmed, self.thresholds = [], {}
        self.started = time.monotonic()
        self.current = None
        self.previous_pts = None
        self.pts_origin = None
        self.exporter = None

    def _update_results(self, *args, **kwargs):
        self.anpr_fps = args[1]
        self._publish_results(*args, **kwargs)

    def read(self):
        # Same one-frame-at-a-time mechanism as FileReplay, with strict PTS
        # validation: VFR must never silently fall back to frame_index / fps.
        started = time.perf_counter()
        ok, frame = self.capture.read()
        self.capture_read_ms = (time.perf_counter() - started) * 1000
        self.last_frame_timestamp = time.monotonic()
        if not ok:
            return False, None
        if frame is None:
            raise RuntimeError("Decoder returned an empty frame")
        pts = self.capture.get(self.cv2.CAP_PROP_POS_MSEC) / 1000
        if not math.isfinite(pts) or pts < 0:
            raise ValueError("Decoder did not provide valid PTS")
        if self.pts_origin is None:
            self.pts_origin = pts
            self.metadata["decoder_pts_origin_s"] = pts
        pts -= self.pts_origin
        if self.previous_pts is not None and pts <= self.previous_pts:
            if self.metadata["variable_frame_rate"]:
                raise ValueError("VFR requires increasing decoder PTS")
            pts = self.frame_count / self.source_fps
        self.previous_pts = self.media_time = pts
        self.frame_count += 1
        self.current = frame
        self.capture_times.append(self.last_frame_timestamp)
        if len(self.capture_times) > 1:
            self.capture_fps = (len(self.capture_times) - 1) / (
                self.capture_times[-1] - self.capture_times[0]
            )
        self.statistics.capture_fps = self.capture_fps
        self.statistics.dropped_capture = self.dropped_frames
        return True, frame

    def attach_sender(self, sender):
        if not self.api_url:
            return
        transport = sender.transport

        def deliver(payload):
            try:
                transport(payload)
                self.delivery["acknowledged"] += 1
            except Exception:
                self.delivery["failed_attempts"] += 1
                raise

        sender.transport = deliver

    def record_submission(self, plate, queued):
        if self.api_url:
            self.delivery["submitted"] += 1
            self.delivery["queued" if queued else "rejected"] += 1

    def drain_sender(self, sender):
        if not self.api_url:
            return
        deadline = time.monotonic() + 5
        while sender.queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        self.delivery["pending_at_shutdown"] = sender.queue.unfinished_tasks

    def record_error(self, frame_id, media_time, code):
        self.errors.append(
            dict(frame=frame_id, video_time_s=media_time, code=code)
        )
        self.fail_worker()

    def record_frame(
        self, frame_id, media_time, output, plates, confirmed, total_ms
    ):
        if frame_id != self.frame_count or media_time != self.media_time:
            raise RuntimeError("Results do not match current frame")
        rows = [
            dict((k, v) for k, v in p.items() if k != "patch_png")
            for p in plates
        ]
        for row in rows:
            row.update(frame=frame_id, video_time_s=media_time)
        self.detections.extend(rows)
        self.confirmed.extend(
            dict(p, frame=frame_id, video_time_s=media_time) for p in confirmed
        )
        self.frames.append(
            dict(
                frame=frame_id,
                video_time_s=media_time,
                detections=rows,
                confirmed=confirmed,
                timing_ms=output["timing_ms"],
                worker_timing_ms=output.get("worker_timing_ms", {}),
            )
        )
        self.statistics.add(
            dict(
                output.get("worker_timing_ms", output["timing_ms"]),
                capture_read=self.capture_read_ms,
            )
        )
        image = render_frame(
            self.cv2, self.current, rows, self.anpr_fps, self.capture_fps
        )
        self.exporter.add(image, media_time)
        with self.condition:
            self.last_completed = frame_id
            self.condition.notify_all()


def write_reports(directory, report, detections):
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    fields = [
        "camera_id",
        "observation_id",
        "frame",
        "video_time_s",
        "text",
        "ocr_text",
        "raw_text",
        "subtype",
        "confidence",
        "recognition_status",
        "ocr_confidence",
        "detection_confidence",
        "status",
        "track_id",
        "confirmed_number",
        "box",
    ]
    csv_path = directory / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in detections:
            state = (
                row.get("recognition_status") or row.get("status", "").upper()
            )
            writer.writerow(
                dict(
                    row,
                    text=row.get("merged_text", row.get("text", "")),
                    ocr_text=row.get("text", ""),
                    status=state,
                    recognition_status=state,
                    confidence=row.get("ocr_confidence"),
                    box=json.dumps(row.get("box")),
                )
            )


def process_one(
    path,
    directory,
    device,
    pipeline,
    cv,
    camera_runner=None,
    live=False,
    api_url=None,
    preview_path=None,
):
    if camera_runner is None:
        from camera import main as camera_runner
    directory.mkdir(parents=True, exist_ok=False)
    replay = exporter = None
    started = time.monotonic()
    temporary = directory / "annotated.partial.mp4"
    try:
        metadata = probe_video(path, cv)
        replay = VideoReplay(
            path,
            directory,
            metadata,
            cv,
            live=live,
            api_url=api_url,
            preview_path=preview_path,
        )
        exporter = replay.exporter = CFRWriter(temporary, metadata, cv)
        handlers = {
            s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            log_path = directory / "processing.log"
            with log_path.open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log):
                    camera_runner(
                        dict(
                            VIDEO_SOURCE=str(path),
                            DEVICE=device,
                            SHOW_WINDOW="false",
                            API_URL=api_url or "",
                        ),
                        replay=replay,
                        pipeline_override=pipeline,
                    )
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
        if getattr(replay, "interrupted", False):
            raise KeyboardInterrupt("Video processing interrupted")
        if replay.stop.is_set() and replay.frame_count < metadata["frames"]:
            raise RuntimeError("Processing interrupted before EOF")
        if (
            replay.errors
            or replay.worker_error
            or len(replay.frames) != metadata["frames"]
        ):
            raise RuntimeError("Incomplete decoding or recognition")
        exported = exporter.finish()
        elapsed = time.monotonic() - started
        report = dict(
            status="done",
            source=str(path.resolve()),
            camera=next(
                (p for p in path.parts if p in ("Front", "Back")), None
            ),
            source_metadata=metadata,
            processed_frames=len(replay.frames),
            detection_count=len(replay.detections),
            unique_vehicle_count=None,
            confirmed_numbers=sorted({p["plate"] for p in replay.confirmed}),
            confirmations=replay.confirmed,
            frames=replay.frames,
            processing_fps=len(replay.frames) / elapsed,
            elapsed_seconds=elapsed,
            output_metadata=exported,
            statistics=replay.statistics.summary(),
            api_delivery=dict(
                replay.delivery,
                complete=(
                    not replay.api_url
                    or replay.delivery["acknowledged"]
                    == replay.delivery["submitted"]
                ),
            ),
            dropped_capture=replay.dropped_frames,
            dropped_processing=max(0, replay.frame_count - len(replay.frames)),
            dropped_preview=replay.publisher.dropped
            if replay.publisher
            else None,
            metrics_note="source_metadata.fps is playback FPS;"
            " capture_fps is actual sequential re"
            "ad cadence; "
            "processing_fps includes export/verif"
            "ication; Browser FPS is measured onl"
            "y in browser.",
            output="annotated.mp4",
            conversion=("CFR hold preceding annotated source frame; no audio"),
        )
        write_reports(directory, report, replay.detections)
        (directory / "statistics.json").write_text(
            json.dumps(
                dict(
                    elapsed_seconds=elapsed,
                    processing_fps=report["processing_fps"],
                    **{
                        k: report[k]
                        for k in (
                            "statistics",
                            "dropped_capture",
                            "dropped_processing",
                            "dropped_preview",
                            "api_delivery",
                            "metrics_note",
                        )
                    },
                ),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, directory / "annotated.mp4")
        return {
            k: v
            for k, v in report.items()
            if k not in ("frames", "confirmations")
        }
    except BaseException as error:
        if exporter is not None:
            exporter.close()
        temporary.unlink(missing_ok=True)
        # Failed/interrupted runs must never be published as done.
        report = dict(
            status="failed",
            source=str(path),
            error=str(error) or type(error).__name__,
            processed_frames=len(replay.frames) if replay else 0,
            elapsed_seconds=time.monotonic() - started,
            statistics=replay.statistics.summary() if replay else None,
            api_delivery=replay.delivery if replay else None,
        )
        write_reports(directory, report, replay.detections if replay else [])
        raise
    finally:
        if replay is not None:
            replay.release()
            if replay.publisher is not None:
                replay.publisher.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--live", action="store_true", help="Publish to existing /live"
    )
    parser.add_argument(
        "--api-url",
        help="Existing API /plates URL; opts into event delivery and Live",
    )
    parser.add_argument(
        "--preview-path",
        type=Path,
        help="Shared Preview file path (API must read same path)",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="Literal path relative to input; repeatable",
    )
    args = parser.parse_args()
    try:
        root, paths = discover(args.input)
    except ValueError as error:
        parser.error(str(error))
    paths = [p for p in paths if p.relative_to(root) not in args.exclude]
    if not paths:
        parser.error("No videos remain after --exclude")
    output = args.output.resolve()
    if args.input.is_dir() and (output == root or root in output.parents):
        parser.error("--output must be outside the input directory")
    # Use a separate directory per source including its extension to avoid
    # collisions between foo.mp4 and foo.mov. Never overwrite existing results.
    targets = [output / p.relative_to(root) for p in paths]
    if any(p.exists() for p in targets):
        parser.error("Output already exists; choose a new output directory")
    try:
        import cv2

        if not args.probe_only:
            from full_stand.config import StandConfig

            config = StandConfig(device=args.device, confidence=0.5)
            config.validate()
            from full_stand.pipeline import PlatePipeline

            pipeline = PlatePipeline(config)
        output.mkdir(parents=True, exist_ok=True)
        summary = []
        failed = False
        for path, target in zip(paths, targets):
            print(f"Processing: {path}", flush=True)
            interrupted = False
            try:
                if args.probe_only:
                    row = dict(
                        source=str(path),
                        status="probed",
                        metadata=probe_video(path, cv2),
                    )
                else:
                    row = process_one(
                        path,
                        target,
                        args.device,
                        pipeline,
                        cv2,
                        **(
                            dict(
                                live=True,
                                api_url=args.api_url,
                                preview_path=args.preview_path,
                            )
                            if args.live or args.api_url
                            else {}
                        ),
                    )
            except KeyboardInterrupt:
                row = dict(source=str(path), status="interrupted")
                failed = interrupted = True
            except Exception as error:
                row = dict(source=str(path), status="failed", error=str(error))
                failed = True
            summary.append(row)
            (output / "summary.json").write_text(
                json.dumps(
                    summary, ensure_ascii=False, indent=2, allow_nan=False
                ),
                encoding="utf-8",
            )
            print(f"{row['status']}: {path}", flush=True)
            if interrupted:
                return 130
        return 1 if failed else 0
    except ImportError as error:
        parser.exit(2, f"Missing dependency: {error}. Nothing installed.\n")
    except (FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
