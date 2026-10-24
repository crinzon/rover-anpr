"""Evaluate files through camera.main's actual worker, tracker and EventSender.

No separate recognition/tracking implementation. Requires the Camera runtime.
"""

import argparse
import base64
import contextlib
import csv
import hashlib
import json
import math
import statistics
import struct
import threading
import time
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

VIDEO_NAMES = [
    "IMG_4207.MOV",
    "IMG_4209.MOV",
    "IMG_2901.MOV",
    "IMG_2905.MOV",
    "IMG_2910.MOV",
]


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def mov_timing(path):
    """Read original MOV sample timings instead of estimated durations."""
    data = path.read_bytes()

    def boxes(start, end):
        while start + 8 <= end:
            size, kind = struct.unpack_from(">I4s", data, start)
            header = 8
            if size == 1:
                size = struct.unpack_from(">Q", data, start + 8)[0]
                header = 16
            if size == 0:
                size = end - start
            if size < header or start + size > end:
                break
            yield kind, start + header, start + size
            start += size

    def child(box, kind):
        return next((b for b in boxes(box[1], box[2]) if b[0] == kind), None)

    moov = next(b for b in boxes(0, len(data)) if b[0] == b"moov")
    for track in boxes(moov[1], moov[2]):
        if track[0] != b"trak":
            continue
        mdia = child(track, b"mdia")
        handler = child(mdia, b"hdlr")
        if data[handler[1] + 8 : handler[1] + 12] != b"vide":
            continue
        mdhd = child(mdia, b"mdhd")
        offset = mdhd[1] + (20 if data[mdhd[1]] == 1 else 12)
        timescale = struct.unpack_from(">I", data, offset)[0]
        stts = child(child(child(mdia, b"minf"), b"stbl"), b"stts")
        entries = struct.unpack_from(">I", data, stts[1] + 4)[0]
        timing = [
            struct.unpack_from(">II", data, stts[1] + 8 + i * 8)
            for i in range(entries)
        ]
        count = sum(n for n, delta in timing)
        duration = sum(n * delta for n, delta in timing) / timescale
        return dict(
            duration_s=duration,
            frames=count,
            fps=count / duration,
            variable_frame_rate=len({delta for n, delta in timing if n}) > 1,
            duration_source="MOV video stts/mdhd",
            time_base=1 / timescale,
        )
    raise ValueError("Video timing track missing")


class FileReplay:
    def __init__(self, path, directory, run_id, api_base, captures_root):
        import cv2

        self.cv2, self.path, self.directory = cv2, path, directory
        directory.mkdir(parents=True, exist_ok=False)
        self.session_id = uuid.uuid4().hex
        self.capture_dir = (
            captures_root / "evaluation_results" / run_id / path.stem
        )
        self.image_prefix = (
            f"/captures/evaluation_results/{run_id}/{path.stem}"
        )
        self.api_base = api_base.rstrip("/")
        self.capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            raise RuntimeError("Video cannot be opened")
        self.metadata = dict(
            file=path.name,
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            opencv_fps=self.capture.get(cv2.CAP_PROP_FPS),
            opencv_frame_count=int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            rotation_degrees=self.capture.get(cv2.CAP_PROP_ORIENTATION_META),
        )
        self.metadata.update(mov_timing(path))
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.last_completed = self.frame_count = 0
        self.worker_error = False
        self.media_time = self.last_frame_timestamp = self.capture_read_ms = (
            0.0
        )
        self.capture_fps, self.dropped_frames = self.metadata["fps"], 0
        self.preview = SimpleNamespace(
            submit=lambda *a: None,
            update_results=lambda *a: None,
            set_inference_error=lambda *a: None,
            close=lambda: None,
        )
        self.diagnostics = SimpleNamespace(
            due=lambda: False, finish=lambda *a: None, error=lambda *a: None
        )
        self.frames, self.detections, self.confirmed, self.errors = (
            [],
            [],
            [],
            [],
        )
        self.deliveries, self.submissions = [], []
        self.thresholds = {}
        self.first_decoded = None
        self.evidence = directory / "crops"
        self.evidence.mkdir()
        self.best_crops = {}
        self.jsonl = (directory / "frames.jsonl").open("w", encoding="utf-8")

    def read(self):
        started = time.perf_counter()
        success, frame = self.capture.read()
        self.capture_read_ms = (time.perf_counter() - started) * 1000
        self.last_frame_timestamp = time.monotonic()
        if success:
            self.frame_count += 1
            pts = self.capture.get(self.cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(pts) or (
                self.frame_count > 1 and pts <= self.media_time
            ):
                pts = (self.frame_count - 1) / self.capture_fps
            self.media_time = pts
            if self.first_decoded is None:
                self.first_decoded = self.last_frame_timestamp
                self.metadata.update(
                    width=frame.shape[1], height=frame.shape[0]
                )
        return success, frame

    def release(self):
        self.stop.set()
        self.capture.release()

    def attach_sender(self, sender):
        original = sender.transport

        def send(payload):
            row = dict(
                event_id=payload["plates"][0]["event_id"],
                session_id=self.session_id,
                plate=payload["plates"][0]["plate"],
                success=False,
            )
            try:
                original(payload)
                row["success"] = True
            except Exception as error:
                row["error_type"] = type(error).__name__
                raise
            finally:
                self.deliveries.append(row)

        sender.transport = send

    def record_submission(self, plate, queued):
        self.submissions.append(
            dict(frame=self.frame_count, plate=plate["plate"], queued=queued)
        )

    def drain_sender(self, sender):
        deadline = time.monotonic() + 120
        while sender.queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        if sender.queue.unfinished_tasks:
            self.errors.append(
                dict(
                    code="delivery_drain_timeout",
                    pending=sender.queue.unfinished_tasks,
                )
            )

    def wait_frame(self, frame_id):
        with self.condition:
            complete = self.condition.wait_for(
                lambda: self.last_completed >= frame_id or self.worker_error,
                timeout=120,
            )
            if not complete or self.worker_error:
                raise RuntimeError(
                    "Replay recognition worker failed or timed out"
                )

    def fail_worker(self):
        with self.condition:
            self.worker_error = True
            self.condition.notify_all()

    def record_error(self, frame_id, media_time, code):
        self.errors.append(
            dict(frame=frame_id, video_time_s=media_time, code=code)
        )
        with self.condition:
            self.last_completed = frame_id
            self.condition.notify_all()

    def record_frame(
        self, frame_id, media_time, output, plates, confirmed, total_ms
    ):
        source = output["source"]
        slim = []
        for p in plates:
            item = {
                key: value for key, value in p.items() if key != "patch_png"
            }
            x1, y1, x2, y2 = p["box"]
            cx, cy = (
                (x1 + x2) / 2 / source["width"],
                (y1 + y2) / 2 / source["height"],
            )
            item.update(
                frame=frame_id,
                video_time_s=media_time,
                position=dict(
                    x=round(cx - 0.5, 3),
                    y=round(cy - 0.5, 3),
                    offset_x_px=round((x1 + x2 - source["width"]) / 2),
                    offset_y_px=round((y1 + y2 - source["height"]) / 2),
                    direction="LEFT"
                    if cx < 0.4
                    else "RIGHT"
                    if cx > 0.6
                    else "CENTER",
                ),
            )
            key = p.get("raw_text") or p["text"] or "unreadable"
            if p.get("patch_png") and p[
                "ocr_confidence"
            ] > self.best_crops.get(key, -1):
                name = hashlib.sha256(key.encode()).hexdigest()[:12] + ".png"
                (self.evidence / name).write_bytes(
                    base64.b64decode(p["patch_png"].split(",", 1)[1])
                )
                self.best_crops[key] = p["ocr_confidence"]
                item["crop_path"] = "crops/" + name
            slim.append(item)
        rows = [
            dict(frame=frame_id, video_time_s=media_time, **p)
            for p in confirmed
        ]
        self.confirmed.extend(rows)
        self.detections.extend(slim)
        item = dict(
            frame=frame_id,
            video_time_s=media_time,
            timing_ms=output["timing_ms"],
            pipeline_with_input_jpeg_ms=total_ms,
            decode_ms=self.capture_read_ms,
            detections=slim,
            confirmed=rows,
        )
        self.frames.append(item)
        self.jsonl.write(
            json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
        )
        self.jsonl.flush()
        with self.condition:
            self.last_completed = frame_id
            self.condition.notify_all()

    def finish(self, wall_seconds):
        self.jsonl.close()
        if self.frame_count != self.metadata["frames"]:
            self.errors.append(
                dict(
                    code="decoded_frame_count_mismatch",
                    expected=self.metadata["frames"],
                    actual=self.frame_count,
                )
            )
        if (
            len(self.frames) + len([e for e in self.errors if "frame" in e])
            != self.frame_count
        ):
            self.errors.append(dict(code="not_all_frames_accounted_for"))
        pipeline_times = [f["timing_ms"]["total"] for f in self.frames]
        found = []
        for text in sorted(
            {p["raw_text"] or p["text"] for p in self.detections}
        ):
            rows = [
                p
                for p in self.detections
                if (p["raw_text"] or p["text"]) == text
            ]
            confidences = [p["ocr_confidence"] for p in rows]
            crop_name = (
                hashlib.sha256((text or "unreadable").encode()).hexdigest()[
                    :12
                ]
                + ".png"
            )
            found.append(
                dict(
                    text=text,
                    observations=len(rows),
                    first_s=rows[0]["video_time_s"],
                    last_s=rows[-1]["video_time_s"],
                    mean_confidence=statistics.mean(confidences),
                    max_confidence=max(confidences),
                    format_valid=any(p["rover_format_valid"] for p in rows),
                    confirmed=any(p["status"] == "confirmed" for p in rows),
                    rejection_reasons=dict(
                        Counter(
                            reason for p in rows for reason in p["reasons"]
                        )
                    ),
                    directions=sorted(
                        {p["position"]["direction"] for p in rows}
                    ),
                    crop_path="crops/" + crop_name
                    if (self.evidence / crop_name).exists()
                    else None,
                )
            )
        tracks = []
        for track in sorted({p["track_id"] for p in self.confirmed}):
            rows = [p for p in self.confirmed if p["track_id"] == track]
            tracks.append(
                dict(
                    track_id=track,
                    plates=sorted({p["plate"] for p in rows}),
                    first_s=rows[0]["video_time_s"],
                    last_s=rows[-1]["video_time_s"],
                    mean_confidence=statistics.mean(
                        p["confidence"] for p in rows
                    ),
                    observations=len(rows),
                    first_position=rows[0]["position"],
                    last_position=rows[-1]["position"],
                    best_image=next(
                        (
                            p["image_path"]
                            for p in reversed(rows)
                            if p.get("image_path")
                        ),
                        None,
                    ),
                )
            )
        suspicious = []
        for item in found:
            flags = []
            if not item["format_valid"]:
                flags.append("invalid_format")
            if not item["confirmed"]:
                flags.append("not_confirmed")
            if item["observations"] < 3:
                flags.append("fewer_than_three_observations")
            if flags:
                suspicious.append(dict(text=item["text"], flags=flags))
        for track in tracks:
            if len(track["plates"]) > 1:
                suspicious.append(
                    dict(
                        track_id=track["track_id"],
                        flags=["confirmed_number_changed"],
                        variants=track["plates"],
                    )
                )
        report = dict(
            metadata=self.metadata,
            session_id=self.session_id,
            thresholds=self.thresholds,
            protocol=(
                "all frames sequentially through camera.main; tracking uses "
                "video timestamps; no live dropping"
            ),
            processed_frames=len(self.frames),
            decoded_frames=self.frame_count,
            wall_seconds=wall_seconds,
            processing_fps=len(self.frames) / wall_seconds,
            real_time_factor=self.metadata["duration_s"] / wall_seconds,
            pipeline_mean_ms=statistics.mean(pipeline_times)
            if pipeline_times
            else None,
            pipeline_p50_ms=percentile(pipeline_times, 50),
            pipeline_p95_ms=percentile(pipeline_times, 95),
            pipeline_max_ms=max(pipeline_times) if pipeline_times else None,
            detection_observations=len(self.detections),
            confirmed_observations=len(self.confirmed),
            found_numbers=found,
            confirmed_tracks=tracks,
            suspicious=suspicious,
            errors=self.errors,
            api=dict(
                submitted=len(self.submissions),
                queued=sum(r["queued"] for r in self.submissions),
                acknowledged=sum(r["success"] for r in self.deliveries),
                failed_attempts=sum(not r["success"] for r in self.deliveries),
            ),
            ground_truth=(
                "not provided; confidence is model output, not measured "
                "accuracy"
            ),
        )
        write_json(self.directory / "results.json", report)
        write_csv(self.directory / "detections.csv", self.detections)
        write_csv(self.directory / "confirmed.csv", self.confirmed)
        write_csv(self.directory / "deliveries.csv", self.deliveries)
        write_csv(
            self.directory / "frames.csv",
            [
                dict(
                    frame=f["frame"],
                    video_time_s=f["video_time_s"],
                    pipeline_ms=f["timing_ms"]["total"],
                    detection_ms=f["timing_ms"]["detect"],
                    ocr_ms=f["timing_ms"]["recognize"],
                    decode_ms=f["decode_ms"],
                    detections=len(f["detections"]),
                    confirmed=len(f["confirmed"]),
                )
                for f in self.frames
            ],
        )
        return report


def api_json(base, path):
    with urllib.request.urlopen(
        base.rstrip("/") + path, timeout=15
    ) as response:
        return json.load(response)


def build_report(root):
    summary = json.loads((root / "summary.json").read_text())
    track_rows, number_rows, detection_rows, issues, audits = (
        [],
        [],
        [],
        [],
        [],
    )
    reports = summary["videos"]
    manifest = summary["manifest"]
    for result in reports:
        video = result["metadata"]["file"]
        directory = root / Path(video).stem
        frames = [
            json.loads(line)
            for line in (directory / "frames.jsonl").read_text().splitlines()
        ]
        confirmations = [
            dict(item, video_time_s=frame["video_time_s"])
            for frame in frames
            for item in frame["confirmed"]
        ]
        deliveries = (
            list(csv.DictReader((directory / "deliveries.csv").open()))
            if confirmations
            else []
        )
        ack_ids = [
            row["event_id"] for row in deliveries if row["success"] == "True"
        ]
        audit = dict(
            video=video,
            expected_frames=result["metadata"]["frames"],
            processed=len(frames),
            unique_frame_ids=len({f["frame"] for f in frames}),
            contiguous_frame_ids=[f["frame"] for f in frames]
            == list(range(1, len(frames) + 1)),
            confirmed_events=len(confirmations),
            acknowledged_events=len(ack_ids),
            unique_acknowledged_ids=len(set(ack_ids)),
            errors=result["errors"],
        )
        position_checks = []
        for frame in frames:
            for confirmed in frame["confirmed"]:
                matching = [
                    p
                    for p in frame["detections"]
                    if p["track_id"] == confirmed["track_id"]
                ]
                position_checks.append(
                    bool(matching)
                    and matching[0]["position"] == confirmed["position"]
                )
        audit["confirmed_positions_verified"] = sum(position_checks)
        audit["positions_match_production"] = all(position_checks)
        audit["passed"] = (
            audit["expected_frames"]
            == audit["processed"]
            == audit["unique_frame_ids"]
            and audit["contiguous_frame_ids"]
            and len(confirmations) == len(ack_ids) == len(set(ack_ids))
            and not result["errors"]
            and audit["positions_match_production"]
        )
        audits.append(audit)
        detections = [
            dict(
                item, frame=frame["frame"], video_time_s=frame["video_time_s"]
            )
            for frame in frames
            for item in frame["detections"]
        ]
        for item in detections:
            x1, y1, x2, y2 = item["box"]
            position = item["position"]
            detection_rows.append(
                dict(
                    video=video,
                    frame=item["frame"],
                    time_s=item["video_time_s"],
                    raw_text=item["raw_text"],
                    text=item["text"],
                    detector_confidence=item["detection_confidence"],
                    ocr_confidence=item["ocr_confidence"],
                    readable=item["readable"],
                    model_format_valid=item["model_format_valid"],
                    rover_format_valid=item["rover_format_valid"],
                    status=item["status"],
                    reasons="|".join(item["reasons"]),
                    track_id=item["track_id"],
                    confirmed_number=item["confirmed_number"],
                    votes=item["confirmations"],
                    required=item.get("required"),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    relative_x=position["x"],
                    relative_y=position["y"],
                    offset_x_px=position["offset_x_px"],
                    offset_y_px=position["offset_y_px"],
                    direction=position["direction"],
                )
            )
        for number in result["found_numbers"]:
            rows = [
                item
                for item in detections
                if (item["raw_text"] or item["text"]) == number["text"]
            ]
            accepted = [
                item
                for item in rows
                if item["readable"]
                and item["ocr_confidence"] >= 0.85
                and item["rover_format_valid"]
            ]
            row = dict(
                video=video,
                **number,
                accepted_observations=len(accepted),
                accepted_mean_confidence=statistics.mean(
                    p["ocr_confidence"] for p in accepted
                )
                if accepted
                else None,
            )
            number_rows.append(row)
            if not number["confirmed"] and accepted:
                issues.append(
                    dict(
                        video=video,
                        text=number["text"],
                        flag="accepted_hypothesis_not_confirmed",
                        observations=len(rows),
                        accepted_observations=len(accepted),
                        max_confidence=number["max_confidence"],
                        crop_path=number["crop_path"],
                    )
                )
        for track in result["confirmed_tracks"]:
            rows = [
                item
                for item in confirmations
                if item["track_id"] == track["track_id"]
            ]
            observed = [
                item
                for item in detections
                if item["track_id"] == track["track_id"]
            ]
            changes = []
            previous = None
            for item in rows:
                if item["plate"] != previous:
                    changes.append(
                        dict(time_s=item["video_time_s"], plate=item["plate"])
                    )
                    previous = item["plate"]
            record = dict(
                video=video,
                **track,
                label_sequence=changes,
                min_confidence=min(p["confidence"] for p in rows),
                max_confidence=max(p["confidence"] for p in rows),
                first_box=next(
                    (
                        p["box"]
                        for p in observed
                        if p["frame"]
                        == next(
                            f["frame"]
                            for f in frames
                            if f["video_time_s"] == rows[0]["video_time_s"]
                        )
                    ),
                    None,
                ),
                confirmed_label_counts=dict(Counter(p["plate"] for p in rows)),
            )
            track_rows.append(record)
            if len(changes) > 1:
                issues.append(
                    dict(
                        video=video,
                        track_id=track["track_id"],
                        flag="confirmed_label_instability",
                        transitions=changes,
                        counts=record["confirmed_label_counts"],
                    )
                )
            if len(rows) <= 2:
                issues.append(
                    dict(
                        video=video,
                        track_id=track["track_id"],
                        flag="short_confirmed_support",
                        plates=track["plates"],
                        confirmed_output_frames=len(rows),
                        accepted_observations=len(
                            [
                                p
                                for p in observed
                                if p["readable"]
                                and p["ocr_confidence"] >= 0.85
                                and p["rover_format_valid"]
                            ]
                        ),
                    )
                )
    write_csv(root / "detections_all.csv", detection_rows)
    write_csv(root / "numbers_all.csv", number_rows)
    write_csv(root / "confirmed_tracks.csv", track_rows)
    write_json(root / "suspicious_review.json", issues)
    project = Path(__file__).resolve().parent
    source_hashes = [
        hashlib.sha256(
            (project / "test_videos" / r["metadata"]["file"]).read_bytes()
        ).hexdigest()
        == r["metadata"]["sha256"]
        for r in reports
        if (project / "test_videos" / r["metadata"]["file"]).is_file()
    ]
    weight_hashes = [
        hashlib.sha256(
            (project / "checkpoints" / name).read_bytes()
        ).hexdigest()
        == digest
        for name, digest in manifest["checkpoint_sha256"].items()
        if (project / "checkpoints" / name).is_file()
    ]
    write_json(
        root / "verification.json",
        dict(
            videos=audits,
            source_hashes_unchanged=all(source_hashes)
            if len(source_hashes) == len(reports)
            else None,
            checkpoint_hashes_unchanged=all(weight_hashes)
            if len(weight_hashes) == len(manifest["checkpoint_sha256"])
            else None,
            confirmed_positions_verified=sum(
                a["confirmed_positions_verified"] for a in audits
            ),
            all_confirmed_positions_match_production=all(
                a["positions_match_production"] for a in audits
            ),
            all_frames_and_deliveries_verified=all(
                a["passed"] for a in audits
            ),
            api_history_rows=summary["api_history_rows"],
            confirmed_tracks=len(track_rows),
        ),
    )
    total_frames = sum(r["processed_frames"] for r in reports)
    wall = sum(r["wall_seconds"] for r in reports)
    expected_frames = sum(r["metadata"]["frames"] for r in reports)
    source_seconds = sum(r["metadata"]["duration_s"] for r in reports)
    events = sum(r["confirmed_observations"] for r in reports)
    acknowledgements = sum(r["api"]["acknowledged"] for r in reports)
    failed_attempts = sum(r["api"]["failed_attempts"] for r in reports)
    audit_text = (
        "Потерь кадров и ошибок обработки не обнаружено."
        if all(a["passed"] for a in audits)
        else (
            "Проверка полноты обнаружила несоответствия; см. "
            "verification.json."
        )
    )
    lines = [
        "# Испытания ANPR на пяти реальных видеозаписях",
        "",
        (
            f"Запуск UTC: {manifest['started_at']}. Среда: "
            f"{manifest['gpu']}, PyTorch {manifest['torch']}, OpenCV "
            f"{manifest['opencv']}."
        ),
        (
            "Источник результатов — фактический прогон всех файлов. Числа "
            "ниже не являются прогнозом."
        ),
        "",
        "## Методика",
        "",
        (
            "Использован существующий `camera.main`: YOLO26n → OCR → "
            "штатная проверка формата → трекинг и голосование → EventSender "
            "→ работающий API → SQLite."
        ),
        (
            "Модели загружены на CUDA один раз для пяти файлов. Порог YOLO "
            "0.5, IoU 0.45, вход 640; OCR ≥ 0.85; подтверждение 3 из "
            "последних 5 принятых наблюдений, смена номера 7 из истории 10."
        ),
        (
            "Обработан каждый кадр, без повторного RTSP-захвата и без "
            "live-пропуска кадров. Межкадровые таймауты считаются по PTS "
            "видео. Это пакетный тест существующего алгоритма, а не "
            "измерение задержки живой камеры."
        ),
        (
            f"FPS обработки включает декодирование, JPEG входа, pipeline, "
            f"трекинг, запись артефактов и завершение доставки API. Загрузка "
            f"моделей и общий прогрев {manifest['warmup_ms'] / 1000:.2f} с "
            f"вынесены за пределы времени роликов."
        ),
        (
            "Duration/FPS проверены по таблицам времени MOV, не только по "
            "приблизительным свойствам OpenCV. Значения FPS и "
            "variable_frame_rate сохранены в метаданных каждого ролика. "
            "Источники не масштабировались до входа в существующий "
            "pipeline."
        ),
        "",
        "## Сравнение видео",
        "",
        (
            "| Видео | Длительность, с | Разрешение | FPS источника | Кадры "
            "| FPS обработки | Pipeline mean / p95, мс | Подтверждённые "
            "треки | API-события |"
        ),
        "|---|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for r in reports:
        m = r["metadata"]
        lines.append(
            (
                f"| {m['file']} | {m['duration_s']:.3f} | "
                f"{m['width']}×{m['height']} | {m['fps']:.0f} | "
                f"{r['processed_frames']} | {r['processing_fps']:.2f} | "
                f"{r['pipeline_mean_ms']:.2f} / {r['pipeline_p95_ms']:.2f} | "
                f"{len(r['confirmed_tracks'])} | {r['api']['acknowledged']} "
                f"|"
            )
        )
    lines += [
        "",
        (
            f"Всего: "
            f"**{total_frames}/{expected_frames} кадров**, "
            f"{source_seconds:.3f} с исходного видео, {wall:.3f} с обработки, "
            f"средняя полная пропускная способность "
            f"**{total_frames / wall:.2f} FPS**."
        ),
        (
            f"Получено {sum(r['detection_observations'] for r in reports)} "
            f"наблюдения детектор→OCR, {events} подтверждённых покадровых "
            f"событий и {len(track_rows)} треков. API подтвердил "
            f"{acknowledgements} событий, неудачных попыток: "
            f"{failed_attempts}; `/history` содержит "
            f"{summary['api_history_rows']} строк этого запуска. "
            f"{audit_text}"
        ),
        (
            f"Число треков/строк SQLite не равно числу событий: один трек "
            f"обновляется многими событиями. {len(track_rows)} треков дали "
            f"{sum(len(t['plates']) for t in track_rows)} вариантов "
            f"подтверждённых строк."
        ),
        "",
        "## Подтверждённые номера и положение",
        "",
        (
            "Буквы записаны в латинском алфавите модели. Confidence ниже — "
            "среднее штатной скользящей уверенности в подтверждённых "
            "выходах, не вероятность истинности номера."
        ),
        "",
        (
            "| Видео | Трек | Подтверждённые строки | Confidence, % | "
            "Первое/последнее подтверждение, с | Первое → последнее "
            "положение (x,y) |"
        ),
        "|---|---:|---|---:|---|---|",
    ]
    for t in track_rows:
        a, b = t["first_position"], t["last_position"]
        lines.append(
            (
                f"| {t['video']} | {t['track_id']} | "
                f"{', '.join(t['plates'])} | "
                f"{100 * t['mean_confidence']:.2f} | {t['first_s']:.3f} / "
                f"{t['last_s']:.3f} | {a['direction']} "
                f"({a['x']:.3f},{a['y']:.3f}) → {b['direction']} "
                f"({b['x']:.3f},{b['y']:.3f}) |"
            )
        )
    lines += [
        "",
        (
            "Относительные координаты: центр изображения = (0,0), диапазон "
            "примерно [-0.5,+0.5], x вправо, y вниз. LEFT соответствует "
            "центру box левее 40% ширины, RIGHT — правее 60%. Все исходные "
            "x1/y1/x2/y2 и пиксельные смещения доступны в "
            "`detections_all.csv`."
        ),
        "",
        "## Ошибки, неоднозначности и подозрительные результаты",
        "",
        (root / "manual_review.md").read_text(encoding="utf-8")
        if (root / "manual_review.md").exists()
        else (
            "Ручная проверка не выполнена. Эвристические флаги находятся в "
            "suspicious_review.json."
        ),
        (
            "Сырые OCR-строки с низкой уверенностью, неподходящим "
            "subtype/форматом и недостатком подтверждений сохранены, а не "
            "удалены из отчёта. Их полный список — numbers_all.csv; причины "
            "каждого отбрасывания — detections_all.csv. Они не считаются "
            "автоматически ложными подтверждениями."
        ),
        (
            "Полной эталонной разметки нет: **precision, recall и точность "
            "распознавания не вычислялись**. Высокая confidence и "
            "количество подтверждений не заменяют сверку с истинными "
            "номерами."
        ),
        "",
        "## Файлы результатов",
        "",
        (
            "- `manifest.json`: версии, GPU, SHA-256 весов; SHA-256 каждого "
            "источника — в JSON ролика."
        ),
        (
            "- `summary.json`, `comparison.csv`: сводные реальные метрики "
            "пяти роликов."
        ),
        (
            "- `detections_all.csv`: все покадровые чтения с confidence, "
            "координатами, положением и причинами отклонения."
        ),
        (
            "- `numbers_all.csv`: все уникальные сырые строки и их "
            "статистика, включая неподтверждённые."
        ),
        (
            "- `confirmed_tracks.csv`: треки, уверенности, позиции и "
            "последовательность смены текста."
        ),
        (
            "- `suspicious_review.json`, `verification.json`: эвристические "
            "флаги и проверка полноты кадров/доставки."
        ),
        (
            "- `api_history.json`: фактические записи SQLite этого запуска, "
            "полученные через API."
        ),
        (
            "- В каталоге каждого ролика: `results.json`, `frames.jsonl`, "
            "`frames.csv`, `detections.csv`, `confirmed.csv`, "
            "`deliveries.csv`, `visibility.csv`, `worker.log`, полные "
            "снимки и `crops/`."
        ),
        "",
        (
            "Результаты и видео исключены из Git. Веса не изменялись. "
            "Docker не пересобирался, volumes не удалялись. Новые тестовые "
            "события остаются в API/SQLite с отдельными session_id и путями "
            "этого запуска."
        ),
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("test_videos"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--report-only",
        type=Path,
        help="rebuild a report in a chosen evaluation directory",
    )
    parser.add_argument("--api-base", default="http://api:8000")
    parser.add_argument(
        "--captures-root", type=Path, default=Path("/app/captures")
    )
    args = parser.parse_args()
    if args.report_only is not None:
        build_report(args.report_only)
        return
    if args.output is None:
        parser.error("--output is required for inference")
    paths = [args.videos / name for name in VIDEO_NAMES]
    if not all(path.is_file() for path in paths):
        parser.error("All five source videos must exist")
    args.output.mkdir(parents=True, exist_ok=False)
    if api_json(args.api_base, "/health").get("status") != "ok":
        raise RuntimeError("Existing API is not healthy")
    import cv2
    import torch

    from camera import main as camera_main
    from full_stand.config import StandConfig
    from full_stand.pipeline import PlatePipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required; CPU fallback is forbidden for this evaluation"
        )
    config = StandConfig(device="cuda", confidence=0.5)
    config.validate()
    manifest = dict(
        started_at=datetime.now(timezone.utc).isoformat(),
        torch=torch.__version__,
        opencv=cv2.__version__,
        gpu=torch.cuda.get_device_name(0),
        checkpoint_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (config.detector, config.recognizer)
        },
        videos=VIDEO_NAMES,
        api_health="ok",
    )
    write_json(args.output / "manifest.json", manifest)
    print("Loading existing models once on CUDA", flush=True)
    pipeline = PlatePipeline(config)
    if pipeline.device.type != "cuda":
        raise RuntimeError("Pipeline is not on CUDA")
    manifest["warmup_ms"] = pipeline.warmup()
    write_json(args.output / "manifest.json", manifest)
    reports = []
    for path in paths:
        replay = FileReplay(
            path,
            args.output / path.stem,
            args.output.name,
            args.api_base,
            args.captures_root,
        )
        print("Processing " + path.name, flush=True)
        started = time.perf_counter()
        try:
            with (
                (replay.directory / "worker.log").open(
                    "w", encoding="utf-8"
                ) as log,
                contextlib.redirect_stdout(log),
            ):
                camera_main(
                    dict(
                        VIDEO_SOURCE=str(path),
                        API_URL=args.api_base.rstrip("/") + "/plates",
                        DEVICE="cuda",
                        SHOW_WINDOW="false",
                    ),
                    replay=replay,
                    pipeline_override=pipeline,
                )
        except Exception as error:
            replay.errors.append(
                dict(code="run_failed", error_type=type(error).__name__)
            )
        report = replay.finish(time.perf_counter() - started)
        reports.append(report)
        print(
            json.dumps(
                dict(
                    file=path.name,
                    frames=report["processed_frames"],
                    fps=report["processing_fps"],
                    confirmed=[
                        p["plates"] for p in report["confirmed_tracks"]
                    ],
                    errors=report["errors"],
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )
    prefix = "/captures/evaluation_results/" + args.output.name + "/"
    persisted = [
        row
        for row in api_json(args.api_base, "/history")["detections"]
        if (row.get("image_path") or "").startswith(prefix)
    ]
    write_json(args.output / "api_history.json", persisted)
    write_json(
        args.output / "summary.json",
        dict(
            manifest=manifest, videos=reports, api_history_rows=len(persisted)
        ),
    )
    comparison = []
    for report in reports:
        m = report["metadata"]
        confirmed = sorted(
            {text for t in report["confirmed_tracks"] for text in t["plates"]}
        )
        comparison.append(
            dict(
                video=m["file"],
                duration_s=m["duration_s"],
                width=m.get("width"),
                height=m.get("height"),
                source_fps=m["fps"],
                processed_frames=report["processed_frames"],
                processing_fps=report["processing_fps"],
                pipeline_mean_ms=report["pipeline_mean_ms"],
                pipeline_p95_ms=report["pipeline_p95_ms"],
                confirmed_plates=confirmed,
                confirmed_tracks=len(report["confirmed_tracks"]),
                errors=len(report["errors"]),
                api_acknowledged=report["api"]["acknowledged"],
            )
        )
    write_csv(args.output / "comparison.csv", comparison)
    build_report(args.output)
    print("Evaluation complete: " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
