"""Bounded preview: latest frame, asynchronous JPEG, atomic transport."""

import base64
import json
import math
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "captures" / ".live-preview"
MAX_BYTES = 8_000_000
STALE_SECONDS = 2.0
PUBLIC_FIELDS = (
    "session_id",
    "timestamp",
    "frame_id",
    "result_frame_id",
    "fps",
    "processing_ms",
    "plates",
    "width",
    "height",
    "recognition",
    "preview_fps",
    "target_fps",
    "capture_fps",
    "capture_read_ms",
    "capture_ipc_ms",
    "inference_age_ms",
    "inference_input_jpeg_ms",
    "jpeg_ms",
    "inference_queue_ms",
    "inference_total_ms",
    "preview_render_ms",
    "preview_wait_ms",
    "volume_write_ms",
    "published_at",
    "overlay_age_ms",
    "dropped_preview",
    "dropped_capture",
    "jpeg_bytes",
    "inference_state",
    "inference_error",
)


class RunStatistics:
    """Measured online aggregates; no synthetic rates or unbounded samples."""

    def __init__(self):
        self.started = time.monotonic()
        self.frames = 0
        self.latencies = {}
        self.capture_fps = None
        self.dropped_capture = None

    def add(self, timings):
        self.frames += 1
        for name, value in timings.items():
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                continue
            row = self.latencies.setdefault(
                name, dict(count=0, sum_ms=0.0, min_ms=value, max_ms=value)
            )
            row["count"] += 1
            row["sum_ms"] += value
            row["min_ms"] = min(row["min_ms"], value)
            row["max_ms"] = max(row["max_ms"], value)

    def summary(self):
        elapsed = time.monotonic() - self.started
        return dict(
            elapsed_seconds=elapsed,
            processed_frames=self.frames,
            average_anpr_fps=self.frames / elapsed if elapsed > 0 else None,
            capture_fps_last=self.capture_fps,
            dropped_capture=self.dropped_capture,
            latency_ms={
                k: dict(
                    count=v["count"],
                    mean_ms=v["sum_ms"] / v["count"],
                    min_ms=v["min_ms"],
                    max_ms=v["max_ms"],
                )
                for k, v in self.latencies.items()
            },
            scope="Worker completions / elapsed wall ti"
            "me; not source playback FPS. "
            "worker_cycle excludes async API deli"
            "very, preview and file export.",
        )


class PreviewReader:
    """Cache each atomic file version once; compute age on every request."""

    def __init__(self, path=DEFAULT_PATH, max_age=STALE_SECONDS):
        self.max_age = max_age
        self.path = path
        self.lock = threading.Lock()
        self.version = None
        self.cached = (None, {})
        self.cached_at = None

    def read(self):
        with self.lock:
            try:
                with open(self.path, "rb") as source:
                    stat = os.fstat(source.fileno())
                    version = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                    if version != self.version:
                        started = time.perf_counter()
                        data = source.read(MAX_BYTES + 1)
                        if len(data) > MAX_BYTES:
                            raise ValueError
                        header, jpeg = data.split(b"\n", 1)
                        raw = json.loads(header)
                        if not jpeg.startswith(
                            b"\xff\xd8"
                        ) or not jpeg.endswith(b"\xff\xd9"):
                            raise ValueError
                        meta = {k: raw[k] for k in PUBLIC_FIELDS if k in raw}
                        meta["api_read_ms"] = (
                            time.perf_counter() - started
                        ) * 1000
                        meta["transfer_to_api_ms"] = max(
                            0,
                            (
                                time.time()
                                - meta.get("published_at", raw["timestamp"])
                            )
                            * 1000,
                        )
                        self.cached = (jpeg, meta)
                        self.cached_at = time.monotonic()
                        self.version = version
                jpeg, meta = self.cached
                age = time.time() - meta["timestamp"]
                published = meta.get("published_at")
                if published is not None:
                    # Producer and API may run on different Windows/WSL clocks.
                    # Their same-clock duration remains valid; local monotonic
                    # time makes an unchanged file expire
                    # even with a future TS.
                    producer_age = published - meta["timestamp"]
                    if not math.isfinite(producer_age) or producer_age < 0:
                        return None, {}
                    age = max(
                        age,
                        producer_age
                        + max(0, time.monotonic() - self.cached_at),
                    )
                if not 0 <= age < self.max_age:
                    return None, {}
                return jpeg, dict(meta, frame_age_ms=age * 1000)
            except (OSError, ValueError, KeyError, TypeError):
                self.cached = (None, {})
                self.version = None
                return None, {}


_reader = PreviewReader()


def read_preview(path=None):
    return _reader.read() if path is None else PreviewReader(path).read()


def plate_crop(frame, box):
    """Bounded JPEG from the untouched source frame, not
    resized/annotated preview."""
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
    x2, y2 = min(w, math.ceil(x2)), min(h, math.ceil(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    scale = min(1.0, 320 / crop.shape[1], 160 / crop.shape[0])
    if scale < 1:
        crop = cv2.resize(
            crop,
            (
                max(1, round(crop.shape[1] * scale)),
                max(1, round(crop.shape[0] * scale)),
            ),
        )
    ok, data = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return (
        "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
        if ok
        else None
    )


class PreviewPublisher:
    def __init__(self, settings, path=DEFAULT_PATH):
        self.fps = float(settings.get("LIVE_FPS", "20"))
        self.width = int(settings.get("LIVE_MAX_WIDTH", "1920"))
        self.quality = int(settings.get("LIVE_JPEG_QUALITY", "90"))
        self.overlay_age = float(settings.get("LIVE_OVERLAY_MAX_AGE", "0.25"))
        if (
            not math.isfinite(self.fps)
            or not 1 <= self.fps <= 30
            or not 320 <= self.width <= 3840
            or not 30 <= self.quality <= 100
            or not math.isfinite(self.overlay_age)
            or not 0 <= self.overlay_age <= 1
        ):
            raise ValueError("Invalid live preview settings")
        self.path = Path(settings.get("LIVE_PREVIEW_PATH", path))
        self.readings_path = Path(str(self.path) + ".readings.json")
        self.readings = dict(
            session_id=uuid.uuid4().hex,
            detections=0,
            ocr_readings=0,
            complete_readings=0,
            partial_readings=0,
            correct=None,
            evaluated=0,
            accuracy_percent=None,
            counting_unit="accepted_plate_box_per_processed_frame",
            last_reading=None,
        )
        self.last_reading_write = 0.0
        self.statistics = RunStatistics()
        self.input_frames = 0
        self.closed = False
        self.keep_frame = False
        self.stats_path = Path(
            settings.get("STATS_PATH")
            or self.path.parent / ("live_stats_" + uuid.uuid4().hex + ".json")
        )
        self.lock = threading.Lock()
        self.pending = None
        self.results = ([], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)
        self.inference_error = None
        self.dropped = 0
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def record_readings(
        self, frame, raw_plates, live_plates, captured_at, session_id
    ):
        # One detector box per processed frame, never unique cars or tracks.
        readable = [
            p for p in raw_plates if p.get("readable") and p.get("text")
        ]
        self.readings.update(session_id=session_id, updated_at=time.time())
        self.readings["detections"] += len(raw_plates)
        self.readings["ocr_readings"] += len(readable)
        self.readings["complete_readings"] += sum(
            "#" not in p["text"] for p in readable
        )
        self.readings["partial_readings"] += sum(
            "#" in p["text"] for p in readable
        )
        if raw_plates:
            import cv2

            # The last box in pipeline order is deterministic;
            # all boxes count above.
            chosen = raw_plates
            if getattr(self, "prefer_confirmed", False):
                boxes = {
                    tuple(p["box"])
                    for p in live_plates
                    if p.get("status") == "CONFIRMED"
                }
                chosen = [
                    p for p in raw_plates if tuple(p["box"]) in boxes
                ] or raw_plates
            raw = chosen[-1]
            live = next((p for p in live_plates if p["box"] == raw["box"]), {})
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = raw["box"]
            x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
            x2, y2 = min(w, math.ceil(x2)), min(h, math.ceil(y2))
            reading = dict(
                text=live.get("text") or raw.get("text") or "",
                ocr_text=raw.get("text") or "",
                raw_text=raw.get("raw_text") or "",
                status=live.get("status") or raw.get("status") or "UNREADABLE",
                confidence=raw.get("confidence"),
                captured_at=captured_at,
                track_id=live.get("track_id"),
                box=[x1, y1, x2, y2],
                crop=None,
            )
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                scale = min(1.0, 640 / crop.shape[1], 240 / crop.shape[0])
                if scale < 1:
                    crop = cv2.resize(
                        crop,
                        (
                            max(1, round(crop.shape[1] * scale)),
                            max(1, round(crop.shape[0] * scale)),
                        ),
                    )
                ok, encoded = cv2.imencode(
                    ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                if ok:
                    reading["crop"] = (
                        "data:image/jpeg;base64,"
                        + base64.b64encode(encoded).decode("ascii")
                    )
            self.readings["last_reading"] = reading
        if time.monotonic() - self.last_reading_write >= 0.5:
            self._write_readings()

    def _write_readings(self):
        self.readings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            str(self.readings_path) + "." + uuid.uuid4().hex + ".tmp"
        )
        temporary.write_text(
            json.dumps(self.readings, allow_nan=False), encoding="utf-8"
        )
        try:
            for attempt in range(30):
                try:
                    os.replace(temporary, self.readings_path)
                    break
                except PermissionError:
                    if attempt == 29:
                        raise
                    time.sleep(0.02)
        finally:
            temporary.unlink(missing_ok=True)
        self.last_reading_write = time.monotonic()

    def record_timings(self, timings):
        with self.lock:
            self.statistics.add(timings)

    def set_inference_error(self, code):
        with self.lock:
            self.inference_error = code

    def update_results(
        self,
        plates,
        fps,
        processing_ms,
        captured_at,
        input_jpeg_ms=0.0,
        inference_queue_ms=0.0,
        inference_total_ms=0.0,
        frame_id=None,
    ):
        with self.lock:
            self.inference_error = None
            self.results = (
                plates,
                fps,
                processing_ms,
                captured_at,
                input_jpeg_ms,
                inference_queue_ms,
                inference_total_ms,
                frame_id,
            )

    def submit(
        self,
        frame,
        captured_at,
        capture_fps=0.0,
        capture_read_ms=0.0,
        dropped_capture=0,
        frame_id=None,
    ):
        # Frame ownership stays immutable; never resize/draw on the ANPR input.
        with self.lock:
            if self.stop.is_set():
                return
            if self.pending is not None:
                self.dropped += 1
            self.input_frames += 1
            self.statistics.capture_fps = capture_fps
            self.statistics.dropped_capture = dropped_capture
            self.pending = (
                frame,
                captured_at,
                capture_fps,
                capture_read_ms,
                dropped_capture,
                time.monotonic(),
                frame_id,
                self.results,
            )
        self.wake.set()

    def _run(self):
        import cv2

        next_encode = 0.0
        completed = deque(maxlen=60)
        write_ms = 0.0
        while not self.stop.is_set():
            self.wake.wait(0.1)
            if self.stop.wait(max(0.0, next_encode - time.monotonic())):
                break
            with self.lock:
                item, self.pending = self.pending, None
                results = self.results
                inference_error = self.inference_error
                dropped = self.dropped
                self.wake.clear()
            if item is None:
                continue
            (
                frame,
                captured,
                capture_fps,
                capture_ms,
                dropped_capture,
                received,
                capture_frame_id,
                submitted_results,
            ) = item
            now = time.monotonic()
            next_encode = now + 1 / self.fps
            if now - captured > STALE_SECONDS:
                continue
            # Freeze an already matched pair. A later inference completion
            # must not replace metadata for an older pending image.
            if (
                capture_frame_id is not None
                and submitted_results[-1] == capture_frame_id
            ):
                results = submitted_results
            (
                plates,
                fps,
                ms,
                result_capture,
                input_jpeg_ms,
                inference_queue_ms,
                inference_total_ms,
                result_frame_id,
            ) = results
            overlay_age = captured - result_capture
            matched_id = (
                capture_frame_id is None and result_frame_id is None
            ) or (
                capture_frame_id is not None
                and capture_frame_id == result_frame_id
            )
            recognized = matched_id and 0 <= overlay_age <= self.overlay_age
            plates = plates if recognized else []
            try:
                h, w = frame.shape[:2]
                scale = min(1.0, self.width / w)
                # resize returns an independent array, including scale=1.
                started = time.perf_counter()
                image = cv2.resize(frame, (round(w * scale), round(h * scale)))
                plates = [
                    dict(p, crop=plate_crop(frame, p["box"])) for p in plates
                ]
                for plate in plates:
                    x1, y1, x2, y2 = [round(v * scale) for v in plate["box"]]
                    color = (
                        (60, 220, 90) if plate["confirmed"] else (0, 190, 255)
                    )
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    state = plate.get("status") or (
                        "confirmed" if plate["confirmed"] else "pending"
                    )
                    label = (
                        f"{plate['text'] or 'UNREADABLE'} "
                        f"{plate['confidence']:.0%} "
                        f"{state}"
                    )
                    cv2.putText(
                        image,
                        label,
                        (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
                capture_label = (
                    f"{capture_fps:.1f}" if capture_fps is not None else "n/a"
                )
                cv2.putText(
                    image,
                    f"Capture {capture_label} FPS | ANPR {fps:.1f} FPS",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                render_ms = (time.perf_counter() - started) * 1000
                encode_started = time.perf_counter()
                ok, encoded = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                jpeg_ms = (time.perf_counter() - encode_started) * 1000
                if not ok:
                    continue
                finished = time.monotonic()
                preview_fps = (
                    (len(completed) - 1) / (completed[-1] - completed[0])
                    if len(completed) > 1
                    else 0.0
                )
                jpeg = encoded.tobytes()
                meta = dict(
                    session_id=self.readings.get("session_id"),
                    timestamp=time.time() - (finished - captured),
                    frame_id=capture_frame_id,
                    result_frame_id=result_frame_id,
                    fps=fps,
                    processing_ms=ms,
                    plates=plates,
                    width=image.shape[1],
                    height=image.shape[0],
                    recognition=recognized,
                    preview_fps=preview_fps,
                    target_fps=self.fps,
                    capture_fps=capture_fps,
                    capture_read_ms=capture_ms,
                    capture_ipc_ms=max(0, (received - captured) * 1000),
                    inference_age_ms=max(0, (finished - result_capture) * 1000)
                    if result_capture
                    else 0,
                    inference_input_jpeg_ms=input_jpeg_ms,
                    jpeg_ms=jpeg_ms,
                    preview_render_ms=render_ms,
                    inference_queue_ms=inference_queue_ms,
                    inference_total_ms=inference_total_ms,
                    preview_wait_ms=(now - received) * 1000,
                    volume_write_ms=write_ms,
                    published_at=time.time(),
                    overlay_age_ms=overlay_age * 1000 if recognized else None,
                    dropped_preview=dropped,
                    dropped_capture=dropped_capture,
                    jpeg_bytes=len(jpeg),
                    inference_error=inference_error,
                    inference_state=(
                        "error"
                        if inference_error
                        else "waiting"
                        if not result_capture
                        else "stale"
                        if finished - result_capture > 2
                        else "running"
                    ),
                )
                data = json.dumps(meta).encode() + b"\n" + jpeg
                if len(data) > MAX_BYTES:
                    continue
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_suffix(".tmp")
                write_started = time.perf_counter()
                temporary.write_bytes(data)
                os.replace(temporary, self.path)
                write_ms = (time.perf_counter() - write_started) * 1000
                completed.append(time.monotonic())
            except Exception:
                # Do not log backend diagnostics or camera configuration.
                continue

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout=2)
        if not self.keep_frame:
            self.path.unlink(missing_ok=True)
        with self.lock:
            report = self.statistics.summary()
            report.update(
                input_frames=self.input_frames,
                dropped_processing=max(
                    0, self.input_frames - self.statistics.frames
                ),
                dropped_preview=self.dropped,
                inference_error=self.inference_error,
                status="worker_error" if self.inference_error else "stopped",
            )
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_readings()
        self.stats_path.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )


class SystemMetrics:
    """At most one background sample/1.5s, with missing hardware reported
    as null."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.last_cpu = None
        self.value = dict(
            cpu_percent=None,
            gpu_percent=None,
            ram_used_bytes=None,
            ram_total_bytes=None,
            sampled_at=None,
            scope=os.name + " API host (whole system)",
        )

    def snapshot(self):
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
            return dict(self.value)

    def _sample_host(self):
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            idle, kernel, user = (
                wintypes.FILETIME(),
                wintypes.FILETIME(),
                wintypes.FILETIME(),
            )
            if not ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            ):
                raise OSError("SystemTimes unavailable")

            def ticks(v):
                return (v.dwHighDateTime << 32) + v.dwLowDateTime

            idle, total = ticks(idle), ticks(kernel) + ticks(user)

            class Memory(ctypes.Structure):
                _fields_ = [
                    ("length", wintypes.DWORD),
                    ("load", wintypes.DWORD),
                ] + [
                    (n, ctypes.c_ulonglong)
                    for n in (
                        "total",
                        "available",
                        "page_total",
                        "page_available",
                        "virtual_total",
                        "virtual_available",
                        "extended",
                    )
                ]

            memory = Memory()
            memory.length = ctypes.sizeof(Memory)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(memory)
            ):
                raise OSError("Memory unavailable")
            used, capacity = memory.total - memory.available, memory.total
        else:
            values = [
                int(x)
                for x in Path("/proc/stat")
                .read_text()
                .splitlines()[0]
                .split()[1:]
            ]
            total = sum(values[:8])
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            mem = {
                line.split(":")[0]: int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
            }
            capacity = mem["MemTotal"]
            used = capacity - mem["MemAvailable"]
        percent = None
        if self.last_cpu and total > self.last_cpu[1]:
            percent = max(
                0.0,
                min(
                    100.0,
                    100
                    * (
                        1
                        - (idle - self.last_cpu[0])
                        / (total - self.last_cpu[1])
                    ),
                ),
            )
        self.last_cpu = (idle, total)
        return dict(
            cpu_percent=percent, ram_used_bytes=used, ram_total_bytes=capacity
        )

    def _run(self):
        import subprocess

        while True:
            started = time.monotonic()
            sample = dict(
                cpu_percent=None,
                gpu_percent=None,
                ram_used_bytes=None,
                ram_total_bytes=None,
            )
            try:
                sample.update(self._sample_host())
            except (OSError, ValueError, KeyError, AttributeError):
                pass
            try:
                kwargs = dict(
                    capture_output=True, text=True, timeout=1.0, check=True
                )
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                output = subprocess.run(
                    [
                        "nvidia-smi",
                        "--id=0",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    **kwargs,
                )
                value = float(output.stdout.strip())
                if math.isfinite(value) and 0 <= value <= 100:
                    sample["gpu_percent"] = value
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            with self.lock:
                self.value.update(sample, sampled_at=time.time())
            time.sleep(max(0.1, 1.5 - (time.monotonic() - started)))


system_metrics = SystemMetrics()
