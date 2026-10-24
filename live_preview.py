"""Bounded preview: latest frame, asynchronous JPEG, atomic transport."""

import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "captures" / ".live-preview"
MAX_BYTES = 8_000_000
STALE_SECONDS = 2.0
PUBLIC_FIELDS = (
    "timestamp",
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


class PreviewReader:
    """Cache each atomic file version once; compute age on every request."""

    def __init__(self, path=DEFAULT_PATH):
        self.path = path
        self.lock = threading.Lock()
        self.version = None
        self.cached = (None, {})

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
                        self.version = version
                jpeg, meta = self.cached
                age = time.time() - meta["timestamp"]
                if not 0 <= age < STALE_SECONDS:
                    return None, {}
                return jpeg, dict(meta, frame_age_ms=age * 1000)
            except (OSError, ValueError, KeyError, TypeError):
                self.cached = (None, {})
                self.version = None
                return None, {}


_reader = PreviewReader()


def read_preview(path=None):
    return _reader.read() if path is None else PreviewReader(path).read()


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
        self.path = path
        self.lock = threading.Lock()
        self.pending = None
        self.results = ([], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.inference_error = None
        self.dropped = 0
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

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
            )

    def submit(
        self,
        frame,
        captured_at,
        capture_fps=0.0,
        capture_read_ms=0.0,
        dropped_capture=0,
    ):
        # Frame ownership stays immutable; never resize/draw on the ANPR input.
        with self.lock:
            if self.stop.is_set():
                return
            if self.pending is not None:
                self.dropped += 1
            self.pending = (
                frame,
                captured_at,
                capture_fps,
                capture_read_ms,
                dropped_capture,
                time.monotonic(),
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
            ) = item
            now = time.monotonic()
            next_encode = now + 1 / self.fps
            if now - captured > STALE_SECONDS:
                continue
            (
                plates,
                fps,
                ms,
                result_capture,
                input_jpeg_ms,
                inference_queue_ms,
                inference_total_ms,
            ) = results
            overlay_age = captured - result_capture
            recognized = 0 <= overlay_age <= self.overlay_age
            plates = plates if recognized else []
            try:
                h, w = frame.shape[:2]
                scale = min(1.0, self.width / w)
                # resize returns an independent array, including scale=1.
                started = time.perf_counter()
                image = cv2.resize(frame, (round(w * scale), round(h * scale)))
                for plate in plates:
                    x1, y1, x2, y2 = [round(v * scale) for v in plate["box"]]
                    color = (
                        (60, 220, 90) if plate["confirmed"] else (0, 190, 255)
                    )
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    label = (
                        f"{plate['text']} {plate['confidence']:.0%} "
                        f"{'confirmed' if plate['confirmed'] else 'pending'}"
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
                    timestamp=time.time() - (finished - captured),
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
                self.path.parent.mkdir(exist_ok=True)
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
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout=2)
        self.path.unlink(missing_ok=True)
