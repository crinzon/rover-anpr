"""Isolated video capture with bounded operations and reconnect backoff."""

import math
import multiprocessing
import os
import threading
import time
from collections import deque
from urllib.parse import urlsplit


def safe_source(source):
    """Do not include URL userinfo, paths or query secrets in diagnostics."""
    if isinstance(source, int):
        return f"USB {source}"
    try:
        parts = urlsplit(source)
        if parts.scheme:
            return f"{parts.scheme}://***@{parts.hostname or 'camera'}/***"
    except ValueError:
        pass
    return "<video source>"


def _capture(
    connection, source, factory=None, backend="auto", transport="tcp"
):
    # Native FFmpeg/OpenCV diagnostics can contain the original URL.
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 2)
    camera = None
    reader = None
    try:
        if factory is None:
            is_rtsp = str(source).lower().startswith(("rtsp://", "rtsps://"))
            if is_rtsp:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    f"rtsp_transport;{transport}"
                )
            import cv2

            api = (
                cv2.CAP_FFMPEG
                if (backend == "ffmpeg" or is_rtsp)
                else cv2.CAP_ANY
            )
            camera = cv2.VideoCapture(source, api)
        else:
            camera = factory(source)
        connection.send(("opened", bool(camera.isOpened())))
        # Decode continuously even while IPC or ANPR is busy. One immutable
        # frame reference replaces its predecessor; no decoded-frame queue.
        condition = threading.Condition()
        latest = None
        sequence = 0
        failed = False
        done = threading.Event()

        def drain():
            nonlocal latest, sequence, failed
            capture_times = deque(maxlen=60)
            try:
                while not done.is_set():
                    started = time.monotonic()
                    success, frame = camera.read()
                    captured = time.monotonic()
                    with condition:
                        if not success or frame is None:
                            failed = True
                            condition.notify_all()
                            return
                        sequence += 1
                        capture_times.append(captured)
                        fps = (
                            (len(capture_times) - 1)
                            / (captured - capture_times[0])
                            if len(capture_times) > 1
                            else 0.0
                        )
                        latest = (
                            frame,
                            captured,
                            (captured - started) * 1000,
                            fps,
                        )
                        condition.notify_all()
            except Exception:
                with condition:
                    failed = True
                    condition.notify_all()

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        delivered = 0
        while connection.recv() == "read":
            with condition:
                condition.wait_for(lambda: sequence > delivered or failed)
                if sequence <= delivered:
                    connection.send(("frame", None))
                    break
                snapshot = (*latest, max(0, sequence - delivered - 1))
                delivered = sequence
            connection.send(("frame", snapshot))
        done.set()
        reader.join(0.1)
    except Exception:
        # Never forward backend exceptions containing credentials.
        pass
    finally:
        # Never release VideoCapture concurrently with a native read. The
        # owning process is reaped by VideoStream if the read is blocked.
        if camera is not None and (reader is None or not reader.is_alive()):
            camera.release()
        connection.close()


class VideoStream:
    def __init__(
        self,
        source,
        initial=0.5,
        maximum=10.0,
        timeout=5.0,
        factory=None,
        logger=print,
        backend="auto",
        transport="tcp",
    ):
        values = (initial, maximum, timeout)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError(
                "Video delays and timeout must be finite and positive"
            )
        if maximum < initial:
            raise ValueError(
                "Maximum reconnect delay must be >= initial delay"
            )
        if backend not in ("auto", "ffmpeg") or transport not in (
            "tcp",
            "udp",
        ):
            raise ValueError("Invalid video backend or RTSP transport")
        self.backend, self.transport = backend, transport
        self.last_frame_timestamp = 0.0
        self.capture_read_ms = self.capture_fps = 0.0
        self.dropped_frames = 0
        self.source = source
        self.initial, self.maximum, self.timeout = values
        self.delay = initial
        self.factory, self.logger = factory, logger
        self.stop = threading.Event()
        self.process = self.connection = None
        self.retry = False
        self.context = multiprocessing.get_context("spawn")

    def _receive(self):
        deadline = time.monotonic() + self.timeout
        while not self.stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if self.connection.poll(min(0.1, remaining)):
                return self.connection.recv()
        raise TimeoutError

    def _disconnect(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            self.process.join(0.1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(0.5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(0.5)
            self.process = None

    def read(self):
        if self.stop.is_set():
            return False, None
        if self.retry:
            if self.stop.wait(self.delay):
                return False, None
            self.delay = min(self.maximum, self.delay * 2)
        try:
            if self.process is None:
                parent, child = self.context.Pipe()
                self.connection = parent
                self.process = self.context.Process(
                    target=_capture,
                    args=(
                        child,
                        self.source,
                        self.factory,
                        self.backend,
                        self.transport,
                    ),
                    daemon=True,
                )
                self.process.start()
                child.close()
                if self._receive() != ("opened", True):
                    raise OSError
            self.connection.send("read")
            kind, snapshot = self._receive()
            if kind != "frame" or snapshot is None:
                raise OSError
            (
                frame,
                self.last_frame_timestamp,
                self.capture_read_ms,
                self.capture_fps,
                dropped,
            ) = snapshot
            self.dropped_frames += dropped
            self.retry = False
            self.delay = self.initial
            return True, frame
        except (OSError, EOFError, TimeoutError):
            self._disconnect()
            self.retry = True
            if self.stop.is_set():
                return False, None
            self.logger(
                f"Video unavailable: {safe_source(self.source)}; "
                f"retry in {self.delay:g}s"
            )
            return False, None

    def release(self):
        self.stop.set()
        self._disconnect()
