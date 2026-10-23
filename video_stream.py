"""Isolated video capture with bounded operations and reconnect backoff."""

import math
import multiprocessing
import os
import threading
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


def _capture(connection, source, factory=None):
    # Native FFmpeg/OpenCV diagnostics can contain the original URL.
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 2)
    camera = None
    try:
        if factory is None:
            import cv2

            factory = cv2.VideoCapture
        camera = factory(source)
        connection.send(("opened", bool(camera.isOpened())))
        while camera.isOpened() and connection.recv() == "read":
            success, frame = camera.read()
            valid = success and frame is not None
            connection.send(("frame", frame if valid else None))
            if not valid:
                break
    except Exception:
        # Never forward backend exceptions containing credentials.
        pass
    finally:
        if camera is not None:
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
        self.source = source
        self.initial, self.maximum, self.timeout = values
        self.delay = initial
        self.factory, self.logger = factory, logger
        self.stop = threading.Event()
        self.process = self.connection = None
        self.retry = False
        self.context = multiprocessing.get_context("spawn")

    def _receive(self):
        if self.connection.poll(self.timeout):
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
                    args=(child, self.source, self.factory),
                    daemon=True,
                )
                self.process.start()
                child.close()
                if self._receive() != ("opened", True):
                    raise OSError
            self.connection.send("read")
            kind, frame = self._receive()
            if kind != "frame" or frame is None:
                raise OSError
            self.retry = False
            self.delay = self.initial
            return True, frame
        except (OSError, EOFError, TimeoutError):
            self._disconnect()
            self.retry = True
            self.logger(
                f"Video unavailable: {safe_source(self.source)}; "
                f"retry in {self.delay:g}s"
            )
            return False, None

    def release(self):
        self.stop.set()
        self._disconnect()
