"""Manual webcam test: Windows owns camera/UI; WSL runs existing YOLO + OCR."""

import argparse
import csv
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import cv2

from evaluate_videos import FileReplay

SOURCE = "http://172.22.192.1:8765/video"
RESULT_PORT = 8766
TITLE = "Rover ANPR - Q to quit"


def reading(plate):
    from plate_evidence import observation

    raw = plate.get("text", "")
    if not plate.get("readable") or not raw:
        return "NO OCR", plate.get("subtype", "unknown"), "UNREADABLE"
    evidence = observation(plate)
    if evidence is None:
        return raw, plate.get("subtype", "unknown"), "RAW_UNVERIFIED"
    text, subtype = evidence
    return text, subtype, "PARTIAL" if "#" in text else "READ"


class CsvObservations:
    """Compress consecutive equal OCR results, not objects or tracks."""

    fields = [
        "mode",
        "time",
        "last_time",
        "source_session",
        "first_frame",
        "last_frame",
        "observations",
        "plate_index",
        "track_id",
        "text",
        "ocr_text",
        "raw_text",
        "subtype",
        "confidence",
        "confidence_min",
        "confidence_max",
        "confidence_mean",
        "status",
        "filter_result",
        "filter_accepted",
    ]

    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=False)
        self.files, self.writers = {}, {}
        self.pending = None
        try:
            for mode in ("NORMAL", "OCCLUDED"):
                handle = (directory / (mode + ".csv")).open(
                    "x", newline="", encoding="utf-8-sig"
                )
                self.files[mode] = handle
                self.writers[mode] = csv.DictWriter(
                    handle, fieldnames=self.fields
                )
                self.writers[mode].writeheader()
                handle.flush()
        except BaseException:
            for handle in self.files.values():
                handle.close()
            raise

    def add(self, packet):
        mode = packet["mode"]
        plates = packet["plates"]
        signature = (
            mode,
            packet["source_session"],
            tuple(
                (
                    p.get("track_id"),
                    p.get("text"),
                    p["ocr_text"],
                    p["raw_text"],
                    p["subtype"],
                    p["status"],
                    p["filter_result"],
                    p["filter_accepted"],
                )
                for p in plates
            ),
        )
        if self.pending and (
            signature != self.pending["signature"]
            or time.monotonic() - self.pending["started"] >= 1
        ):
            self.flush()
        if self.pending is None:
            self.pending = dict(
                signature=signature,
                started=time.monotonic(),
                first=packet,
                last=packet,
                count=0,
                confidence=[[] for _ in plates],
            )
        pending = self.pending
        pending["last"] = packet
        pending["count"] += 1
        for values, plate in zip(pending["confidence"], plates):
            values.append(plate["confidence"])

    def flush(self):
        pending = self.pending
        if pending is None:
            return
        first, last = pending["first"], pending["last"]
        mode = first["mode"]
        for index, plate in enumerate(last["plates"] or [None]):
            row = dict(
                mode=mode,
                time=first["observed_at"],
                last_time=last["observed_at"],
                source_session=first["source_session"],
                first_frame=first["frame_id"],
                last_frame=last["frame_id"],
                observations=pending["count"],
                plate_index=index + 1 if plate else "",
                status="NO_PLATE_DETECTED",
            )
            if plate:
                values = pending["confidence"][index]
                row.update(
                    {
                        key: plate[key]
                        for key in (
                            "track_id",
                            "text",
                            "ocr_text",
                            "raw_text",
                            "subtype",
                            "confidence",
                            "status",
                            "filter_result",
                            "filter_accepted",
                        )
                    }
                )
                row.update(
                    confidence_min=min(values),
                    confidence_max=max(values),
                    confidence_mean=sum(values) / len(values),
                )
            self.writers[mode].writerow(row)
        self.files[mode].flush()
        self.pending = None

    def tick(self):
        if self.pending and time.monotonic() - self.pending["started"] >= 1:
            self.flush()

    def close(self):
        try:
            self.flush()
        finally:
            for handle in self.files.values():
                handle.close()


class WebcamReplay(FileReplay):
    """Live transport adapter to camera.main, with the same per-frame
    handshake."""

    def __init__(self, source, sender, directory, api_url=None, loader=None):
        from event_sender import EventSender
        from process_video import LocalSender

        self.source, self.sender, self.loader = source, sender, loader
        self.session_id = uuid.uuid4().hex
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.capture_dir = self.directory
        self.image_prefix = "/captures/webcam-tests/" + self.directory.name
        self.sender_factory = EventSender if api_url else LocalSender
        self.stop, self.condition = threading.Event(), threading.Condition()
        self.last_completed = self.frame_count = 0
        self.worker_error = False
        self.media_time = self.last_frame_timestamp = self.capture_read_ms = (
            0.0
        )
        self.capture_fps = None
        self.dropped_frames = 0
        self.capture_session = None
        self.current = self.current_meta = None
        self.first_decoded = None
        self.thresholds = {}
        if api_url:
            from live_preview import PreviewPublisher

            self.preview = PreviewPublisher({})
        else:
            self.preview = SimpleNamespace(
                submit=lambda *a: None,
                update_results=lambda *a: None,
                set_inference_error=lambda *a: None,
                close=lambda: None,
            )
        self.diagnostics = SimpleNamespace(
            due=lambda: False, finish=lambda *a: None, error=lambda *a: None
        )

    def read(self):
        if self.stop.is_set():
            return False, None
        previous = self.current_meta["frame_id"] if self.current_meta else 0
        if self.loader:
            result = self.loader(previous)
            if result is None:
                return False, None
            frame, meta = result
        else:
            address = (
                self.source.rsplit("/", 1)[0] + "/frame?after=" + str(previous)
            )
            try:
                with urllib.request.urlopen(address, timeout=5) as response:
                    meta = dict(
                        frame_id=int(response.headers["X-Frame-ID"]),
                        capture_session=response.headers["X-Capture-Session"],
                        mode=response.headers["X-Test-Mode"],
                        captured_at=float(response.headers["X-Captured-At"]),
                        capture_fps=float(response.headers["X-Capture-FPS"])
                        if response.headers.get("X-Capture-FPS")
                        else None,
                        capture_read_ms=float(
                            response.headers["X-Capture-Read-Ms"]
                        )
                        if response.headers.get("X-Capture-Read-Ms")
                        else None,
                        media_time=float(response.headers["X-Media-Time"]),
                    )
                    import numpy as np

                    frame = cv2.imdecode(
                        np.frombuffer(response.read(), np.uint8),
                        cv2.IMREAD_COLOR,
                    )
            except OSError as error:
                if self.frame_count:
                    print(
                        "Windows frame transport stopped:", error, flush=True
                    )
                    return False, None
                raise RuntimeError(
                    "Windows frame transport unavailable: " + str(error)
                ) from error
        if frame is None:
            raise RuntimeError("Invalid Windows JPEG frame")
        if (
            self.capture_session is not None
            and meta["capture_session"] != self.capture_session
        ):
            raise RuntimeError(
                "Windows capture restarted: restart W"
                "SL to reset tracked identities"
            )
        if meta["frame_id"] <= previous or meta["mode"] not in (
            "NORMAL",
            "OCCLUDED",
        ):
            raise RuntimeError("Invalid source frame identity or mode")
        self.capture_fps = meta.get("capture_fps")
        self.capture_read_ms = meta.get("capture_read_ms")
        if previous:
            self.dropped_frames += meta["frame_id"] - previous - 1
        self.capture_session = meta["capture_session"]
        self.current, self.current_meta = frame, meta
        self.media_time = meta["media_time"]
        self.last_frame_timestamp = time.monotonic() - max(
            0, time.time() - meta["captured_at"]
        )
        self.frame_count += 1
        return True, frame

    def record_frame(
        self, frame_id, media_time, output, plates, confirmed, total_ms
    ):
        if frame_id != self.frame_count or media_time != self.media_time:
            raise RuntimeError(
                "Recognition does not belong to current source frame"
            )
        rows = []
        for plate, raw in zip(plates, output["plates"]):
            filtered = reading(raw)
            rows.append(
                dict(
                    camera_id=plate.get("camera_id"),
                    observation_id=plate.get("observation_id"),
                    text=plate.get("merged_text", raw["text"]) or "NO OCR",
                    ocr_text=raw["text"],
                    raw_text=raw.get("raw_text", raw["text"]),
                    subtype=plate.get("subtype", raw["subtype"]),
                    status=plate.get("recognition_status")
                    or (
                        "UNREADABLE"
                        if not raw["readable"]
                        else "RAW_UNVERIFIED"
                    ),
                    track_id=plate.get("track_id"),
                    confidence=raw["confidence"],
                    box=plate["box"],
                    position=plate.get("position"),
                    filter_result=filtered[0]
                    if filtered[2] in ("READ", "PARTIAL")
                    else "",
                    filter_accepted=filtered[2] in ("READ", "PARTIAL"),
                )
            )
        meta = self.current_meta
        packet = dict(
            width=self.current.shape[1],
            height=self.current.shape[0],
            plates=rows,
            frame_id=meta["frame_id"],
            mode=meta["mode"],
            capture_session=meta["capture_session"],
            source_session=self.session_id,
            inference_frame_id=frame_id,
            observed_at=datetime.fromtimestamp(
                meta["captured_at"], timezone.utc
            ).isoformat(),
        )
        self.sender(packet)
        with self.condition:
            self.last_completed = frame_id
            self.condition.notify_all()

    def record_error(self, *args):
        self.fail_worker()

    def attach_sender(self, sender):
        pass

    def record_submission(self, *args):
        pass

    def drain_sender(self, sender):
        deadline = time.monotonic() + 3
        while (
            getattr(getattr(sender, "queue", None), "unfinished_tasks", 0)
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)

    def release(self):
        self.stop.set()


def run_wsl(source, api_url=None, device="cpu"):
    from camera import main as camera_main

    host = urlsplit(source).hostname
    if not host:
        raise ValueError("MJPEG URL has no hostname")
    if api_url:
        with urllib.request.urlopen(
            api_url.rsplit("/", 1)[0] + "/health", timeout=3
        ) as response:
            if response.status != 200:
                raise RuntimeError("API health check failed")
    directory = (
        Path(__file__).resolve().parent
        / "captures"
        / "webcam-tests"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:

        def send(packet):
            sender.sendto(
                json.dumps(packet, allow_nan=False).encode(),
                (host, RESULT_PORT),
            )

        replay = WebcamReplay(source, send, directory, api_url=api_url)
        print(
            "WSL: production camera.main tracker + plate_evidence; no GUI.",
            flush=True,
        )
        camera_main(
            dict(
                DEVICE=device,
                SHOW_WINDOW="false",
                CAMERA_ID="laptop_webcam",
                API_URL=api_url or "",
            ),
            replay=replay,
        )
        if replay.worker_error:
            raise RuntimeError("Production recognition worker failed")


def run_windows():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Condition, Event, Thread

    if sys.platform != "win32":
        raise RuntimeError("--windows must run in Windows Python, not WSL")
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    condition, stop = Condition(), Event()
    latest = [0, None, 0.0, "NORMAL", 0.0]
    frames = OrderedDict()
    capture_session = uuid.uuid4().hex
    capture_started = time.monotonic()
    capture_times = deque(maxlen=60)
    capture_fps = capture_read_ms = None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path.startswith("/frame?after="):
                try:
                    after = int(self.path.split("=", 1)[1])
                except ValueError:
                    self.send_error(400)
                    return
                with condition:
                    condition.wait_for(
                        lambda: latest[0] > after or stop.is_set(), timeout=4
                    )
                    number, jpeg, captured, test_mode, media_time = latest
                if stop.is_set() or number <= after or jpeg is None:
                    self.send_error(503)
                    return
                self.send_response(200)
                for name, value in {
                    "Content-Type": "image/jpeg",
                    "Content-Length": len(jpeg),
                    "X-Frame-ID": number,
                    "X-Capture-Session": capture_session,
                    "X-Captured-At": captured,
                    "X-Test-Mode": test_mode,
                    "X-Media-Time": media_time,
                    "X-Capture-FPS": capture_fps
                    if capture_fps is not None
                    else "",
                    "X-Capture-Read-Ms": capture_read_ms
                    if capture_read_ms is not None
                    else "",
                }.items():
                    self.send_header(name, str(value))
                self.end_headers()
                try:
                    self.wfile.write(jpeg)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if self.path != "/video":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            previous = 0
            try:
                while not stop.is_set():
                    with condition:
                        condition.wait_for(
                            lambda: latest[0] != previous or stop.is_set(),
                            timeout=0.5,
                        )
                        if stop.is_set():
                            return
                        number, jpeg = latest[:2]
                    if jpeg is None or number == previous:
                        continue
                    previous = number
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nC"
                        b"ontent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
            ):
                pass

    server = None
    logger = None
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if not camera.isOpened():
            raise RuntimeError(
                "Camera unavailable; stop old webcam_stream.py first"
            )
        receiver.bind(("0.0.0.0", RESULT_PORT))
        receiver.setblocking(False)
        server = ThreadingHTTPServer(("0.0.0.0", 8765), Handler)
        server.daemon_threads = True
        Thread(target=server.serve_forever, daemon=True).start()
        directory = (
            Path(__file__).resolve().parent
            / "results"
            / "webcam-tests"
            / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        logger = CsvObservations(directory)
        print("Windows: 1 NORMAL / 2 OCCLUDED / Q save and quit.", flush=True)
        print("CSV:", directory, flush=True)
        result, received = None, 0.0
        mode = "NORMAL"
        while True:
            read_started = time.perf_counter()
            ok, frame = camera.read()
            capture_read_ms = (time.perf_counter() - read_started) * 1000
            if not ok or frame is None:
                raise RuntimeError("Windows camera frame unavailable")
            captured_at = time.time()
            capture_times.append(time.monotonic())
            if len(capture_times) > 1:
                capture_fps = (len(capture_times) - 1) / (
                    capture_times[-1] - capture_times[0]
                )
            media_time = time.monotonic() - capture_started
            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                raise RuntimeError("Cannot encode MJPEG frame")
            with condition:
                number = latest[0] + 1
                latest[:] = [
                    number,
                    jpeg.tobytes(),
                    captured_at,
                    mode,
                    media_time,
                ]
                frames[number] = frame
                while len(frames) > 64:
                    frames.popitem(last=False)
                condition.notify_all()
            while True:
                try:
                    data, address = receiver.recvfrom(65535)
                except BlockingIOError:
                    break
                try:
                    candidate = json.loads(data)
                    if (
                        isinstance(candidate, dict)
                        and candidate.get("width", 0) > 0
                        and candidate.get("height", 0) > 0
                        and isinstance(candidate.get("plates"), list)
                    ):
                        if candidate.get("mode") not in ("NORMAL", "OCCLUDED"):
                            continue
                        if not all(
                            key in candidate
                            for key in (
                                "frame_id",
                                "observed_at",
                                "source_session",
                            )
                        ):
                            continue
                        if candidate.get("capture_session") != capture_session:
                            continue
                        if (
                            result
                            and candidate["frame_id"] <= result["frame_id"]
                        ):
                            continue
                        logger.add(candidate)
                        result, received, _peer = (
                            candidate,
                            time.monotonic(),
                            address,
                        )
                except (ValueError, TypeError):
                    continue
            logger.tick()
            view, labels = frame.copy(), []
            if (
                result is None
                or time.monotonic() - received > 1.5
                or result["frame_id"] not in frames
            ):
                labels = ["WAITING FOR WSL / RESULT STALE"]
            else:
                # Never overlay old detections on a newer live camera frame.
                view = frames[result["frame_id"]].copy()
                sx = frame.shape[1] / result["width"]
                sy = frame.shape[0] / result["height"]
                for index, plate in enumerate(result["plates"], 1):
                    x1, y1, x2, y2 = plate["box"]
                    box = [
                        round(x1 * sx),
                        round(y1 * sy),
                        round(x2 * sx),
                        round(y2 * sy),
                    ]
                    color = (
                        (0, 190, 0)
                        if plate["status"] == "CONFIRMED"
                        else (0, 190, 255)
                    )
                    cv2.rectangle(
                        view, tuple(box[:2]), tuple(box[2:]), color, 2
                    )
                    cv2.putText(
                        view,
                        f"{index}: {plate['text']}",
                        (box[0], max(18, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                    labels.append(
                        f"ID {plate.get('track_id', '-')} {plate['text']} | "
                        f"{plate['subtype']} | "
                        f"{plate['status']} | {plate['confidence']:.2f}"
                    )
                if not labels:
                    labels = ["NO PLATE DETECTED"]
            labels += [
                f"MODE: {mode} | result: "
                f"{result['mode'] if result else 'WAITING'}",
                "1 NORMAL / 2 OCCLUDED / Q save and quit",
                "PENDING / PARTIAL / CONFIRMED | matched source frame",
            ]
            view = cv2.copyMakeBorder(
                view,
                0,
                24 * len(labels) + 8,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(25, 25, 25),
            )
            for row, label in enumerate(labels):
                cv2.putText(
                    view,
                    label,
                    (8, frame.shape[0] + 22 + row * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow(TITLE, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("1"), ord("2")):
                logger.flush()
                mode = "NORMAL" if key == ord("1") else "OCCLUDED"
                print(
                    "Capture mode:",
                    mode,
                    "(latched into each Windows frame)",
                    flush=True,
                )
            elif key in (ord("q"), ord("Q")):
                break
    finally:
        if logger is not None:
            logger.close()
        stop.set()
        with condition:
            condition.notify_all()
        if server is not None:
            server.shutdown()
            server.server_close()
        receiver.close()
        camera.release()
        cv2.destroyAllWindows()  # Windows-only cleanup.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="WSL inference device; no silent fallback",
    )
    parser.add_argument(
        "--windows", action="store_true", help="Windows camera/viewer"
    )
    parser.add_argument(
        "--source", default=SOURCE, help="Windows camera transport URL"
    )
    parser.add_argument(
        "--api-url",
        help="Optional existing API POST /plates URL; enables web preview",
    )
    args = parser.parse_args()
    try:
        run_windows() if args.windows else run_wsl(
            args.source, args.api_url, args.device
        )
    except (
        RuntimeError,
        FileNotFoundError,
        ValueError,
        OSError,
        cv2.error,
    ) as exc:
        print(f"ANPR error: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
