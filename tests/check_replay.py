"""Exercise camera.main replay and EventSender HTTP with a fake pipeline.

Synthetic six-frame MOV; no neural weights, CUDA, production API, or RTSP.
"""

import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera import main
from evaluate_videos import FileReplay

received = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        received.append(
            json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        )
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Pipeline:
    def __init__(self):
        self.calls = 0

    def __call__(self, payload):
        self.calls += 1
        return dict(
            source=dict(width=640, height=480),
            timing_ms=dict(total=1, detect=0.5, recognize=0.5),
            plates=[
                dict(
                    text="A123AA77",
                    raw_text="A123AA77",
                    confidence=0.95,
                    readable=True,
                    valid=True,
                    subtype="type1",
                    detection=0.9,
                    box=[100.0, 100.0, 300.0, 160.0],
                )
            ],
        )


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "synthetic.MOV"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480)
        )
        assert writer.isOpened()
        frame = np.random.default_rng(42).integers(
            0, 256, (480, 640, 3), dtype=np.uint8
        )
        for _ in range(6):
            writer.write(frame)
        writer.release()
        api = f"http://127.0.0.1:{server.server_port}"
        replay = FileReplay(
            path, root / "artifacts", "unit-test", api, root / "captures"
        )
        pipeline = Pipeline()
        with patch("camera.StandConfig.validate", return_value=None):
            main(
                dict(
                    DEVICE="cpu", SHOW_WINDOW="false", API_URL=api + "/plates"
                ),
                replay=replay,
                pipeline_override=pipeline,
            )
        report = replay.finish(1.0)
        assert pipeline.calls == 6
        assert report["processed_frames"] == report["decoded_frames"] == 6
        assert not report["errors"], report["errors"]
        assert report["confirmed_observations"] == 4
        assert report["api"]["acknowledged"] == len(received) == 4
        assert report["confirmed_tracks"][0]["plates"] == ["A123AA77"]
        assert replay.frames[-1]["video_time_s"] < 0.3
        assert (root / "artifacts" / "frames.jsonl").is_file()
        print(
            (
                "PASS: six original-worker frames, video-time tracking, "
                "four acknowledged API events"
            )
        )
finally:
    server.shutdown()
    server.server_close()
