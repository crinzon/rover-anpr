"""Model-free integration of real Camera, sequential replay and MP4 codec."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from process_video import process_one


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


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    source = root / "Front" / "sample[0@0].mp4"
    source.parent.mkdir()
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480)
    )
    assert writer.isOpened(), "mp4v codec unavailable"
    for index in range(6):
        writer.write(
            np.random.default_rng(index).integers(
                0, 256, (480, 640, 3), dtype=np.uint8
            )
        )
    writer.release()
    pipeline = Pipeline()
    with patch("full_stand.config.StandConfig.validate"):
        report = process_one(source, root / "result", "cpu", pipeline, cv2)
    assert pipeline.calls == report["processed_frames"] == 6
    assert report["confirmed_numbers"] == ["A123AA77"]
    detailed = json.loads((root / "result" / "report.json").read_text())
    assert detailed["frames"][0]["detections"][0]["status"] != "confirmed"
    assert detailed["frames"][2]["detections"][0]["status"] == "confirmed"
    assert report["output_metadata"]["frames"] == 6
    assert abs(report["output_metadata"]["duration_s"] - 0.6) < 0.01
print("Actual Camera worker + MP4 export: passed")
