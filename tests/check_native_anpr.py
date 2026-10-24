"""Native dependency regressions; synthetic images/tensors, no weights or RTSP.

Run explicitly in the Camera image: python tests/check_native_anpr.py
"""

import ast
import sys
import tempfile
import threading
import time
import types
import unittest
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anpr_diagnostics import describe_plate, set_confirmation
from full_stand.config import StandConfig
from full_stand.detection import PlateDetector
from full_stand.pipeline import PlatePipeline
from plate_validator import (
    confirmed_confidence,
    is_valid_plate,
    normalize_plate,
)


def nested(name, state):
    tree = ast.parse((ROOT / "camera.py").read_text())
    function = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    if isinstance(function.body[0], ast.Nonlocal):
        function.body[0] = ast.copy_location(
            ast.Global(names=function.body[0].names), function.body[0]
        )
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "camera.py", "exec"
        ),
        state,
    )
    return state[name]


def detector(predictions):
    instance = PlateDetector.__new__(PlateDetector)
    instance.names = {0: "plate"}
    instance.device = torch.device("cpu")
    instance.imgsz, instance.conf, instance.iou, instance.max_det = (
        64,
        0.5,
        0.45,
        64,
    )
    instance.letterbox = lambda image: image
    instance.calls = 0

    def model(tensor):
        instance.calls += 1
        return predictions.clone()

    instance.model = model
    return instance


class NativeANPRTest(unittest.TestCase):
    def test_dense_raw_proposals_and_single_forward(self):
        prediction = torch.tensor(
            [
                [
                    [32.0, 10.0, 32.0],
                    [32.0, 10.0, 32.0],
                    [20.0, 8.0, 20.0],
                    [10.0, 8.0, 10.0],
                    [0.6, 0.4, 0.9],
                ]
            ]
        )
        instance = detector(prediction)
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        boxes, scores = instance(image)
        debug_boxes, debug_scores, debug = instance(image, diagnostics=True)
        self.assertEqual(instance.calls, 2)  # one call per invocation
        np.testing.assert_array_equal(boxes, debug_boxes)
        np.testing.assert_array_equal(scores, debug_scores)
        self.assertEqual(len(debug["raw_proposals"]), 3)
        self.assertEqual(
            debug["raw_proposals"][0]["reason"], "nms_or_max_detections"
        )
        self.assertEqual(
            debug["raw_proposals"][1]["reason"], "detector_confidence"
        )
        self.assertIsNone(debug["raw_proposals"][2]["reason"])
        self.assertEqual(debug["kept_ids"], [2])

    def test_end2end_output_and_pipeline_size_gate(self):
        prediction = torch.tensor(
            [
                [
                    [10.0, 10.0, 13.0, 13.0, 0.9, 0.0],
                    [20.0, 20.0, 40.0, 30.0, 0.8, 0.0],
                    [45.0, 45.0, 60.0, 60.0, 0.1, 0.0],
                ]
            ]
        )
        pipeline = PlatePipeline.__new__(PlatePipeline)
        pipeline.config = StandConfig(confidence=0.5, imgsz=64)
        pipeline.detector = detector(prediction)
        pipeline._read = lambda image, boxes, scores: [
            dict(index=i, box=box.tolist(), text="A123AA77")
            for i, box in enumerate(boxes)
        ]
        ok, jpeg = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))
        self.assertTrue(ok)
        ordinary = pipeline(jpeg.tobytes())
        result = pipeline(jpeg.tobytes(), diagnostics=True)
        self.assertEqual(pipeline.detector.calls, 2)
        self.assertEqual(len(result["diagnostics"]["raw_proposals"]), 3)
        self.assertEqual(
            result["diagnostics"]["raw_proposals"][0]["reason"],
            "box_too_small",
        )
        self.assertEqual(result["plates"][0]["raw_id"], 1)
        self.assertEqual(
            ordinary["plates"][0]["text"], result["plates"][0]["text"]
        )
        self.assertEqual(len(ordinary["plates"]), len(result["plates"]))

    def test_real_quality_and_three_frame_confirmation(self):
        condition = threading.Condition()
        events, samples, errors = [], [], []
        frame = np.random.default_rng(42).integers(
            0, 256, (480, 640, 3), dtype=np.uint8
        )
        source_plate = dict(
            box=[100.0, 100.0, 300.0, 160.0],
            text="A123AA77",
            raw_text="A123AA77",
            confidence=0.95,
            detection=0.9,
            readable=True,
            valid=True,
            subtype="type1",
            raw_id=0,
        )
        state = dict(
            running=True,
            replay=None,
            latest_media_timestamp=0.0,
            frame_ready=condition,
            lock=condition,
            latest_frame=None,
            latest_frame_id=0,
            latest_frame_timestamp=0.0,
            results=[],
            result_time=0,
            next_track_id=1,
            tracks={},
            history_size=10,
            min_confirmations=3,
            ocr_threshold=0.85,
            change_confirmations=7,
            track_timeout=1,
            reid_timeout=30,
            visibility_timeout=1,
            match_distance=0.12,
            quality_improvement=0.08,
            cv2=cv2,
            time=time,
            datetime=datetime,
            Counter=Counter,
            deque=deque,
            config=types.SimpleNamespace(imgsz=640),
            normalize_plate=normalize_plate,
            is_valid_plate=is_valid_plate,
            confirmed_confidence=confirmed_confidence,
            describe_plate=describe_plate,
            set_confirmation=set_confirmation,
            session_id="synthetic",
            log_visibility=lambda *a, **k: None,
            match_reid=lambda *a: None,
            send_to_api=lambda items: events.extend(items),
            preview=types.SimpleNamespace(update_results=lambda *a: None),
            diagnostics=types.SimpleNamespace(
                due=lambda: True,
                finish=lambda output, items, *a: samples.append(items),
            ),
            pipeline=lambda payload, **kwargs: dict(
                plates=[dict(source_plate)],
                source=dict(width=640, height=480),
                timing_ms=dict(total=1),
            ),
        )
        quality = nested("plate_quality", state)
        score, sharpness = quality(frame, source_plate["box"], 0.95)
        self.assertGreater(score, 0)
        self.assertGreater(sharpness, 0)
        nested("weighted_choice", state)
        worker = nested("recognize", state)

        def run():
            try:
                worker()
            except Exception as error:
                errors.append(error)

        with tempfile.TemporaryDirectory() as directory:
            state["captures_dir"] = Path(directory)
            thread = threading.Thread(target=run)
            thread.start()
            try:
                for index in range(1, 4):
                    with condition:
                        state.update(
                            latest_frame=frame,
                            latest_frame_id=index,
                            latest_frame_timestamp=time.monotonic(),
                        )
                        condition.notify_all()
                    deadline = time.monotonic() + 2
                    while (
                        len(samples) < index
                        and not errors
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.01)
                    self.assertFalse(errors, errors)
                    self.assertEqual(len(samples), index)
                self.assertEqual(
                    samples[0][0]["reasons"], ["insufficient_confirmations"]
                )
                self.assertEqual(samples[2][0]["status"], "confirmed")
                deadline = time.monotonic() + 1
                while not events and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(events[0]["plate"], "A123AA77")
                self.assertTrue(list(Path(directory).glob("*.jpg")))
            finally:
                with condition:
                    state["running"] = False
                    condition.notify_all()
                thread.join(2)
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main(verbosity=2)
