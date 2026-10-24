"""Exercise the real worker without importing camera's hardware entrypoint."""

import ast
import threading
import time
import types
import unittest
from collections import deque
from pathlib import Path


class CameraFramesTest(unittest.TestCase):
    def run_worker(
        self, encode_ok=True, inference_error=False, detection=False
    ):
        source = Path(__file__).resolve().parents[1] / "camera.py"
        tree = ast.parse(source.read_text())
        worker = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "recognize"
        )
        condition = threading.Condition()
        calls = []
        attempted = threading.Event()
        previews = []

        class Frame:
            shape = (100, 200, 3)

            def copy(self):
                return self

        def encode(*args):
            calls.append("encode")
            attempted.set()
            return encode_ok, types.SimpleNamespace(tobytes=lambda: b"frame")

        def pipeline(payload):
            if inference_error:
                raise ValueError("synthetic failure")
            plates = (
                [
                    dict(
                        box=[1, 2, 30, 20],
                        text="",
                        confidence=0.2,
                        readable=False,
                    )
                ]
                if detection
                else []
            )
            return {"plates": plates, "timing_ms": {"total": 0}}

        state = dict(
            running=True,
            replay=None,
            latest_media_timestamp=0.0,
            deque=deque,
            normalize_plate=lambda value: value,
            ocr_threshold=0.85,
            diagnostics=types.SimpleNamespace(
                due=lambda: False, error=lambda *a: None
            ),
            preview=types.SimpleNamespace(
                set_inference_error=lambda *a: None,
                update_results=lambda *a, **k: previews.append(a),
            ),
            frame_ready=condition,
            lock=condition,
            latest_frame=None,
            latest_frame_id=0,
            latest_frame_timestamp=0.0,
            results=[],
            result_time=0,
            next_track_id=1,
            tracks={},
            time=time,
            cv2=types.SimpleNamespace(imencode=encode),
            pipeline=pipeline,
        )
        worker.body[0] = ast.copy_location(
            ast.Global(names=worker.body[0].names), worker.body[0]
        )
        exec(
            compile(
                ast.Module(body=[worker], type_ignores=[]), str(source), "exec"
            ),
            state,
        )
        thread = threading.Thread(target=state["recognize"])
        thread.start()
        try:
            self.assertFalse(attempted.wait(0.03))
            for frame_id in (1, 2, 5):
                attempted.clear()
                with condition:
                    state.update(
                        latest_frame=Frame(),
                        latest_frame_id=frame_id,
                        latest_frame_timestamp=time.monotonic(),
                    )
                    condition.notify_all()
                self.assertTrue(attempted.wait(1))
                # Spurious notifications must not retry the same frame.
                with condition:
                    condition.notify_all()
                time.sleep(0.15)
                self.assertEqual(len(calls), (1, 2, 5).index(frame_id) + 1)
        finally:
            with condition:
                state["running"] = False
                condition.notify_all()
            thread.join(1)
        self.assertFalse(thread.is_alive())
        if encode_ok and not inference_error:
            self.assertEqual(len(previews), 3)
            for preview in previews:
                self.assertEqual(len(preview[0]), 1 if detection else 0)
                if detection:
                    self.assertFalse(preview[0][0]["confirmed"])

    def test_unreadable_detection_still_has_live_box(self):
        self.run_worker(detection=True)

    def test_each_frame_once(self):
        self.run_worker()

    def test_encoding_failure_does_not_retry(self):
        self.run_worker(encode_ok=False)

    def test_inference_failure_does_not_retry(self):
        self.run_worker(inference_error=True)


if __name__ == "__main__":
    unittest.main()
