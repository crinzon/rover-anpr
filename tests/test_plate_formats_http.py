"""Real pipeline decoding + Camera + HTTP/SQLite; synthetic OCR tensors.

No model weights are loaded. These tests do not measure visual OCR accuracy.
"""

import contextlib
import importlib.util
import io
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from test_api_http import ApiHttpTest
from test_plate_evidence import FORMATS

from current_plates import CurrentPlates


@unittest.skipUnless(
    all(
        importlib.util.find_spec(n)
        for n in ("torch", "cv2", "fastapi", "uvicorn", "ultralytics")
    ),
    "Camera and API runtimes required",
)
class FormatsHttpTest(ApiHttpTest):
    def run_sequence(self, sequence):
        import cv2
        import numpy as np
        import torch
        from utils.alphabet import (
            EOS_IDX,
            MAX_SEQ_LEN,
            PAD_IDX,
            char_to_idx,
            num_classes,
        )

        from camera import main
        from evaluate_videos import FileReplay
        from full_stand.config import StandConfig
        from full_stand.pipeline import PlatePipeline
        from plate_formats import SUBTYPES

        root = Path(self.temp.name) / f"case-{time.monotonic_ns()}"
        root.mkdir()
        self.database.DB_PATH = root / "api.db"
        self.database.init_db()
        self.api.current_plates = CurrentPlates()
        source = root / "fixture.MOV"
        writer = cv2.VideoWriter(
            str(source), cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480)
        )
        self.assertTrue(writer.isOpened())
        canvas = np.random.default_rng(42).integers(
            0, 256, (480, 640, 3), dtype=np.uint8
        )
        for _ in sequence:
            writer.write(canvas)
        writer.release()

        class Predictor:
            calls = 0

            def __call__(self, images):
                text, subtype = sequence[self.calls]
                self.calls += 1
                tokens = torch.full((1, MAX_SEQ_LEN), PAD_IDX)
                ids = [char_to_idx[c] for c in text] + [EOS_IDX]
                tokens[0, : len(ids)] = torch.tensor(ids)
                probs = torch.zeros((1, MAX_SEQ_LEN, num_classes))
                probs.scatter_(2, tokens.unsqueeze(-1), 0.99)
                type_probs = torch.zeros((1, len(SUBTYPES)))
                type_probs[0, SUBTYPES.index(subtype)] = 0.99
                return dict(
                    tokens=tokens,
                    probs=probs,
                    subtype=torch.tensor([SUBTYPES.index(subtype)]),
                    type_probs=type_probs,
                    confidence=torch.tensor([0.98]),
                )

        engine = PlatePipeline.__new__(PlatePipeline)
        engine.config = StandConfig(device="cpu")
        engine.device = torch.device("cpu")
        engine.predictor = Predictor()
        # Two-row fixture passes through a near-square OCR bucket.
        box = (
            [100.0, 100.0, 240.0, 210.0]
            if sequence[0][1] == "type1a"
            else [100.0, 100.0, 300.0, 160.0]
        )
        engine.detector = lambda image: (
            np.array([box], dtype=np.float32),
            np.array([0.99], dtype=np.float32),
        )
        replay = FileReplay(
            source, root / "replay", "contract", self.base, root / "captures"
        )
        original_attach = replay.attach_sender
        original_record = replay.record_frame
        snapshots = []

        def attach(sender):
            original_attach(sender)
            replay.sender = sender

        def record(*args):
            replay.sender.queue.join()
            rows = self.http("/plates?include_partial=true")[1]["plates"]
            snapshots.append(rows)
            original_record(*args)

        replay.attach_sender, replay.record_frame = attach, record
        try:
            with (
                patch("camera.StandConfig.validate"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main(
                    dict(
                        DEVICE="cpu",
                        SHOW_WINDOW="false",
                        API_URL=self.base + "/plates",
                    ),
                    replay=replay,
                    pipeline_override=engine,
                )
        finally:
            replay.jsonl.close()
        self.assertEqual(engine.predictor.calls, len(sequence))
        self.assertFalse(replay.errors)
        self.assertTrue(all(p["success"] for p in replay.deliveries))
        return replay, snapshots

    def test_verified_types_from_ocr_tensors_to_http_and_history(self):
        for subtype, text in FORMATS:
            with self.subTest(subtype=subtype, text=text):
                replay, _ = self.run_sequence([(text, subtype)] * 3)
                rows = self.http("/plates")[1]["plates"]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["plate"], text)
                self.assertEqual(rows[0]["subtype"], subtype)
                self.assertEqual(rows[0]["status"], "CONFIRMED")
                history = self.http("/history")[1]["detections"]
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0]["plate"], text)
                self.assertEqual(history[0]["subtype"], subtype)
                self.assertEqual(history[0]["status"], "CONFIRMED")
                actual = replay.frames[0]["detections"][0]["subtype"]
                self.assertEqual(actual, subtype)

    def test_complementary_partial_reads_refine_same_track(self):
        replay, snapshots = self.run_sequence(
            [("A123##77", "type1")] * 3 + [("####BC77", "type1")] * 3
        )
        self.assertEqual(snapshots[0][0]["plate"], "A123##77")
        self.assertEqual(snapshots[0][0]["status"], "PARTIAL")
        self.assertEqual(snapshots[-1][0]["plate"], "A123BC77")
        self.assertEqual(snapshots[-1][0]["status"], "CONFIRMED")
        self.assertEqual({rows[0]["track_id"] for rows in snapshots}, {1})
        history = self.http("/history?include_partial=true")[1]["detections"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "CONFIRMED")
        self.assertEqual(len(replay.confirmed), 1)

    def test_unknowns_remain_partial_and_hidden_from_legacy(self):
        self.run_sequence([("A123##77", "type1")] * 5)
        self.assertEqual(self.http("/plates")[1]["plates"], [])
        self.assertEqual(self.http("/history")[1]["detections"], [])
        row = self.http("/plates?include_partial=true")[1]["plates"][0]
        self.assertEqual(row["plate"], "A123##77")
        self.assertEqual(row["status"], "PARTIAL")
        history = self.http("/history?include_partial=true")[1]["detections"]
        self.assertEqual(history[0]["subtype"], "type1")
        self.assertEqual(history[0]["status"], "PARTIAL")

    def test_conflicting_ocr_cannot_create_a_confirmed_full_number(self):
        _, snapshots = self.run_sequence(
            [("A123BC77", "type1")] * 2
            + [("A123BX77", "type1")]
            + [("A123BC77", "type1")] * 10
        )
        self.assertEqual(snapshots[2][0]["plate"], "A123B#77")
        self.assertTrue(
            all(
                not rows or rows[0]["status"] != "CONFIRMED"
                for rows in snapshots[:-1]
            )
        )
        self.assertEqual(snapshots[-1][0]["plate"], "A123BC77")
        self.assertEqual(snapshots[-1][0]["status"], "CONFIRMED")

    def test_one_clear_frame_cannot_confirm_missing_characters(self):
        self.run_sequence(
            [("A123##77", "type1")] * 3 + [("A123BC77", "type1")]
        )
        self.assertEqual(self.http("/plates")[1]["plates"], [])
        row = self.http("/plates?include_partial=true")[1]["plates"][0]
        self.assertEqual(row["status"], "PARTIAL")
        self.assertEqual(row["plate"], "A123##77")

    def test_http_rejects_unknowns_claimed_as_confirmed(self):
        event = self.event(
            subtype="type1", status="CONFIRMED", plate="A123##77"
        )
        code, _ = self.http(
            "/plates", dict(session_id=self.session, plates=[event])
        )
        self.assertEqual(code, 422)


if __name__ == "__main__":
    unittest.main()
