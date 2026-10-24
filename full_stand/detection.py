from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms as nms_utils
from ultralytics.utils import ops


class PlateDetector:
    def __init__(
        self,
        weights: Path,
        device: torch.device,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
    ) -> None:
        source: YOLO = YOLO(str(weights))
        self.names: Dict[int, str] = dict(source.names)
        self.model = source.model.fuse().eval().to(device)
        self.parameters: int = sum(p.numel() for p in self.model.parameters())
        self.device: torch.device = device
        self.imgsz: int = imgsz
        self.conf: float = conf
        self.iou: float = iou
        self.max_det: int = max_det
        self.letterbox: LetterBox = LetterBox(
            (imgsz, imgsz), auto=True, stride=32
        )

    def _prepare(self, image: np.ndarray) -> Tensor:
        padded: np.ndarray = self.letterbox(image=image)
        chw: np.ndarray = np.ascontiguousarray(
            padded[..., ::-1].transpose(2, 0, 1)
        )
        return (
            torch.from_numpy(chw)
            .to(self.device)
            .float()
            .div_(255.0)
            .unsqueeze(0)
        )

    @torch.no_grad()
    def __call__(self, image: np.ndarray, diagnostics: bool = False):
        tensor: Tensor = self._prepare(image)
        produced = self.model(tensor)
        predictions: Tensor = (
            produced[0] if isinstance(produced, (list, tuple)) else produced
        )
        # Snapshot proposals before confidence/NMS; reuse this forward pass.
        raw = None
        if diagnostics:
            if predictions.shape[-1] == 6:
                proposals = predictions[0].float()
                raw_boxes = proposals[:, :4].clone()
                raw_scores = proposals[:, 4]
                raw_classes = proposals[:, 5]
                layout = "end2end_xyxy"
            else:
                proposals = predictions[0].transpose(0, 1)
                raw_boxes = ops.xywh2xyxy(proposals[:, :4])
                raw_scores, raw_classes = proposals[
                    :, 4 : 4 + len(self.names)
                ].max(1)
                layout = "dense_xywh"
            raw_boxes = ops.scale_boxes(
                tuple(tensor.shape[2:]), raw_boxes, image.shape[:2]
            )
            raw = (
                torch.cat(
                    (raw_boxes, raw_scores[:, None], raw_classes[:, None]),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
        result = nms_utils.non_max_suppression(
            predictions.clone() if diagnostics else predictions,
            self.conf,
            self.iou,
            None,
            False,
            max_det=self.max_det,
            nc=0,
            end2end=False,
            rotated=False,
            return_idxs=diagnostics,
        )
        detections = result[0][0] if diagnostics else result[0]
        kept = result[1][0].cpu().tolist() if diagnostics else []
        if not detections.shape[0]:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros(0, dtype=np.float32)
        else:
            boxes = (
                ops.scale_boxes(
                    tuple(tensor.shape[2:]), detections[:, :4], image.shape[:2]
                )
                .cpu()
                .numpy()
            )
            scores = detections[:, 4].float().cpu().numpy()
        if not diagnostics:
            return boxes, scores
        selected = set(kept)
        rows = []
        for index, row in enumerate(raw):
            finite = bool(np.isfinite(row).all())
            reason = (
                "invalid_prediction"
                if not finite
                else "detector_confidence"
                if row[4] <= self.conf
                else "nms_or_max_detections"
                if index not in selected
                else None
            )
            rows.append(
                dict(
                    id=index,
                    box=row[:4].tolist() if finite else None,
                    confidence=float(row[4]) if finite else None,
                    class_id=int(row[5]) if finite else None,
                    reason=reason,
                    selected=index in selected,
                )
            )
        return (
            boxes,
            scores,
            dict(
                raw_proposals=rows,
                kept_ids=kept,
                model_output_shape=list(predictions.shape),
                layout=layout,
                detector_input_shape=list(tensor.shape[2:]),
            ),
        )

    def warmup(self, image: Optional[np.ndarray] = None) -> None:
        canvas: np.ndarray = (
            image
            if image is not None
            else np.full(
                (self.imgsz, self.imgsz, 3),
                114,
                dtype=np.uint8,
            )
        )
        for _ in range(3):
            self(canvas)
