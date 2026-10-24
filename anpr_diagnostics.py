"""Opt-in snapshots without camera URLs, model imports or event writes."""

import base64
import json
import math
import os
import time
from pathlib import Path

from plate_validator import is_valid_plate, normalize_plate

DIAGNOSTIC_PATH = (
    Path(__file__).resolve().parent / "captures" / ".anpr-diagnostics"
)
MAX_DIAGNOSTIC_BYTES = 16 * 1024 * 1024
REPORT_FIELDS = (
    "enabled",
    "state",
    "completed_at",
    "expires_after",
    "frame_id",
    "captured_at",
    "source",
    "thresholds",
    "timing_ms",
    "raw_proposals",
    "model_output_shape",
    "detector_input_shape",
    "layout",
    "plates",
    "frame_jpeg",
    "error_code",
    "gate_counts",
)


def describe_plate(
    plate, ocr_threshold, source_width, detector_size, source_height=0
):
    text = normalize_plate(plate["text"])
    valid = is_valid_plate(text)
    reasons = []
    if not plate["readable"]:
        reasons.append("unreadable_subtype")
    if plate["confidence"] < ocr_threshold:
        reasons.append("ocr_confidence")
    if not valid:
        reasons.append("rover_format")
    box = plate["box"]
    scale = detector_size / max(source_width, source_height)
    return dict(
        raw_id=plate.get("raw_id"),
        box=box,
        detection_confidence=plate.get("detection"),
        raw_text=plate.get("raw_text", plate["text"]),
        text=text,
        ocr_confidence=plate["confidence"],
        readable=plate["readable"],
        subtype=plate.get("subtype"),
        model_format_valid=plate.get("valid"),
        rover_format_valid=valid,
        reasons=reasons,
        status="rejected" if reasons else "waiting_confirmation",
        confirmations=0,
        track_id=None,
        confirmed_number=None,
        patch_png=plate.get("patch_png"),
        original_size=[box[2] - box[0], box[3] - box[1]],
        detector_size=[(box[2] - box[0]) * scale, (box[3] - box[1]) * scale],
    )


def set_confirmation(
    item,
    track_id,
    count,
    required,
    confirmed_number,
    history=None,
    change_required=7,
):
    item.update(
        track_id=track_id,
        confirmations=count,
        required=required,
        confirmed_number=confirmed_number,
    )
    if confirmed_number is None:
        item["status"] = "waiting_confirmation"
        item["reasons"] = ["insufficient_confirmations"]
    elif item["text"] != confirmed_number:
        item["confirmations"] = history.count(item["text"]) if history else 0
        item["required"] = change_required
        item["status"] = "waiting_number_change"
        item["reasons"] = ["confirmed_number_mismatch"]
    else:
        item["status"] = "confirmed"
        item["reasons"] = []


class Diagnostics:
    def __init__(self, settings, thresholds, path=DIAGNOSTIC_PATH):
        self.enabled = (
            settings.get("ANPR_DIAGNOSTICS", "false").lower() == "true"
        )
        self.interval = float(settings.get("ANPR_DIAGNOSTICS_INTERVAL", "1"))
        if not math.isfinite(self.interval) or not 0.2 <= self.interval <= 10:
            raise ValueError("Invalid diagnostic interval")
        self.thresholds, self.path = thresholds, path
        self.next_sample = 0.0
        if self.enabled:
            self.error("waiting_for_inference", state="waiting")
        else:
            self.path.unlink(missing_ok=True)

    def due(self):
        now = time.monotonic()
        if not self.enabled or now < self.next_sample:
            return False
        self.next_sample = now + self.interval
        return True

    def _write(self, report):
        try:
            report.update(
                enabled=True,
                completed_at=time.time(),
                expires_after=max(3, self.interval * 3),
            )
            data = json.dumps(report, allow_nan=False).encode()
            if len(data) > MAX_DIAGNOSTIC_BYTES:
                data = json.dumps(
                    dict(
                        enabled=True,
                        state="error",
                        error_code="sample_too_large",
                        completed_at=time.time(),
                        expires_after=3,
                    )
                ).encode()
            self.path.parent.mkdir(exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_bytes(data)
            os.replace(temporary, self.path)
        except (OSError, ValueError, TypeError):
            # Debugging must not interrupt recognition or leak configuration.
            pass

    def error(self, code, state="error"):
        if self.enabled:
            self._write(
                dict(state=state, error_code=code, thresholds=self.thresholds)
            )

    def finish(self, output, plates, frame_id, captured_at, jpeg):
        if not self.enabled:
            return
        raw = output.get("diagnostics", {})
        proposals = raw.get("raw_proposals", [])
        ocr_threshold = self.thresholds.get("ocr_confidence", 0.85)
        counts = dict(
            raw=len(proposals),
            detector_confidence=sum(
                p.get("confidence") is not None
                and p["confidence"]
                > self.thresholds.get("detector_confidence", 0.5)
                for p in proposals
            ),
            after_nms_or_topk=sum(bool(p.get("selected")) for p in proposals),
            ocr=len(plates),
            readable=sum(p["readable"] for p in plates),
            ocr_confidence=sum(
                p["readable"] and p["ocr_confidence"] >= ocr_threshold
                for p in plates
            ),
            rover_format=sum(
                p["readable"]
                and p["ocr_confidence"] >= ocr_threshold
                and p["rover_format_valid"]
                for p in plates
            ),
            confirmed=sum(p["status"] == "confirmed" for p in plates),
        )
        self._write(
            dict(
                state="ok",
                frame_id=frame_id,
                gate_counts=counts,
                captured_at=time.time() - (time.monotonic() - captured_at),
                source=output["source"],
                thresholds=self.thresholds,
                timing_ms=output["timing_ms"],
                plates=plates,
                raw_proposals=raw.get("raw_proposals", []),
                model_output_shape=raw.get("model_output_shape"),
                detector_input_shape=raw.get("detector_input_shape"),
                layout=raw.get("layout"),
                frame_jpeg="data:image/jpeg;base64,"
                + base64.b64encode(jpeg).decode("ascii"),
            )
        )


def read_diagnostics(path=DIAGNOSTIC_PATH):
    try:
        with open(path, "rb") as source:
            data = source.read(MAX_DIAGNOSTIC_BYTES + 1)
        if len(data) > MAX_DIAGNOSTIC_BYTES:
            raise ValueError
        report = json.loads(data)
        age = time.time() - report["completed_at"]
        if not 0 <= age < report["expires_after"]:
            return dict(enabled=True, state="stale", age_ms=age * 1000)
        return dict(
            {k: report[k] for k in REPORT_FIELDS if k in report},
            age_ms=age * 1000,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return dict(enabled=False, state="disabled_or_no_sample")
