"""Thread-safe current observations; expiry is independent for each track."""

import copy
import math
import threading
import time

from plate_validator import is_valid_plate


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_plate(plate, received_at):
    if not isinstance(plate, dict):
        raise ValueError("each plate must be an object")
    camera = plate.get("camera_id")
    if camera is not None and (not isinstance(camera, str) or not camera):
        raise ValueError("camera_id must be a nonempty string when supplied")
    oid = plate.get("observation_id")
    if oid is not None and (not isinstance(oid, str) or not oid):
        raise ValueError(
            "observation_id must be a nonempty string when supplied"
        )
    track = plate.get("track_id")
    if (
        isinstance(track, bool)
        or not isinstance(track, (str, int))
        or str(track) == ""
    ):
        raise ValueError("track_id must be a nonempty string or integer")
    if not isinstance(plate.get("plate"), str) or not plate["plate"]:
        raise ValueError("plate must be a nonempty string")
    status = plate.get("status", "CONFIRMED")
    if status not in ("PARTIAL", "PENDING", "CONFIRMED"):
        raise ValueError("invalid recognition status")
    if "#" in plate["plate"] and status != "PARTIAL":
        raise ValueError("unknown characters require PARTIAL status")
    if status == "PARTIAL" and "#" not in plate["plate"]:
        raise ValueError("a complete unconfirmed reading is PENDING")
    subtype = plate.get("subtype")
    if "status" in plate or subtype is not None:
        if not is_valid_plate(plate["plate"], subtype or "unknown"):
            raise ValueError("plate does not match its OCR subtype")
    if "confirmed" in plate and plate["confirmed"] != (status == "CONFIRMED"):
        raise ValueError("confirmed flag contradicts status")
    for field in ("confidence", "quality"):
        if not finite_number(plate.get(field)) or not 0 <= plate[field] <= 1:
            raise ValueError(f"{field} must be a finite number in [0, 1]")
    if "event_id" in plate and (
        not isinstance(plate["event_id"], str) or not plate["event_id"]
    ):
        raise ValueError("event_id must be a nonempty string")
    captured = plate.get("captured_at")
    if captured is not None:
        if not finite_number(captured) or captured <= 0:
            raise ValueError("captured_at must be Unix UTC seconds")
        if captured > received_at + 1.0:
            raise ValueError("captured_at is more than 1s in the future")
    position = plate.get("position")
    if position is not None:
        if not isinstance(position, dict):
            raise ValueError("position must be an object")
        offsets = ("offset_x_px", "offset_y_px")
        if any(k in position for k in offsets):
            if not all(finite_number(position.get(k)) for k in offsets):
                raise ValueError("both pixel offsets must be finite numbers")
        for field in ("x", "y"):
            if field in position and not finite_number(position[field]):
                raise ValueError(f"position.{field} must be finite")


class CurrentPlates:
    def __init__(self, ttl=2.0, wall_clock=None, monotonic_clock=None):
        if not finite_number(ttl) or ttl <= 0:
            raise ValueError("ttl must be positive and finite")
        self.ttl = ttl
        self.wall_clock = wall_clock or (lambda: time.time())
        self.monotonic_clock = monotonic_clock or (lambda: time.monotonic())
        self.lock = threading.RLock()
        self._tracks = {}

    def _prune(self, now):
        for key in list(self._tracks):
            if self._tracks[key]["deadline"] <= now:
                del self._tracks[key]

    def update(self, session_id, plates, persist):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        if not isinstance(plates, list):
            raise ValueError("plates must be an array")
        with self.lock:
            received_at, now = self.wall_clock(), self.monotonic_clock()
            # Validate the complete request before applying any event.
            for plate in plates:
                validate_plate(plate, received_at)
            self._prune(now)
            for plate in plates:
                captured = plate.get("captured_at")
                observed = received_at if captured is None else captured
                remaining = self.ttl - max(0.0, received_at - observed)
                key = (
                    session_id,
                    plate.get("camera_id") or "unspecified",
                    str(plate["track_id"]),
                )
                previous = self._tracks.get(key)
                # Late events are acknowledged but cannot revive a track or
                # overwrite a newer observation in memory/SQLite.
                if remaining <= 0:
                    continue
                if previous and observed <= previous["observed"]:
                    continue
                if not persist(session_id, plate):
                    continue
                payload = copy.deepcopy(plate)
                payload.update(
                    session_id=session_id,
                    captured_at=captured,
                    received_at=received_at,
                    timestamp_source=(
                        "captured_at"
                        if captured is not None
                        else "received_at"
                    ),
                    expires_at=received_at + remaining,
                )
                self._tracks[key] = dict(
                    observed=observed, deadline=now + remaining, plate=payload
                )

    def snapshot(self):
        with self.lock:
            self._prune(self.monotonic_clock())
            return copy.deepcopy([v["plate"] for v in self._tracks.values()])
