import csv
import math
import signal
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2

from anpr_diagnostics import Diagnostics, describe_plate
from event_sender import EventSender
from full_stand.config import StandConfig
from full_stand.pipeline import PlatePipeline
from live_preview import PreviewPublisher
from plate_evidence import (
    identity_compatible,
    observation,
    registry_supported,
    reid_match,
    remember_identity,
    resolve,
    spatial_match,
)
from plate_validator import (
    normalize_plate,
    plate_position,
)
from video_settings import load_env, stream_options
from video_stream import VideoStream


def main(settings=None, replay=None, pipeline_override=None):
    settings = load_env() if settings is None else settings
    video_source = settings.get("VIDEO_SOURCE", "0")
    api_url = settings.get("API_URL", "http://127.0.0.1:8000/plates")
    device = settings.get("DEVICE", "mps")
    show_window = settings.get("SHOW_WINDOW", "true").lower() == "true"

    config = StandConfig(device=device, confidence=0.5)
    config.validate()
    pipeline = (
        PlatePipeline(config)
        if pipeline_override is None
        else pipeline_override
    )
    session_id = replay.session_id if replay is not None else uuid.uuid4().hex
    camera_id = (
        settings.get("CAMERA_ID")
        or getattr(replay, "camera_id", None)
        or "unspecified"
    )
    captures_dir = (
        replay.capture_dir
        if replay is not None
        else Path(__file__).resolve().parent / "captures"
    )
    captures_dir.mkdir(parents=True, exist_ok=True)
    history_size = 10
    min_confirmations = 3
    ocr_threshold = 0.85
    change_confirmations = 7
    track_timeout = 1.0
    # Retain briefly lost identities, but recover only through strict OCR +
    # geometry/motion gates. This is not permission to match by plate alone.
    reid_timeout = float(settings.get("TRACK_REID_TIMEOUT", "3.0"))
    if not math.isfinite(reid_timeout) or reid_timeout < track_timeout:
        raise ValueError("TRACK_REID_TIMEOUT must be finite and >= 1 second")
    match_distance = 0.12
    quality_improvement = 0.08
    visibility_timeout = 1.0  # секунд без обнаружения до события ИСЧЕЗ
    visibility_log = (
        replay.directory / "visibility.csv"
        if replay is not None
        else Path(__file__).resolve().parent / "visibility_log.csv"
    )
    source = int(video_source) if video_source.isdigit() else video_source
    camera = (
        replay
        if replay is not None
        else VideoStream(source, **stream_options(settings))
    )
    lock = threading.Lock()
    frame_ready = threading.Condition(lock)
    latest_frame = None
    latest_frame_id = 0
    latest_frame_timestamp = 0.0
    latest_media_timestamp = 0.0
    results = []
    result_time = 0.0
    running = True

    def request_stop(signum, frame):
        nonlocal running
        running = False
        camera.stop.set()
        if replay is not None:
            replay.interrupted = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    tracks = {}
    next_track_id = 1
    visibility_file = open(visibility_log, "a", newline="", encoding="utf-8")
    visibility_writer = csv.writer(visibility_file)
    if visibility_log.stat().st_size == 0:
        visibility_writer.writerow(
            ["timestamp", "event", "track_id", "plate", "absent_seconds"]
        )
        visibility_file.flush()

    def log_visibility(
        event, track_id, track, absent_seconds=None, observed_plate=None
    ):
        timestamp = (
            datetime.now().astimezone().isoformat(timespec="milliseconds")
        )
        plate = (
            track.get("confirmed_number")
            or get_track_number(track)
            or observed_plate
            or "?"
        )
        seconds_text = (
            "" if absent_seconds is None else f"{absent_seconds:.2f}"
        )
        visibility_writer.writerow(
            [timestamp, event, track_id, plate, seconds_text]
        )
        visibility_file.flush()
        detail = (
            f" | отсутствовал {seconds_text} с"
            if absent_seconds is not None
            else ""
        )
        print(
            f"[{timestamp}] {event}: объект {track_id} | {plate}{detail}",
            flush=True,
        )

    sender_factory = getattr(replay, "sender_factory", EventSender)
    sender = sender_factory(
        api_url,
        capacity=int(settings.get("DELIVERY_QUEUE_SIZE", "256")),
        attempts=int(settings.get("DELIVERY_ATTEMPTS", "5")),
        initial=float(settings.get("DELIVERY_RETRY_INITIAL", "0.5")),
        maximum=float(settings.get("DELIVERY_RETRY_MAX", "10")),
        timeout=float(settings.get("DELIVERY_TIMEOUT", "2")),
    )

    if replay is not None:
        replay.attach_sender(sender)

    def send_to_api(plates):
        for plate in plates:
            queued = sender.submit(session_id, plate)
            if replay is not None:
                replay.record_submission(plate, queued)

    def get_track_number(track):
        if not track["history"]:
            return None
        numbers = [item["plate"] for item in track["history"]]
        return Counter(numbers).most_common(1)[0][0]

    def match_reid(candidate, excluded_tracks, current_time):
        matches = []
        for track_id, track in tracks.items():
            if track_id in excluded_tracks:
                continue
            score = reid_match(
                candidate,
                track,
                current_time,
                reid_timeout,
                match_distance,
                min_confirmations,
            )
            if score is not None:
                matches.append(track_id)
        # Multiple plausible identities are ambiguous even if one is nearer.
        return matches[0] if len(matches) == 1 else None

    def plate_quality(frame, box, confidence):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 6:
            return 0.0, 0.0
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        size_score = min((x2 - x1) / 160.0, 1.0) * min((y2 - y1) / 35.0, 1.0)
        sharpness_score = min(sharpness / 250.0, 1.0)
        quality = confidence * (0.45 * size_score + 0.55 * sharpness_score)
        return round(quality, 4), sharpness

    preview = (
        replay.preview if replay is not None else PreviewPublisher(settings)
    )
    thresholds = dict(
        detector_confidence=config.confidence,
        detector_iou=config.iou,
        detector_imgsz=config.imgsz,
        min_box_side=config.min_box_side,
        max_detections=config.max_detections,
        ocr_confidence=ocr_threshold,
        min_confirmations=min_confirmations,
        recent_window=history_size,
        history_size=history_size,
        change_confirmations=change_confirmations,
        track_timeout=track_timeout,
        reid_timeout=reid_timeout,
        rover_format="OCR subtype patterns; # is unknown",
    )
    diagnostics = (
        replay.diagnostics
        if replay is not None
        else Diagnostics(settings, thresholds)
    )
    if replay is not None:
        replay.thresholds = thresholds

    def recognize():
        nonlocal latest_frame, results, result_time, next_track_id
        last_processed_frame_id = 0
        completed = deque(maxlen=30)
        while running:
            with frame_ready:
                frame_ready.wait_for(
                    lambda: not running
                    or latest_frame_id > last_processed_frame_id
                )
                if not running:
                    break
                frame = latest_frame
                frame_id = latest_frame_id
                frame_timestamp = latest_frame_timestamp
                media_timestamp = latest_media_timestamp
                # Claim before encoding/inference, including failed attempts.
                last_processed_frame_id = frame_id
            # Cross-process/host API timestamps must be wall-clock UTC, not
            # monotonic seconds. Keep capture age, including IPC/queue delay.
            captured_at = time.time() - max(
                0.0, time.monotonic() - frame_timestamp
            )
            inference_queue_ms = (time.monotonic() - frame_timestamp) * 1000
            input_started = time.perf_counter()
            success, buffer = cv2.imencode(".jpg", frame)
            input_jpeg_ms = (time.perf_counter() - input_started) * 1000
            if not success:
                if replay is not None:
                    replay.record_error(
                        frame_id, media_timestamp, "input_encode_failed"
                    )
                continue
            sample_debug = replay is not None or diagnostics.due()
            payload = buffer.tobytes()
            try:
                output = (
                    pipeline(payload, diagnostics=True)
                    if sample_debug and replay is None
                    else pipeline(payload)
                )
                inference_total_ms = (
                    time.perf_counter() - input_started
                ) * 1000
                plates = output["plates"]
                print(f"Обработка: {output['timing_ms']['total']:.0f} мс")
            except Exception:
                preview.set_inference_error("pipeline_failed")
                diagnostics.error("pipeline_failed")
                print("Recognition error: pipeline_failed")
                if replay is not None:
                    replay.record_error(
                        frame_id, media_timestamp, "pipeline_failed"
                    )
                time.sleep(0.1)
                continue
            height, width = frame.shape[:2]
            current_time = (
                media_timestamp if replay is not None else time.monotonic()
            )
            live_plates = [
                dict(
                    box=p["box"],
                    text=normalize_plate(p["text"]),
                    confidence=p["confidence"],
                    confirmed=False,
                    ocr_text=p.get("text", ""),
                    raw_text=p.get("raw_text", ""),
                    subtype=p.get("subtype"),
                    status=p.get("status")
                    or ("PENDING" if p.get("text") else "UNREADABLE"),
                )
                for p in plates
            ]
            diagnostic_plates = (
                [
                    describe_plate(
                        p, ocr_threshold, width, config.imgsz, height
                    )
                    for p in plates
                ]
                if sample_debug
                else []
            )
            candidates = []
            for plate in plates:
                observed = observation(plate, ocr_threshold)
                if observed is None:
                    continue
                number, subtype = observed
                x1, y1, x2, y2 = plate["box"]
                plate_width = x2 - x1
                plate_height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                x = center_x / width
                y = center_y / height
                quality, sharpness = plate_quality(
                    frame, plate["box"], plate["confidence"]
                )
                candidates.append(
                    {
                        "frame_id": frame_id,
                        "camera_id": camera_id,
                        "session_id": session_id,
                        "quality": quality,
                        "sharpness": sharpness,
                        "plate": number,
                        "subtype": subtype,
                        "box": plate["box"],
                        "center": (x, y),
                        "normalized_box": [
                            x1 / width,
                            y1 / height,
                            x2 / width,
                            y2 / height,
                        ],
                        "width": plate_width,
                        "height": plate_height,
                        "position": plate_position(
                            plate["box"], width, height
                        ),
                        "distance_m": None,
                        "confidence": round(plate["confidence"], 3),
                        "character_probabilities": {
                            p["position"]: p["probability"]
                            for p in plate.get("characters", [])
                        },
                    }
                )
                print(
                    f"Номер: {number} | "
                    f"Размер: {plate_width:.0f}x{plate_height:.0f} px | "
                    f"Уверенность: {plate['confidence']:.1%} | "
                    f"Чёткость: {sharpness:.0f} | Качество: {quality:.1%}"
                )
            for track_id in list(tracks):
                track = tracks[track_id]
                gap = current_time - track["last_seen"]
                if track.get("visible", True) and gap > visibility_timeout:
                    track["visible"] = False
                    track["disappeared_at"] = track["last_seen"]
                    log_visibility("ИСЧЕЗ", track_id, track)
                if gap > reid_timeout:
                    del tracks[track_id]
            pairs = []
            for candidate_index, candidate in enumerate(candidates):
                for track_id, track in tracks.items():
                    if track.get("subtype") != candidate["subtype"]:
                        continue
                    if current_time - track["last_seen"] > track_timeout:
                        continue
                    if not identity_compatible(candidate, track):
                        continue
                    # A missing detection/OCR frame is a reacquisition even
                    # inside the active timeout. Do not allow one-character
                    # substitutions to join similar cars across that gap.
                    previous_frame = track["history"][-1]["frame_id"]
                    if previous_frame != frame_id - 1:
                        # Rejected OCR is not a lost detector when the existing
                        # unique geometry bridge
                        # observed this object last
                        # frame.
                        # Require strict visible agreement here; a genuine lost
                        # box still goes through the stricter re-ID path below.
                        if track.get(
                            "motion_frame_id"
                        ) != frame_id - 1 or not identity_compatible(
                            candidate,
                            track,
                            max_conflicts=0,
                            min_shared=max(
                                min_confirmations,
                                (len(candidate["plate"]) + 1) // 2,
                            ),
                        ):
                            continue
                    distance = spatial_match(
                        candidate, track, current_time, match_distance
                    )
                    if distance is not None:
                        pairs.append((distance, candidate_index, track_id))
            pairs.sort()
            pair_candidates = Counter(index for _, index, _ in pairs)
            pair_tracks = Counter(track_id for _, _, track_id in pairs)
            assignments = {}
            used_tracks = set()
            reidentified = set()
            # A newly spawned, unadmitted OCR-noise track must not steal a
            # geometrically/temporally supported exact return of an older
            # identity. Respect all ambiguity checks before the greedy pass.
            recovery_proposals = {}
            for candidate_index, candidate in enumerate(candidates):
                direct = [
                    tid for _, index, tid in pairs if index == candidate_index
                ]
                if any(
                    tracks[tid].get("registry_admitted", False)
                    for tid in direct
                ):
                    continue
                recovered = match_reid(candidate, set(), current_time)
                if recovered is not None:
                    recovery_proposals.setdefault(recovered, []).append(
                        candidate_index
                    )
            for track_id, indices in recovery_proposals.items():
                if len(indices) == 1:
                    assignments[indices[0]] = track_id
                    used_tracks.add(track_id)
                    reidentified.add(track_id)
            for distance, candidate_index, track_id in pairs:
                if (
                    pair_candidates[candidate_index] > 1
                    or pair_tracks[track_id] > 1
                ):
                    # Resolve by strict evidence below, never nearest-only.
                    continue
                if candidate_index in assignments:
                    continue
                if track_id in used_tracks:
                    continue
                assignments[candidate_index] = track_id
                used_tracks.add(track_id)
            unmatched = [
                index
                for index in range(len(candidates))
                if index not in assignments
            ]
            reid_proposals = {}
            for candidate_index in unmatched:
                candidate = candidates[candidate_index]
                track_id = match_reid(candidate, used_tracks, current_time)
                if track_id is not None:
                    reid_proposals.setdefault(track_id, []).append(
                        candidate_index
                    )
            for track_id, candidate_indices in reid_proposals.items():
                if len(candidate_indices) != 1:
                    continue
                candidate_index = candidate_indices[0]
                if candidate_index in assignments or track_id in used_tracks:
                    continue
                assignments[candidate_index] = track_id
                used_tracks.add(track_id)
                reidentified.add(track_id)
                print(
                    f"Повторная идентификация: "
                    f"объект {track_id} | "
                    f"{candidates[candidate_index]['plate']}"
                )
            confirmed_plates = []
            observations_to_api = []
            display_plates = []
            for candidate_index, candidate in enumerate(candidates):
                if candidate_index in assignments:
                    track_id = assignments[candidate_index]
                    track = tracks[track_id]
                    association = (
                        "reid" if track_id in reidentified else "continuous"
                    )
                    association_gap = current_time - track["last_seen"]
                else:
                    track_id = next_track_id
                    next_track_id += 1
                    track = {
                        "camera_id": camera_id,
                        "session_id": session_id,
                        "center": candidate["center"],
                        "subtype": candidate["subtype"],
                        "history": deque(maxlen=history_size),
                        "motion_history": deque(maxlen=history_size),
                        "last_seen": current_time,
                        "visible": True,
                        "best_quality": -1.0,
                        "confirmed_number": None,
                        "confirmed_subtype": None,
                    }
                    tracks[track_id] = track
                    association, association_gap = "new", 0.0
                    log_visibility(
                        "ПОЯВИЛСЯ",
                        track_id,
                        track,
                        observed_plate=candidate["plate"],
                    )
                    print(f"Новый объект {track_id}: {candidate['plate']}")
                if not track.get("visible", True):
                    gap = current_time - track["last_seen"]
                    track["visible"] = True
                    log_visibility("ВЕРНУЛСЯ", track_id, track, gap)
                track["motion_history"].append(
                    dict(
                        tracking_time=current_time, center=candidate["center"]
                    )
                )
                track["motion_seen"] = current_time
                track["motion_frame_id"] = frame_id
                track["normalized_box"] = candidate["normalized_box"]
                track["center"] = candidate["center"]
                track["last_seen"] = current_time
                track["history"].append(
                    {
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "character_probabilities": candidate[
                            "character_probabilities"
                        ],
                        "timestamp": frame_timestamp,
                        "tracking_time": current_time,
                        "center": candidate["center"],
                        "plate": candidate["plate"],
                        "subtype": candidate["subtype"],
                        "confidence": candidate["confidence"],
                        "width": candidate["width"],
                        "quality": candidate["quality"],
                    }
                )
                merged = resolve(
                    track["history"],
                    track["confirmed_number"],
                    track.get("confirmed_subtype"),
                    min_confirmations,
                    change_confirmations,
                )
                track["confirmed_number"] = merged["confirmed_number"]
                track["confirmed_subtype"] = merged["confirmed_subtype"]
                number, status = merged["plate"], merged["status"]
                track["association_number"] = number
                track["association_evidence"] = merged["character_evidence"]
                remember_identity(
                    track, merged, current_time, min_confirmations
                )
                # Keep provisional readings in Live, but do not persist
                # one/two-frame objects. Once admitted, retain this identity
                # through occlusion without creating a new database record.
                track["registry_admitted"] = track.get(
                    "registry_admitted", False
                ) or registry_supported(merged, min_confirmations)
                # Partial wrong-layout OCR over a
                # continuously observed admitted
                # object is not evidence of a second real plate. It remains in
                # Live but is not persisted or mixed into the owner's history.
                quarantine = []
                if "#" in candidate["plate"]:
                    for other_id, other in tracks.items():
                        if (
                            other_id == track_id
                            or not other.get("registry_admitted")
                            or not other.get("confirmed_number")
                            or other.get("subtype") == candidate["subtype"]
                            or other.get("motion_frame_id")
                            not in (frame_id - 1, frame_id)
                        ):
                            continue
                        if (
                            spatial_match(
                                candidate, other, current_time, match_distance
                            )
                            is not None
                        ):
                            quarantine.append(other_id)
                if len(quarantine) == 1:
                    track["registry_admitted"] = False
                observation_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    repr((camera_id, session_id, track_id, frame_id)),
                ).hex
                if sample_debug:
                    item = next(
                        p
                        for p in diagnostic_plates
                        if p["box"] == candidate["box"]
                    )
                    item.update(
                        track_id=track_id,
                        merged_text=number,
                        camera_id=camera_id,
                        observation_id=observation_id,
                        registry_quarantine=quarantine,
                        recognition_status=status,
                        position=candidate["position"],
                        association=association,
                        association_gap_s=association_gap,
                        confirmed_number=merged["confirmed_number"],
                        character_evidence=merged["character_evidence"],
                        status="confirmed"
                        if status == "CONFIRMED"
                        else "partial"
                        if status == "PARTIAL"
                        else "waiting_confirmation",
                        reasons=[]
                        if status == "CONFIRMED"
                        else ["unknown_or_conflicting_characters"]
                        if status == "PARTIAL"
                        else ["insufficient_confirmations"],
                    )
                for item in live_plates:
                    if item["box"] == candidate["box"]:
                        item.update(
                            text=number,
                            subtype=merged["subtype"],
                            track_id=track_id,
                            camera_id=camera_id,
                            observation_id=observation_id,
                            position=candidate["position"],
                            status=status,
                            confirmed=status == "CONFIRMED",
                        )
                display_plates.append(
                    {
                        "box": candidate["box"],
                        "plate": number,
                        "position": candidate["position"]["direction"],
                    }
                )
                quality = candidate["quality"]
                confirmed = {
                    "camera_id": camera_id,
                    "observation_id": observation_id,
                    "track_id": track_id,
                    "frame_id": frame_id,
                    "captured_at": captured_at,
                    "video_time_s": (
                        media_timestamp if replay is not None else None
                    ),
                    "plate": number,
                    "observed_plate": candidate["plate"],
                    "subtype": merged["subtype"],
                    "status": status,
                    "confirmed": status == "CONFIRMED",
                    "character_evidence": merged["character_evidence"],
                    "position": candidate["position"],
                    "distance_m": None,
                    "confidence": round(merged["confidence"], 3),
                    "quality": round(quality, 4),
                }
                number_changed = track.get("saved_number") != number
                photo_matches_number = candidate["plate"] == number
                should_save = (
                    status == "CONFIRMED" and photo_matches_number
                ) and (
                    number_changed
                    or quality > track["best_quality"] + quality_improvement
                )
                if should_save:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = (
                        f"{session_id[:8]}_{track_id}_{number}_{timestamp}.jpg"
                    )
                    filepath = captures_dir / filename
                    if cv2.imwrite(str(filepath), frame):
                        prefix = (
                            replay.image_prefix
                            if replay is not None
                            else "/captures"
                        )
                        confirmed["image_path"] = f"{prefix}/{filename}"
                        track["best_quality"] = quality
                        track["saved_number"] = number
                if track["registry_admitted"]:
                    observations_to_api.append(confirmed)
                if status == "CONFIRMED":
                    confirmed_plates.append(confirmed)
                print(
                    f"Объект {track_id}: {number} "
                    f"({status}, {len(track['history'])} observations) | "
                    f"Качество: {quality:.1%}"
                )
            # A plate-shaped detection may remain usable for geometry while
            # OCR is rejected/noisy. Bridge only uniquely overlapping admitted
            # tracks, never add character votes or extend last valid OCR time.
            # A detection assigned to a different admitted
            # identity is excluded.
            assigned_boxes = {
                tuple(candidates[i]["box"]): tid
                for i, tid in assignments.items()
            }
            motion_proposals = {}
            for track_id, track in tracks.items():
                if track_id in used_tracks or not track.get(
                    "registry_admitted", False
                ):
                    continue
                matches = []
                for raw_index, raw in enumerate(plates):
                    owner = assigned_boxes.get(tuple(raw["box"]))
                    if owner is not None and tracks[owner].get(
                        "registry_admitted", False
                    ):
                        continue
                    x1, y1, x2, y2 = raw["box"]
                    geometry = dict(
                        frame_id=frame_id,
                        center=(
                            (x1 + x2) / (2 * width),
                            (y1 + y2) / (2 * height),
                        ),
                        normalized_box=[
                            x1 / width,
                            y1 / height,
                            x2 / width,
                            y2 / height,
                        ],
                    )
                    if (
                        spatial_match(
                            geometry, track, current_time, match_distance
                        )
                        is not None
                    ):
                        matches.append((raw_index, geometry))
                if len(matches) == 1:
                    raw_index, geometry = matches[0]
                    motion_proposals.setdefault(raw_index, []).append(
                        (track_id, geometry)
                    )
            for raw_index, proposals in motion_proposals.items():
                if len(proposals) != 1:
                    continue
                track_id, geometry = proposals[0]
                track = tracks[track_id]
                track.update(
                    center=geometry["center"],
                    normalized_box=geometry["normalized_box"],
                    motion_seen=current_time,
                    motion_frame_id=frame_id,
                )
                track["motion_history"].append(
                    dict(tracking_time=current_time, center=geometry["center"])
                )
                if sample_debug:
                    diagnostic_plates[raw_index]["motion_only_track_id"] = (
                        track_id
                    )
            completed.append(time.monotonic())
            fps = (
                (len(completed) - 1) / (completed[-1] - completed[0])
                if len(completed) > 1
                else 0.0
            )
            getattr(preview, "record_readings", lambda *args: None)(
                frame, plates, live_plates, captured_at, session_id
            )
            preview.update_results(
                live_plates,
                fps,
                output["timing_ms"]["total"],
                frame_timestamp,
                input_jpeg_ms,
                inference_queue_ms,
                inference_total_ms,
                (getattr(replay, "current_meta", None) or {}).get(
                    "frame_id", frame_id
                ),
            )
            if settings.get("LIVE_MATCHED_ONLY", "false").lower() == "true":
                preview.submit(
                    frame,
                    frame_timestamp,
                    camera.capture_fps,
                    camera.capture_read_ms,
                    camera.dropped_frames,
                    frame_id,
                )
            if sample_debug:
                diagnostics.finish(
                    output,
                    diagnostic_plates,
                    frame_id,
                    frame_timestamp,
                    payload,
                )
            with lock:
                results = display_plates
                result_time = time.monotonic()
            if observations_to_api:
                send_to_api(observations_to_api)
            worker_timings = dict(output["timing_ms"])
            worker_timings.update(
                input_jpeg=input_jpeg_ms,
                queue=inference_queue_ms,
                pipeline_call=inference_total_ms,
                worker_cycle=(time.perf_counter() - input_started) * 1000,
            )
            output["worker_timing_ms"] = worker_timings
            getattr(preview, "record_timings", lambda timings: None)(
                worker_timings
            )
            if replay is not None:
                replay.record_frame(
                    frame_id,
                    media_timestamp,
                    output,
                    diagnostic_plates,
                    confirmed_plates,
                    inference_total_ms,
                )

    def recognize_guarded():
        try:
            recognize()
        except Exception:
            preview.set_inference_error("worker_failed")
            diagnostics.error("worker_failed")
            print("Recognition worker stopped: worker_failed", flush=True)
            if replay is not None:
                getattr(replay, "record_worker_error", lambda: None)()
                replay.fail_worker()

    worker = threading.Thread(target=recognize_guarded, daemon=True)
    worker.start()
    try:
        while running:
            success, frame = camera.read()
            if not success:
                if replay is not None:
                    break
                if show_window and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            height, width = frame.shape[:2]
            with lock:
                latest_frame = frame
                latest_frame_id += 1
                latest_frame_timestamp = camera.last_frame_timestamp
                latest_media_timestamp = (
                    replay.media_time if replay is not None else 0.0
                )
                frame_ready.notify()
                plates = list(results)
                age = time.monotonic() - result_time
            if replay is not None:
                replay.wait_frame(latest_frame_id)
                if getattr(replay, "owns_preview", False):
                    continue
            if settings.get("LIVE_MATCHED_ONLY", "false").lower() == "true":
                continue
            preview.submit(
                frame,
                camera.last_frame_timestamp,
                camera.capture_fps,
                camera.capture_read_ms,
                camera.dropped_frames,
                (getattr(camera, "current_meta", None) or {}).get(
                    "frame_id", latest_frame_id
                ),
            )
            if not show_window:
                continue
            frame = frame.copy()
            cv2.line(
                frame, (width // 2, 0), (width // 2, height), (255, 255, 0), 1
            )
            if age < 1.5:
                for plate in plates:
                    x1, y1, x2, y2 = map(int, plate["box"])
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    number = plate["plate"]
                    position = plate["position"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(
                        frame,
                        (int(center_x), int(center_y)),
                        4,
                        (0, 0, 255),
                        -1,
                    )
                    cv2.putText(
                        frame,
                        f"{number} | {position}",
                        (x1, max(y1 - 12, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )
            capture_rate = camera.capture_fps
            capture_label = (
                f"{capture_rate:.1f}" if capture_rate is not None else "n/a"
            )
            anpr_rate = getattr(preview, "results", ([], None))[1]
            anpr_label = f"{anpr_rate:.1f}" if anpr_rate is not None else "n/a"
            cv2.putText(
                frame,
                f"Capture {capture_label} FPS | ANPR {anpr_label} FPS",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            if show_window:
                cv2.imshow("Rover ANPR", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        with frame_ready:
            running = False
            frame_ready.notify_all()
        camera.release()
        worker.join()
        preview.close()
        if replay is not None:
            replay.drain_sender(sender)
        sender.close()
        if show_window:
            cv2.destroyAllWindows()
        visibility_file.close()


if __name__ == "__main__":
    main()
