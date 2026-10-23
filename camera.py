import csv
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2

from event_sender import EventSender
from full_stand.config import StandConfig
from full_stand.pipeline import PlatePipeline
from plate_validator import is_valid_plate, normalize_plate
from video_stream import VideoStream


def load_env():
    settings = {}
    env_path = Path(__file__).resolve().parent / ".env"

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip()

    import os

    settings.update(os.environ)
    return settings


def main():
    settings = load_env()
    video_source = settings.get("VIDEO_SOURCE", "0")
    api_url = settings.get("API_URL", "http://127.0.0.1:8000/plates")
    device = settings.get("DEVICE", "mps")
    show_window = settings.get("SHOW_WINDOW", "true").lower() == "true"

    config = StandConfig(device=device, confidence=0.5)
    config.validate()
    pipeline = PlatePipeline(config)
    session_id = uuid.uuid4().hex
    captures_dir = Path(__file__).resolve().parent / "captures"
    captures_dir.mkdir(exist_ok=True)
    history_size = 10
    min_confirmations = 3
    change_confirmations = 7
    track_timeout = 1.0
    reid_timeout = 30.0
    match_distance = 0.12
    quality_improvement = 0.08
    visibility_timeout = 1.0  # секунд без обнаружения до события ИСЧЕЗ
    visibility_log = Path(__file__).resolve().parent / "visibility_log.csv"
    source = int(video_source) if video_source.isdigit() else video_source
    camera = VideoStream(
        source,
        initial=float(settings.get("VIDEO_RETRY_INITIAL", "0.5")),
        maximum=float(settings.get("VIDEO_RETRY_MAX", "10")),
        timeout=float(settings.get("VIDEO_OPERATION_TIMEOUT", "5")),
    )
    lock = threading.Lock()
    frame_ready = threading.Condition(lock)
    latest_frame = None
    latest_frame_id = 0
    latest_frame_timestamp = 0.0
    results = []
    result_time = 0.0
    running = True
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

    sender = EventSender(
        api_url,
        capacity=int(settings.get("DELIVERY_QUEUE_SIZE", "256")),
        attempts=int(settings.get("DELIVERY_ATTEMPTS", "5")),
        initial=float(settings.get("DELIVERY_RETRY_INITIAL", "0.5")),
        maximum=float(settings.get("DELIVERY_RETRY_MAX", "10")),
        timeout=float(settings.get("DELIVERY_TIMEOUT", "2")),
    )

    def send_to_api(plates):
        for plate in plates:
            sender.submit(session_id, plate)

    def plate_distance(first, second):
        if len(first) != len(second):
            return 100
        return sum(a != b for a, b in zip(first, second))

    def get_track_number(track):
        if not track["history"]:
            return None
        numbers = [item["plate"] for item in track["history"]]
        return Counter(numbers).most_common(1)[0][0]

    def get_track_numbers(track):
        return set(item["plate"] for item in track["history"])

    def match_reid(candidate, excluded_tracks, current_time):
        matches = []
        for track_id, track in tracks.items():
            if track_id in excluded_tracks:
                continue
            if current_time - track["last_seen"] > reid_timeout:
                continue
            reference = get_track_number(track)
            if reference is None:
                continue
            difference = plate_distance(candidate["plate"], reference)
            if difference > 1:
                continue
            history_numbers = get_track_numbers(track)
            if candidate["plate"] in history_numbers:
                difference = 0
            matches.append((difference, track_id))
        matches.sort()
        if not matches:
            return None
        best_difference = matches[0][0]
        best_matches = [
            track_id
            for difference, track_id in matches
            if difference == best_difference
        ]
        if len(best_matches) != 1:
            return None
        return best_matches[0]

    def plate_quality(frame, box, confidence):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 6:
            return 0.0, 0.0
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.color_bgr2gray)
        sharpness = float(cv2.Laplacian(gray, cv2.cv_64f).var())
        size_score = min((x2 - x1) / 160.0, 1.0) * min((y2 - y1) / 35.0, 1.0)
        sharpness_score = min(sharpness / 250.0, 1.0)
        quality = confidence * (0.45 * size_score + 0.55 * sharpness_score)
        return round(quality, 4), sharpness

    def weighted_choice(items):
        scores = {}
        counts = Counter()
        for item in items:
            number = item["plate"]
            counts[number] += 1
            scores[number] = scores.get(number, 0.0) + max(
                item["quality"], 0.01
            )
        if not scores:
            return None, 0
        number = max(scores, key=lambda value: (scores[value], counts[value]))
        return number, counts[number]

    def recognize():
        nonlocal latest_frame, results, result_time, next_track_id
        last_processed_frame_id = 0
        while running:
            with frame_ready:
                frame_ready.wait_for(
                    lambda: not running
                    or latest_frame_id > last_processed_frame_id
                )
                if not running:
                    break
                frame = latest_frame.copy()
                frame_id = latest_frame_id
                frame_timestamp = latest_frame_timestamp
                # Claim before encoding/inference, including failed attempts.
                last_processed_frame_id = frame_id
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue
            try:
                output = pipeline(buffer.tobytes())
                plates = output["plates"]
                print(f"Обработка: {output['timing_ms']['total']:.0f} мс")
            except Exception as error:
                print("Recognition error:", error)
                time.sleep(0.1)
                continue
            height, width = frame.shape[:2]
            current_time = time.monotonic()
            candidates = []
            for plate in plates:
                if not plate["readable"] or plate["confidence"] < 0.85:
                    continue
                number = normalize_plate(plate["text"])
                if not is_valid_plate(number):
                    continue
                x1, y1, x2, y2 = plate["box"]
                plate_width = x2 - x1
                plate_height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                x = center_x / width
                y = center_y / height
                direction = (
                    "LEFT" if x < 0.4 else "RIGHT" if x > 0.6 else "CENTER"
                )
                quality, sharpness = plate_quality(
                    frame, plate["box"], plate["confidence"]
                )
                candidates.append(
                    {
                        "quality": quality,
                        "sharpness": sharpness,
                        "plate": number,
                        "box": plate["box"],
                        "center": (x, y),
                        "width": plate_width,
                        "height": plate_height,
                        "position": {
                            "x": round(x - 0.5, 3),
                            "y": round(y - 0.5, 3),
                            "offset_x_px": round(center_x - width / 2),
                            "offset_y_px": round(center_y - height / 2),
                            "direction": direction,
                        },
                        "distance_m": None,
                        "confidence": round(plate["confidence"], 3),
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
                    if current_time - track["last_seen"] > track_timeout:
                        continue
                    dx = candidate["center"][0] - track["center"][0]
                    dy = candidate["center"][1] - track["center"][1]
                    distance = (dx * dx + dy * dy) ** 0.5
                    if distance <= match_distance:
                        pairs.append((distance, candidate_index, track_id))
            pairs.sort()
            assignments = {}
            used_tracks = set()
            for distance, candidate_index, track_id in pairs:
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
                print(
                    f"Повторная идентификация: "
                    f"объект {track_id} | "
                    f"{candidates[candidate_index]['plate']}"
                )
            confirmed_plates = []
            display_plates = []
            for candidate_index, candidate in enumerate(candidates):
                if candidate_index in assignments:
                    track_id = assignments[candidate_index]
                    track = tracks[track_id]
                else:
                    track_id = next_track_id
                    next_track_id += 1
                    track = {
                        "center": candidate["center"],
                        "history": deque(maxlen=history_size),
                        "last_seen": current_time,
                        "visible": True,
                        "best_quality": -1.0,
                        "confirmed_number": None,
                    }
                    tracks[track_id] = track
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
                track["center"] = candidate["center"]
                track["last_seen"] = current_time
                track["history"].append(
                    {
                        "frame_id": frame_id,
                        "timestamp": frame_timestamp,
                        "plate": candidate["plate"],
                        "confidence": candidate["confidence"],
                        "width": candidate["width"],
                        "quality": candidate["quality"],
                    }
                )
                numbers = [item["plate"] for item in track["history"]]
                if track["confirmed_number"] is None:
                    recent = list(track["history"])[-5:]
                    number, count = weighted_choice(recent)
                    if count < min_confirmations:
                        continue
                    track["confirmed_number"] = number
                else:
                    number = track["confirmed_number"]
                    candidate_number, candidate_count = weighted_choice(
                        track["history"]
                    )
                    if (
                        candidate_number != number
                        and candidate_count >= change_confirmations
                    ):
                        track["confirmed_number"] = candidate_number
                        number = candidate_number
                        print(
                            f"Подтверждена смена номера объекта "
                            f"{track_id}: {number}"
                        )
                    count = numbers.count(number)
                display_plates.append(
                    {
                        "box": candidate["box"],
                        "plate": number,
                        "position": candidate["position"]["direction"],
                    }
                )
                matching = [
                    item
                    for item in track["history"]
                    if item["plate"] == number
                ]
                average_confidence = sum(
                    item["confidence"] for item in matching
                ) / len(matching)
                quality = candidate["quality"]
                confirmed = {
                    "track_id": track_id,
                    "plate": number,
                    "position": candidate["position"],
                    "distance_m": None,
                    "confidence": round(average_confidence, 3),
                    "quality": round(quality, 4),
                }
                number_changed = track.get("saved_number") != number
                photo_matches_number = candidate["plate"] == number
                should_save = photo_matches_number and (
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
                        confirmed["image_path"] = f"/captures/{filename}"
                        track["best_quality"] = quality
                        track["saved_number"] = number
                confirmed_plates.append(confirmed)
                print(
                    f"Объект {track_id}: {number} "
                    f"({count}/{len(track['history'])}) | "
                    f"Качество: {quality:.1%}"
                )
            with lock:
                results = display_plates
                result_time = time.monotonic()
            if confirmed_plates:
                send_to_api(confirmed_plates)

    worker = threading.Thread(target=recognize, daemon=True)
    worker.start()
    try:
        while True:
            success, frame = camera.read()
            if not success:
                if show_window and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            height, width = frame.shape[:2]
            with lock:
                latest_frame = frame.copy()
                latest_frame_id += 1
                latest_frame_timestamp = time.monotonic()
                frame_ready.notify()
                plates = list(results)
                age = time.monotonic() - result_time
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
                        cv2.font_hershey_simplex,
                        0.65,
                        (0, 255, 0),
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
        sender.close()
        camera.release()
        if show_window:
            cv2.destroyAllWindows()
        worker.join(timeout=2)
        visibility_file.close()


if __name__ == "__main__":
    main()
