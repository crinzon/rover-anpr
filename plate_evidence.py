"""Conservative per-character evidence, limited to distinct observed frames."""

import math
from collections import Counter

from plate_validator import is_valid_plate, normalize_plate


def observation(plate, threshold=0.85):
    text = normalize_plate(plate["text"])
    subtype = plate.get("subtype", "unknown")
    if not plate["readable"] or not is_valid_plate(text, subtype):
        return None
    ranked = plate.get("subtype_probs", [])
    if (
        ranked
        and next((p["probability"] for p in ranked if p["name"] == subtype), 0)
        < threshold
    ):
        return None
    characters = {p["position"]: p for p in plate.get("characters", [])}
    visible = []
    for index, char in enumerate(text):
        info = characters.get(index)
        probability = info["probability"] if info else plate["confidence"]
        reliable = (
            isinstance(probability, (int, float))
            and math.isfinite(probability)
            and probability >= threshold
        )
        visible.append(char if char != "#" and reliable else "#")
    return "".join(visible), subtype


def identity_compatible(candidate, track, max_conflicts=1, min_shared=0):
    """Reject contradictory reliable text before borrowing a track's history.

    Reuse the existing re-ID allowance of one OCR substitution. Unknown
    positions are not contradictions and cannot prove a different identity.
    This is an association guard, not an additional voting system.
    """
    if any(
        candidate.get(field) is not None
        and track.get(field) is not None
        and candidate[field] != track[field]
        for field in ("camera_id", "session_id")
    ):
        return False
    reference = track.get("confirmed_number") or track.get(
        "association_number"
    )
    if not reference:
        history = track.get("history", ())
        reference = history[-1]["plate"] if history else None
    if reference is None:
        return True
    text = candidate["plate"]
    if len(text) != len(reference):
        return False
    conflicts = sum(
        a != b for a, b in zip(text, reference) if a != "#" and b != "#"
    )
    shared = sum(a == b and a != "#" for a, b in zip(text, reference))
    return conflicts <= max_conflicts and shared >= min_shared


def spatial_match(candidate, track, now, max_distance=0.12):
    """Existing-track geometry gate with constant-velocity history prediction.

    All coordinates are normalized. Require actual box overlap with the
    predicted position as well as the existing center-distance gate.
    """
    box = track.get("normalized_box")
    target = candidate.get("normalized_box")
    if box is None or target is None:
        return None  # Missing geometry is not re-identification evidence.
    samples = [
        p
        for p in track.get("motion_history", track.get("history", ()))
        if "tracking_time" in p and "center" in p
    ]
    vx = vy = 0.0
    if len(samples) >= 2:
        mean_t = sum(p["tracking_time"] for p in samples) / len(samples)
        mean_x = sum(p["center"][0] for p in samples) / len(samples)
        mean_y = sum(p["center"][1] for p in samples) / len(samples)
        denominator = sum((p["tracking_time"] - mean_t) ** 2 for p in samples)
        if denominator > 0:
            vx = (
                sum(
                    (p["tracking_time"] - mean_t) * (p["center"][0] - mean_x)
                    for p in samples
                )
                / denominator
            )
            vy = (
                sum(
                    (p["tracking_time"] - mean_t) * (p["center"][1] - mean_y)
                    for p in samples
                )
                / denominator
            )
    gap = now - track.get("motion_seen", track["last_seen"])
    if gap < 0:
        return None
    predicted = [
        box[0] + vx * gap,
        box[1] + vy * gap,
        box[2] + vx * gap,
        box[3] + vy * gap,
    ]
    dx = candidate["center"][0] - (predicted[0] + predicted[2]) / 2
    dy = candidate["center"][1] - (predicted[1] + predicted[3]) / 2
    distance = math.hypot(dx, dy)
    overlap = (
        min(predicted[2], target[2]) - max(predicted[0], target[0]),
        min(predicted[3], target[3]) - max(predicted[1], target[1]),
    )
    if distance <= max_distance and min(overlap) > 0:
        return distance
    # A forecast can overshoot when a hand/camera stops abruptly.
    # Last-box fallback requires a consecutively detected frame, NOT a loss.
    if (
        candidate.get("frame_id") is not None
        and track.get("motion_frame_id") == candidate["frame_id"] - 1
        and 0 <= gap <= 1.0
    ):
        actual_distance = math.hypot(
            candidate["center"][0] - (box[0] + box[2]) / 2,
            candidate["center"][1] - (box[1] + box[3]) / 2,
        )
        actual_overlap = (
            min(box[2], target[2]) - max(box[0], target[0]),
            min(box[3], target[3]) - max(box[1], target[1]),
        )
        if actual_distance <= max_distance and min(actual_overlap) > 0:
            return actual_distance
    return None


def reid_match(
    candidate, track, now, max_gap=3.0, max_distance=0.12, min_confirmations=3
):
    """Strict recovery inside the existing tracker, never text-only matching.

    Lost PARTIAL tracks may return. Reliable contradictions, insufficient
    shared evidence, incompatible motion or missing geometry must not merge.
    """
    if any(
        candidate.get(field) is not None
        and track.get(field) is not None
        and candidate[field] != track[field]
        for field in ("camera_id", "session_id")
    ):
        return None
    if not 0 <= now - track["last_seen"] <= max_gap:
        return None
    if track.get("subtype") != candidate["subtype"]:
        return None
    reference = track.get("confirmed_number") or track.get(
        "association_number"
    )
    anchor = track.get("identity_anchor")
    anchor_valid = (
        anchor is not None
        and 0 <= now - anchor["time"] <= max_gap
        and anchor["subtype"] == candidate["subtype"]
    )
    if not reference and anchor_valid:
        reference = anchor["plate"]
    if not reference or len(reference) != len(candidate["plate"]):
        return None
    shared = [
        i
        for i, (a, b) in enumerate(zip(reference, candidate["plate"]))
        if a != "#" and b != "#"
    ]
    # Unlike uninterrupted one-symbol OCR noise, a lost object must return
    # without ANY reliable contradiction. A common region alone is not proof.
    if any(reference[i] != candidate["plate"][i] for i in shared):
        return None
    evidence = list(track.get("association_evidence", ()))
    # Current occlusion must not erase a just-established identity snapshot.
    # This adds no votes and does not confirm the current reading.
    if anchor_valid and anchor["plate"] == reference:
        evidence += anchor["evidence"]
    supported = {
        p["position"]
        for p in evidence
        if p["position"] < len(reference)
        and p["char"] == reference[p["position"]]
        and len(set(p["frame_ids"])) >= min_confirmations
    }
    if len(set(shared) & supported) < max(
        min_confirmations, (len(reference) + 1) // 2
    ):
        return None
    return spatial_match(candidate, track, now, max_distance)


def registry_supported(result, min_confirmations=3):
    """Temporal admission, not proof that a detected object is a real plate.

    Reuse the confirmation requirement for each visible character. Missing
    positions need no invented evidence; fully unknown text cannot qualify.
    """
    known = [p for p in result["character_evidence"] if p["char"] != "#"]
    return bool(known) and all(
        len(set(p["frame_ids"])) >= min_confirmations for p in known
    )


def _consensus(items, required):
    groups = Counter((p["subtype"], len(p["plate"])) for p in items)
    if not groups:
        return None
    # Ambiguous subtype/length never borrows symbols from the other group.
    top = groups.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    subtype, length = top[0][0]
    cohort = [
        p
        for p in items
        if (p["subtype"], len(p["plate"])) == (subtype, length)
    ]
    chars, evidence = [], []
    for position in range(length):
        votes, weights = {}, {}
        for item in cohort:
            char = item["plate"][position]
            probability = item.get("character_probabilities", {}).get(
                position, item["confidence"]
            )
            quality = item.get("quality", 1.0)
            if (
                char == "#"
                or not isinstance(probability, (int, float))
                or not isinstance(quality, (int, float))
                or not math.isfinite(probability)
                or probability < 0.85
                or not math.isfinite(quality)
                or quality <= 0
            ):
                continue
            weight = min(probability, 1.0) * min(quality, 1.0)
            votes.setdefault(char, []).append(item["frame_id"])
            weights[char] = weights.get(char, 0.0) + weight
        char = "#"
        if len(votes) == 1:
            char = next(iter(votes))
        elif votes:
            best = max(weights, key=weights.get)
            # Counts alone cannot decide a conflict. Keep the existing minimum
            # independent-frame support AND require >=85% of credible weight.
            if (
                len(votes[best]) >= required
                and weights[best] / sum(weights.values()) >= 0.85
            ):
                char = best
        frames = votes.get(char, [])
        chars.append(char)
        conflicts = sorted(votes) if len(votes) > 1 else []
        evidence.append(
            dict(
                position=position,
                char=char,
                frame_ids=frames,
                conflicts=conflicts,
                weights=weights,
            )
        )
    text = "".join(chars)
    enough = all(len(p["frame_ids"]) >= required for p in evidence)
    full = "#" not in text and is_valid_plate(text, subtype)
    return dict(
        plate=text,
        subtype=subtype,
        character_evidence=evidence,
        supported=full and enough,
        confidence=sum(p["confidence"] for p in cohort) / len(cohort),
    )


def resolve(
    history,
    confirmed_number=None,
    confirmed_subtype=None,
    min_confirmations=3,
    change_confirmations=7,
):
    # Never accept evidence explicitly belonging to different tracked objects.
    history = list(history)
    track_ids = {
        p["track_id"] for p in history if p.get("track_id") is not None
    }
    if len(track_ids) > 1:
        raise ValueError("observations belong to different tracks")
    # A retry/spurious worker notification cannot create additional evidence.
    distinct = {p["frame_id"]: p for p in history}
    items = list(distinct.values())
    if not items:
        raise ValueError("no observations")
    result = _consensus(items[-10:], min_confirmations)
    if result is None:
        latest = items[-1]
        result = dict(
            plate="#" * len(latest["plate"]),
            subtype=latest["subtype"],
            character_evidence=[],
            supported=False,
            confidence=latest["confidence"],
        )
    changed = confirmed_number is not None and (
        result["plate"] != confirmed_number
        or result["subtype"] != confirmed_subtype
    )
    if result["supported"] and changed:
        alternate = _consensus(items[-10:], change_confirmations)
        result["supported"] = bool(
            alternate
            and alternate["supported"]
            and alternate["plate"] == result["plate"]
            and alternate["subtype"] == result["subtype"]
        )
    if result["supported"]:
        confirmed_number = result["plate"]
        confirmed_subtype = result["subtype"]
        status = "CONFIRMED"
    else:
        status = "PARTIAL" if "#" in result["plate"] else "PENDING"
    result.update(
        status=status,
        confirmed_number=confirmed_number,
        confirmed_subtype=confirmed_subtype,
    )
    return result


def remember_identity(track, result, now, min_confirmations=3):
    """One existing-consensus snapshot for bounded re-ID, never new voting."""
    if not registry_supported(result, min_confirmations):
        return
    latest = next(reversed(track.get("history", ())), None)
    if latest and latest["plate"] != result["plate"]:
        return  # Retained old votes are not a fresh complete observation.
    known = sum(char != "#" for char in result["plate"])
    if known < max(min_confirmations, (len(result["plate"]) + 1) // 2):
        return
    previous = track.get("identity_anchor")
    if previous and "#" not in previous["plate"] and "#" in result["plate"]:
        return
    track["identity_anchor"] = dict(
        plate=result["plate"],
        subtype=result["subtype"],
        time=now,
        evidence=[
            dict(p, frame_ids=list(p["frame_ids"]))
            for p in result["character_evidence"]
        ],
    )
