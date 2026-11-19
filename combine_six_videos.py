"""Offline, conservative consolidation of saved ANPR observations.

Never runs models or infers hidden characters.
Never equates a recognized string with a car.
Only manually reviewed observation pairs may join local evidence episodes.
"""

import argparse
import csv
import hashlib
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def clock(seconds):
    milliseconds = round(seconds * 1000)
    return (
        f"{milliseconds // 3600000:02}:"
        f"{milliseconds // 60000 % 60:02}:"
        f"{milliseconds // 1000 % 60:02}."
        f"{milliseconds % 1000:03}"
    )


def confirmed_text(row):
    return (
        row["text"]
        if row["status"] == "CONFIRMED" and "#" not in row["text"]
        else None
    )


def episodes(rows):
    """Separate conflicting CONFIRMED strings even inside a legacy track.

    This is a saved-data partition, not a new tracker/re-ID or OCR algorithm.
    No stitching across different tracks, files, or cameras here.
    """
    buckets = defaultdict(list)
    for row in rows:
        key = (
            row["video"],
            row["track_id"] or "untracked:" + row["observation_id"],
        )
        buckets[key].append(row)
    result = []
    for key, values in sorted(buckets.items()):
        values.sort(key=lambda row: (int(row["frame"]), row["observation_id"]))
        current, text = [], None
        for row in values:
            new = confirmed_text(row)
            if current and new and text and new != text:
                result.append(current)
                current, text = [], None
            current.append(row)
            text = new or text
        if current:
            result.append(current)
    return result


def combine(root, archive, output, reviews=None):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    reports = sorted(archive.rglob("report.json"))
    if len(reports) != 6:
        raise ValueError("Exactly six complete archived reports required")
    audit_path = root / "results/recognition_audit/audit.csv"
    audit = read_csv(audit_path) if audit_path.exists() else []
    audit_map = defaultdict(list)
    for entry in audit:
        audit_map[(entry["video"], entry["frame"], entry["track_id"])].append(
            entry
        )
    observations, sources = [], []
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        csv_path = report_path.with_name("detections.csv")
        relative = report_path.parent.relative_to(archive).as_posix()
        camera, filename = relative.split("/", 1)
        source = root / "test_videos" / camera / filename
        if report["status"] != "done" or not source.is_file():
            raise ValueError("Missing/incomplete source: " + relative)
        hh, mm, ss = map(int, filename.split("-")[0].split("."))
        start = hh * 3600 + mm * 60 + ss
        entries = read_csv(csv_path)
        if len(entries) != report["detection_count"]:
            raise ValueError("CSV/report count mismatch: " + relative)
        sources.append(
            dict(
                video=relative,
                path=str(source),
                size=source.stat().st_size,
                csv=str(csv_path.relative_to(root)),
                csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                report=str(report_path.relative_to(root)),
                processed_frames=report["processed_frames"],
                archived_elapsed_s=report["elapsed_seconds"],
                archived_processing_fps=report["processing_fps"],
                metadata=report["source_metadata"],
            )
        )
        for index, raw in enumerate(entries, 1):
            row = dict(raw)
            row.update(
                observation_id=f"{camera}_{hh:02}{mm:02}{ss:02}_{index:05}",
                video=relative,
                camera_id=camera,
                source_file=filename,
                source_video=str(source.relative_to(root)),
                source_csv=str(csv_path.relative_to(root)),
                source_csv_row=index + 1,
                evidence_version="ARCHIVED_PRE_P0",
                timestamp=clock(start + float(raw["video_time_s"])),
                timestamp_basis="filename_time_of_day_plus_PTS; date_unknown; sync_not_calibrated",  # noqa: E501
                interval_seconds=start + float(raw["video_time_s"]) - 45000,
                frame_reference=f"test_videos/{relative}#frame={raw['frame']}&pts={raw['video_time_s']}",
                visual_assessment="NOT_REVIEWED",
                visual_text="",
                frame_path="",
                context_path="",
                audit_note="",
                group_id="",
                local_episode_id="",
            )
            matches = audit_map[(relative, raw["frame"], raw["track_id"])]
            for entry in matches:
                if entry["ocr_result"] != raw["text"]:
                    continue
                row["frame_path"] = (
                    "results/recognition_audit/" + entry["frame_path"]
                )
                row["context_path"] = (
                    "results/recognition_audit/" + entry["context_path"]
                )
                if entry["object_class"] == "НЕ ПРОВЕРЕНО":
                    continue
                row["visual_assessment"] = (
                    "FALSE_OBJECT"
                    if entry["object_class"].startswith("посторонний")
                    else "REAL_PLATE"
                )
                row["visual_text"] = entry["visually_readable"]
                row["audit_note"] = entry["error_cause"]
                if "несовпадение подтверждённой" in entry["ocr_error"]:
                    row["visual_assessment"] = "WRONG_TRACK_HISTORY"
            observations.append(row)
    parts = episodes(observations)
    lookup, parent = {}, {}
    for number, values in enumerate(parts, 1):
        eid = f"E{number:05}"
        parent[eid] = eid
        for row in values:
            row["local_episode_id"] = eid
            lookup[row["observation_id"]] = row

    def find(key):
        while parent[key] != key:
            key = parent[key]
        return key

    proofs = json.loads(reviews.read_text(encoding="utf-8")) if reviews else []
    accepted = []
    for proof in proofs:
        if not proof.get("reviewed_same_vehicle") or not proof.get(
            "visual_evidence"
        ):
            raise ValueError("Each merge needs an explicit visual review")
        anchors = []
        for anchor in (proof["left"], proof["right"]):
            matches = [
                row
                for row in observations
                if row["video"] == anchor["video"]
                and int(row["frame"]) == anchor["frame"]
                and row["text"] == proof["plate"]
            ]
            if (
                len(matches) != 1
                or matches[0]["visual_assessment"] != "REAL_PLATE"
            ):
                raise ValueError(
                    "Review anchor not unique/visually verified: "
                    + str(anchor)
                )
            anchors.append(matches[0])
        a, b = [find(row["local_episode_id"]) for row in anchors]
        parent[max(a, b)] = min(a, b)
        accepted.append(
            dict(
                proof,
                left_episode=anchors[0]["local_episode_id"],
                right_episode=anchors[1]["local_episode_id"],
            )
        )
    groups = defaultdict(list)
    for row in observations:
        row["group_id"] = "G" + find(row["local_episode_id"])[1:]
        groups[row["group_id"]].append(row)
    registry = []
    for gid, values in sorted(groups.items()):
        strong = sorted({confirmed_text(row) for row in values} - {None})
        # Keep a representative actual partial, never fill its # from another
        # row.
        partials = [row for row in values if row["status"] == "PARTIAL"]
        representative = max(
            partials,
            key=lambda row: (
                len(row["text"].replace("#", "")),
                float(row["confidence"] or 0),
            ),
            default=values[-1],
        )
        assessments = {row["visual_assessment"] for row in values}
        category = (
            "CONTAINS_FALSE_OBJECT"
            if "FALSE_OBJECT" in assessments
            else "CONTAINS_WRONG_TRACK_HISTORY"
            if "WRONG_TRACK_HISTORY" in assessments
            else "VISUAL_REAL_EXAMPLE"
            if "REAL_PLATE" in assessments
            else "NOT_VISUALLY_VERIFIED"
        )
        status = (
            "CONFIRMED"
            if strong
            else "PARTIAL"
            if partials
            else "PENDING"
            if any(row["status"] == "PENDING" for row in values)
            else values[-1]["status"]
        )
        registry.append(
            dict(
                group_id=gid,
                local_episode_ids=";".join(
                    sorted({r["local_episode_id"] for r in values})
                ),
                camera_ids=";".join(sorted({r["camera_id"] for r in values})),
                source_files=";".join(sorted({r["video"] for r in values})),
                track_ids=";".join(
                    sorted(
                        {
                            r["video"] + ":" + (r["track_id"] or "NONE")
                            for r in values
                        }
                    )
                ),
                timestamp_first=min(
                    values, key=lambda r: float(r["interval_seconds"])
                )["timestamp"],
                timestamp_last=max(
                    values, key=lambda r: float(r["interval_seconds"])
                )["timestamp"],
                number=";".join(strong) if strong else representative["text"],
                status=status,
                confidence_max=max(
                    (
                        float(r["confidence"])
                        for r in values
                        if r["confidence"]
                    ),
                    default=None,
                ),
                observation_count=len(values),
                visual_category=category,
                identity_basis="REVIEWED_PAIR_AND_LOCAL_ARCHIVE_EPISODES"
                if len({r["local_episode_id"] for r in values}) > 1
                else "LOCAL_ARCHIVE_EPISODE_NOT_PROOF_OF_ONE_CAR",
                first_observation_id=values[0]["observation_id"],
                visual_evidence=";".join(
                    sorted(
                        {
                            r["context_path"]
                            for r in values
                            if r["visual_assessment"] != "NOT_REVIEWED"
                        }
                    )
                ),
                registration_disposition="QUARANTINE"
                if category.startswith("CONTAINS_")
                or status in ("REJECTED", "UNREADABLE", "UNTRACKED")
                else "PROVISIONAL_NOT_VALIDATED_FLEET_REGISTRY",
            )
        )
    candidates = []
    for left, right in itertools.combinations(registry, 2):
        if left["status"] != "CONFIRMED" or right["status"] != "CONFIRMED":
            continue
        if left["number"] != right["number"]:
            continue
        candidates.append(
            dict(
                left_group=left["group_id"],
                right_group=right["group_id"],
                number=left["number"],
                relation="CROSS_CAMERA"
                if left["camera_ids"] != right["camera_ids"]
                else "SAME_CAMERA_OTHER_EPISODE",
                left_videos=left["source_files"],
                right_videos=right["source_files"],
                left_time=left["timestamp_first"],
                right_time=right["timestamp_first"],
                decision="REVIEW_REQUIRED_NOT_MERGED",
                evidence="Exact recognized string only; same vehicle or repeat passage unknown",  # noqa: E501
            )
        )
    for pair in accepted:
        candidates.append(
            dict(
                left_group="G" + find(pair["left_episode"])[1:],
                right_group="G" + find(pair["right_episode"])[1:],
                left_episode=pair["left_episode"],
                right_episode=pair["right_episode"],
                number=pair["plate"],
                relation=pair["relation"],
                left_videos=pair["left"]["video"],
                right_videos=pair["right"]["video"],
                decision="VISUALLY_PROVEN_SAME_VEHICLE_MERGED",
                evidence=pair["visual_evidence"],
            )
        )
    status_counts = Counter(row["status"] for row in observations)
    both = sorted(
        text
        for text in {r["text"] for r in observations if confirmed_text(r)}
        if {r["camera_id"] for r in observations if confirmed_text(r) == text}
        == {"Front", "Back"}
    )
    verified = {
        r["visual_text"]
        for r in observations
        if r["visual_assessment"] == "REAL_PLATE"
        and re_full_plate(r["visual_text"])
    }
    summary = dict(
        data_version="ARCHIVED_PRE_P0_NOT_FULL_CURRENT_INFERENCE",
        observations=len(observations),
        status_counts=dict(status_counts),
        unique_confirmed_strings=len(
            {r["text"] for r in observations if confirmed_text(r)}
        ),
        unique_all_output_strings=len(
            {r["text"] for r in observations if r["text"]}
        ),
        local_evidence_episodes=len(parts),
        registry_groups=len(registry),
        machine_confirmed_groups=sum(
            r["status"] == "CONFIRMED" for r in registry
        ),
        partial_only_groups=sum(r["status"] == "PARTIAL" for r in registry),
        real_unique_vehicle_count=None,
        visually_verified_distinct_plate_examples=len(verified),
        visually_verified_plate_examples=sorted(verified),
        proven_repeat_pairs=len(accepted),
        unresolved_duplicate_pairs=len(candidates) - len(accepted),
        same_confirmed_strings_both_cameras=both,
        unresolved_identity_groups=sum(
            r["identity_basis"].startswith("LOCAL_") for r in registry
        ),
        known_false_observations=sum(
            r["visual_assessment"] == "FALSE_OBJECT" for r in observations
        ),
        known_wrong_history_observations=sum(
            r["visual_assessment"] == "WRONG_TRACK_HISTORY"
            for r in observations
        ),
        sources=sources,
        archive_processed_frames=sum(s["processed_frames"] for s in sources),
        archive_inference_elapsed_s=sum(
            s["archived_elapsed_s"] for s in sources
        ),
        archive_inference_fps=sum(s["processed_frames"] for s in sources)
        / sum(s["archived_elapsed_s"] for s in sources),
        archive_device="NOT_PROVEN_FROM_SAVED_REPORTS",
        consolidation_elapsed_s=time.perf_counter() - started,
        timestamp_note="Time-of-day from filenames + stored video PTS, date unknown. Front/Back nominally simultaneous per user; offset not calibrated.",  # noqa: E501
        warning="Groups are conservative evidence partitions, NOT a count of real cars; local legacy tracks can be wrong. PARTIAL # never establishes a merge.",  # noqa: E501
    )
    output.mkdir(parents=True)
    write_csv(
        output / "all_observations.csv", observations, list(observations[0])
    )
    write_csv(output / "combined_registry.csv", registry, list(registry[0]))
    write_csv(
        output / "duplicates_review.csv",
        candidates,
        [
            "left_group",
            "right_group",
            "left_episode",
            "right_episode",
            "number",
            "relation",
            "left_videos",
            "right_videos",
            "left_time",
            "right_time",
            "decision",
            "evidence",
        ],
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "reviewed_merges.json").write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def re_full_plate(text):
    return (
        bool(text)
        and "#" not in text
        and text != "не применимо"
        and text != "НЕ ПРОВЕРЕНО"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--archive", type=Path, default=Path("results/batch_test_videos")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/combined_six_videos")
    )
    parser.add_argument("--reviewed-merges", type=Path)
    args = parser.parse_args()
    result = combine(
        args.root,
        args.root / args.archive,
        args.root / args.output,
        args.reviewed_merges,
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "sources"},
            ensure_ascii=False,
            indent=2,
        )
    )
