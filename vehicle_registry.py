"""Separate vehicle/encounter accounting; no tracking, OCR voting or
string merge."""

import hashlib
import json
import uuid

NAMESPACE = uuid.UUID("1fe15065-c8bc-4d2a-8616-779fe5074aeb")


def scoped_session(session_id, camera_id=None):
    # Legacy callers retain their exact DB key. Explicit cameras are isolated.
    return (
        session_id
        if not camera_id or camera_id == "unspecified"
        else "\x1fcamera:"
        + json.dumps([camera_id, session_id], separators=(",", ":"))
    )


def init_schema(connection):
    for sql in (
        "CREATE TABLE IF NOT EXISTS vehicle_p"
        "assages (\n        passage_id TEXT PR"
        "IMARY KEY, encounter_id TEXT NOT NUL"
        "L,\n        camera_id TEXT NOT NULL, "
        "session_id TEXT NOT NULL, track_id T"
        "EXT NOT NULL,\n        vehicle_id TEX"
        "T, detection_id INTEGER, number TEXT"
        ", status TEXT,\n        first_timesta"
        "mp REAL, last_timestamp REAL, identi"
        "ty_class TEXT NOT NULL,\n        UNIQ"
        "UE(camera_id,session_id,track_id))",
        "CREATE TABLE IF NOT EXISTS vehicle_o"
        "bservations (\n        observation_id"
        " TEXT PRIMARY KEY, passage_id TEXT N"
        "OT NULL,\n        camera_id TEXT NOT "
        "NULL, session_id TEXT NOT NULL, trac"
        "k_id TEXT NOT NULL,\n        captured"
        "_at REAL, plate TEXT NOT NULL, statu"
        "s TEXT NOT NULL,\n        confidence "
        "REAL, eligible INTEGER NOT NULL, pay"
        "load TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS vehicle_i"
        "dentity_links (\n        proof_id TEX"
        "T PRIMARY KEY, left_passage TEXT NOT"
        " NULL,\n        right_passage TEXT NO"
        "T NULL, relation TEXT NOT NULL, proo"
        "f_json TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS vehicle_a"
        "liases (\n        alias_id TEXT PRIMA"
        "RY KEY,canonical_vehicle_id TEXT NOT"
        " NULL,proof_id TEXT NOT NULL)",
        "CREATE INDEX IF NOT EXISTS vehicle_o"
        "bservations_passage ON vehicle_obser"
        "vations(passage_id)",
        "CREATE INDEX IF NOT EXISTS vehicle_p"
        "assages_encounter ON vehicle_passage"
        "s(encounter_id)",
    ):
        connection.execute(sql)


def record_observation(connection, session_id, event, detection_id=None):
    event = dict(event)
    event.setdefault("status", "CONFIRMED")  # Existing legacy API default.
    camera = event.get("camera_id") or "unspecified"
    track = str(event["track_id"])
    passage = uuid.uuid5(
        NAMESPACE, json.dumps([camera, session_id, track])
    ).hex
    observation = (
        event.get("observation_id")
        or event.get("event_id")
        or uuid.uuid4().hex
    )
    prior = connection.execute(
        "SELECT passage_id FROM vehicle_observations WHERE observation_id=?",
        (observation,),
    ).fetchone()
    if prior is not None:
        if prior[0] != passage:
            raise ValueError(
                "observation_id reused across cameras/sessions/tracks"
            )
        return passage
    eligible = event.get("visual_assessment") not in (
        "FALSE_OBJECT",
        "WRONG_TRACK_HISTORY",
    ) and event["status"] in ("PARTIAL", "PENDING", "CONFIRMED")
    captured = event.get("captured_at")
    # Missing source timestamps remain NULL; no wall-clock recording time
    # invented.
    connection.execute(
        """INSERT OR IGNORE INTO vehicle_passages
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            passage,
            passage,
            camera,
            session_id,
            track,
            None,
            detection_id,
            event["plate"],
            event["status"],
            captured,
            captured,
            "MODEL_TRACK_IDENTITY_NOT_VISUAL_GROUND_TRUTH"
            if eligible
            else "QUARANTINE",
        ),
    )
    inserted = connection.execute(
        """INSERT OR IGNORE INTO vehicle_observations
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            observation,
            passage,
            camera,
            session_id,
            track,
            captured,
            event["plate"],
            event["status"],
            event.get("confidence"),
            int(eligible),
            json.dumps(event, ensure_ascii=False, allow_nan=False),
        ),
    ).rowcount
    if not inserted:
        existing = connection.execute(
            "SELECT passage_id,payload FROM vehic"
            "le_observations WHERE observation_id"
            "=?",
            (observation,),
        ).fetchone()
        if existing[0] != passage:
            raise ValueError(
                "observation_id reused across cameras/sessions/tracks"
            )
        return passage
    row = connection.execute(
        "SELECT vehicle_id,encounter_id,ident"
        "ity_class FROM vehicle_passages WHER"
        "E passage_id=?",
        (passage,),
    ).fetchone()
    vehicle = row[0]
    if not eligible or row[2] == "QUARANTINE":
        vehicle = None
        connection.execute(
            "UPDATE vehicle_passages SET vehicle_"
            "id=NULL,identity_class='QUARANTINE' "
            "WHERE encounter_id=?",
            (row[1],),
        )
    elif (
        event["status"] == "CONFIRMED"
        and "#" not in event["plate"]
        and not vehicle
    ):
        vehicle = uuid.uuid5(NAMESPACE, "vehicle:" + row[1]).hex
        connection.execute(
            "UPDATE vehicle_passages SET vehicle_id=? WHERE encounter_id=?",
            (vehicle, row[1]),
        )
    connection.execute(
        """UPDATE vehicle_passages SET vehicle_id=COALESCE(?,vehicle_id),
        detection_id=COALESCE(?,detection_id),number=?,status=?,
        last_timestamp=CASE WHEN last_timestamp IS NULL THEN ?
                       WHEN ? IS NULL THEN last_timestamp
                       ELSE MAX(last_timestamp,?) END WHERE passage_id=?""",
        (
            vehicle,
            detection_id,
            event["plate"],
            event["status"],
            captured,
            captured,
            captured,
            passage,
        ),
    )
    return passage


def link_reviewed(connection, left, right, proof):
    """Explicit reviewed continuation; no automatic text-only or repeat-
    visit merge."""
    if proof.get("relation") not in (
        "same_passage_continuation",
        "cross_camera_same_observation",
    ):
        raise ValueError("Repeat passages must remain independent encounters")
    if not proof.get("reviewed_same_vehicle") or not proof.get(
        "evidence_refs"
    ):
        raise ValueError("Visual review with source references required")
    anchors = proof.get("anchor_observation_ids")
    if not isinstance(anchors, list) or len(anchors) != 2:
        raise ValueError("Two exact observation anchors required")
    records = []
    for passage, oid in zip((left, right), anchors):
        row = connection.execute(
            "SELECT passage_id,eligible,plate,sta"
            "tus FROM vehicle_observations WHERE "
            "observation_id=?",
            (oid,),
        ).fetchone()
        if row is None or row[0] != passage or not row[1]:
            raise ValueError("Missing, foreign or quarantined proof anchor")
        records.append(row)
    a, b = (row[2] for row in records)
    if len(a) != len(b) or any(
        x != y for x, y in zip(a, b) if x != "#" and y != "#"
    ):
        raise ValueError(
            "Contradictory plate evidence; review cannot silently overwrite it"
        )
    # An audit proof ID cannot be silently recycled for another merge.
    proof_id = (
        proof.get("proof_id")
        or hashlib.sha256(
            json.dumps(proof, sort_keys=True).encode()
        ).hexdigest()
    )
    prior = connection.execute(
        "SELECT proof_json FROM vehicle_identity_links WHERE proof_id=?",
        (proof_id,),
    ).fetchone()
    if prior is not None and json.loads(prior[0]) != proof:
        raise ValueError("proof_id already belongs to another reviewed link")
    # The proof, not matching text, establishes identity. Preserve
    # original IDs.
    ids = connection.execute(
        "SELECT passage_id,encounter_id,vehic"
        "le_id FROM vehicle_passages WHERE pa"
        "ssage_id IN (?,?)",
        (left, right),
    ).fetchall()
    if len(ids) != 2 and left != right:
        raise ValueError("Unknown passages")
    if connection.execute(
        "SELECT COUNT(*) FROM vehicle_passage"
        "s WHERE passage_id IN (?,?) AND iden"
        "tity_class='QUARANTINE'",
        (left, right),
    ).fetchone()[0]:
        raise ValueError(
            "Quarantined local identity cannot establish a vehicle link"
        )
    encounters = {row[1] for row in ids}
    canonical = min(encounters)
    vehicles = sorted({row[2] for row in ids if row[2]})
    vehicle = vehicles[0] if vehicles else None
    for alias in vehicles:
        if alias != vehicle:
            connection.execute(
                "UPDATE vehicle_aliases SET canonical"
                "_vehicle_id=? WHERE canonical_vehicl"
                "e_id=?",
                (vehicle, alias),
            )
            connection.execute(
                "INSERT OR REPLACE INTO vehicle_aliases VALUES (?,?,?)",
                (alias, vehicle, proof_id),
            )
    for encounter in encounters:
        connection.execute(
            "UPDATE vehicle_passages SET encounte"
            "r_id=?,vehicle_id=COALESCE(?,vehicle"
            "_id) WHERE encounter_id=?",
            (canonical, vehicle, encounter),
        )
    proof_id = (
        proof.get("proof_id")
        or hashlib.sha256(
            json.dumps(proof, sort_keys=True).encode()
        ).hexdigest()
    )
    connection.execute(
        "INSERT OR IGNORE INTO vehicle_identity_links VALUES (?,?,?,?,?)",
        (
            proof_id,
            left,
            right,
            proof["relation"],
            json.dumps(proof, ensure_ascii=False),
        ),
    )
    return canonical


def registry_snapshot(connection):
    connection.row_factory = __import__("sqlite3").Row
    rows = connection.execute(
        "SELECT encounter_id,vehicle_id,COUNT"
        "(*) AS local_tracks,\n        GROUP_C"
        "ONCAT(DISTINCT camera_id) AS camera_"
        "ids,\n        GROUP_CONCAT(DISTINCT n"
        "umber) AS observed_numbers,\n        "
        "GROUP_CONCAT(DISTINCT status) AS sta"
        "tuses,\n        MIN(first_timestamp) "
        "AS first_timestamp,MAX(last_timestam"
        "p) AS last_timestamp,\n        GROUP_"
        "CONCAT(DISTINCT identity_class) AS i"
        "dentity_classes\n        FROM vehicle"
        "_passages GROUP BY encounter_id ORDE"
        "R BY encounter_id"
    ).fetchall()
    aliases = {}
    for alias, canonical in connection.execute(
        "SELECT alias_id,canonical_vehicle_id FROM vehicle_aliases"
    ):
        aliases.setdefault(canonical, []).append(alias)
    return [
        dict(row, vehicle_id_aliases=aliases.get(row["vehicle_id"], []))
        for row in rows
    ]


def observation_snapshot(connection, limit=100):
    connection.row_factory = __import__("sqlite3").Row
    rows = connection.execute(
        "SELECT o.observation_id,o.passage_id"
        ",p.encounter_id,p.vehicle_id,\n      "
        "  o.camera_id,o.session_id,o.track_i"
        "d,o.captured_at,o.plate,o.status,o.c"
        "onfidence,o.eligible\n        FROM ve"
        "hicle_observations o JOIN vehicle_pa"
        "ssages p ON p.passage_id=o.passage_i"
        "d\n        ORDER BY o.rowid DESC LIMI"
        "T ?",
        (max(1, min(int(limit), 1000)),),
    ).fetchall()
    return [dict(row) for row in rows]
