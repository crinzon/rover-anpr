import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from vehicle_registry import (
    init_schema,
    observation_snapshot,
    record_observation,
    registry_snapshot,
    scoped_session,
)

DB_PATH = Path(
    os.getenv("DB_PATH", str(Path(__file__).resolve().parent / "plates.db"))
)


def init_db():
    with closing(sqlite3.connect(DB_PATH)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL,
                confidence REAL NOT NULL,
                detected_at TEXT NOT NULL,
                image_path TEXT
            )
        """)

        connection.execute(
            "CREATE TABLE IF NOT EXISTS processed_events "
            "(event_id TEXT PRIMARY KEY)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS detection_tracks ("
            "session_id TEXT, track_id TEXT, detection_id INTEGER, "
            "quality REAL, number TEXT, image_path TEXT, "
            "PRIMARY KEY (session_id, track_id))"
        )

        init_schema(connection)
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(detections)")
        }

        if "image_path" not in columns:
            connection.execute(
                "ALTER TABLE detections ADD COLUMN image_path TEXT"
            )

        for name, definition in (
            ("subtype", "TEXT"),
            ("status", "TEXT NOT NULL DEFAULT 'CONFIRMED'"),
        ):
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE detections ADD COLUMN {name} {definition}"
                )


def save_detection(plate, confidence, image_path=None):
    with closing(sqlite3.connect(DB_PATH)) as connection, connection:
        cursor = connection.execute(
            """
            INSERT INTO detections (
                plate, confidence, detected_at, image_path
            )
            VALUES (?, ?, ?, ?)
        """,
            (
                plate,
                confidence,
                datetime.now().astimezone().isoformat(),
                image_path,
            ),
        )

        return cursor.lastrowid


def update_detection(detection_id, plate, confidence, image_path=None):
    with closing(sqlite3.connect(DB_PATH)) as connection, connection:
        connection.execute(
            """
            UPDATE detections
            SET plate = ?,
                confidence = ?,
                image_path = COALESCE(?, image_path)
            WHERE id = ?
        """,
            (plate, confidence, image_path, detection_id),
        )


def get_detections():
    with closing(sqlite3.connect(DB_PATH)) as connection, connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT d.*, MAX(p.last_timestamp) AS last_seen,
                   COUNT(o.observation_id) AS observation_count
            FROM detections d
            LEFT JOIN vehicle_passages p ON p.detection_id=d.id
            LEFT JOIN vehicle_observations o ON o.passage_id=p.passage_id
            GROUP BY d.id
            ORDER BY d.id DESC
        """).fetchall()

        return [dict(row) for row in rows]


def apply_event(session_id, plate):
    """Atomically deduplicate event and update persistent track/history."""
    with closing(sqlite3.connect(DB_PATH)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        event_id = plate.get("event_id")
        if event_id:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO processed_events VALUES (?)",
                (event_id,),
            ).rowcount
            if not inserted:
                return False
        key = (
            scoped_session(session_id, plate.get("camera_id")),
            str(plate["track_id"]),
        )
        observation_id = plate.get("observation_id")
        if observation_id:
            prior = connection.execute(
                "SELECT camera_id,session_id,track_id"
                " FROM vehicle_observations WHERE obs"
                "ervation_id=?",
                (observation_id,),
            ).fetchone()
            if prior is not None:
                expected = (
                    plate.get("camera_id") or "unspecified",
                    session_id,
                    str(plate["track_id"]),
                )
                if tuple(prior) != expected:
                    raise ValueError(
                        "observation_id reused across cameras/sessions/tracks"
                    )
                return False
        saved = connection.execute(
            "SELECT t.detection_id, t.quality, t.number, t.image_path, "
            "d.subtype, d.status FROM detection_tracks t "
            "JOIN detections d ON d.id=t.detection_id "
            "WHERE t.session_id=? AND t.track_id=?",
            key,
        ).fetchone()
        number, quality = plate["plate"], plate["quality"]
        image = plate.get("image_path")
        subtype = plate.get("subtype")
        status = plate.get("status", "CONFIRMED")
        if saved is None:
            cursor = connection.execute(
                "INSERT INTO detections "
                "(plate, confidence, detected_at, ima"
                "ge_path, subtype, status) "
                "VALUES (?,?,?,?,?,?)",
                (
                    number,
                    plate["confidence"],
                    datetime.now().astimezone().isoformat(),
                    image,
                    subtype,
                    status,
                ),
            )
            detection_id = cursor.lastrowid
            connection.execute(
                "INSERT INTO detection_tracks VALUES (?,?,?,?,?,?)",
                (*key, detection_id, quality, number, image),
            )
        else:
            (
                detection_id,
                best_quality,
                old_number,
                best_image,
                old_subtype,
                old_status,
            ) = saved
            improved = quality > best_quality + 0.08
            if (
                improved
                or number != old_number
                or subtype != old_subtype
                or status != old_status
            ):
                best_image = image or best_image
                connection.execute(
                    "UPDATE detections SET plate=?, confidence=?, "
                    "image_path=?, subtype=?, status=? "
                    "WHERE id=?",
                    (
                        number,
                        plate["confidence"],
                        best_image,
                        subtype,
                        status,
                        detection_id,
                    ),
                )
                connection.execute(
                    "UPDATE detection_tracks SET quality=?, number=?, "
                    "image_path=? WHERE session_id=? AND track_id=?",
                    (
                        max(best_quality, quality)
                        if improved
                        else best_quality,
                        number,
                        best_image,
                        *key,
                    ),
                )
        record_observation(connection, session_id, plate, detection_id)
        return True


def get_vehicle_registry():
    with closing(sqlite3.connect(DB_PATH)) as connection:
        return registry_snapshot(connection)


def get_vehicle_observations(limit=100):
    with closing(sqlite3.connect(DB_PATH)) as connection:
        return observation_snapshot(connection, limit)
