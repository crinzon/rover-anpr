import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(
    os.getenv("DB_PATH", str(Path(__file__).resolve().parent / "plates.db"))
)


def init_db():
    with sqlite3.connect(DB_PATH) as connection:
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

        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(detections)")
        }

        if "image_path" not in columns:
            connection.execute(
                "ALTER TABLE detections ADD COLUMN image_path TEXT"
            )


def save_detection(plate, confidence, image_path=None):
    with sqlite3.connect(DB_PATH) as connection:
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
    with sqlite3.connect(DB_PATH) as connection:
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
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT *
            FROM detections
            ORDER BY id DESC
        """).fetchall()

        return [dict(row) for row in rows]


def apply_event(session_id, plate):
    """Atomically deduplicate event and update persistent track/history."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("BEGIN IMMEDIATE")
        event_id = plate.get("event_id")
        if event_id:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO processed_events VALUES (?)",
                (event_id,),
            ).rowcount
            if not inserted:
                return False
        key = (session_id, str(plate["track_id"]))
        saved = connection.execute(
            "SELECT detection_id, quality, number, image_path "
            "FROM detection_tracks WHERE session_id=? AND track_id=?",
            key,
        ).fetchone()
        number, quality = plate["plate"], plate["quality"]
        image = plate.get("image_path")
        if saved is None:
            cursor = connection.execute(
                "INSERT INTO detections "
                "(plate, confidence, detected_at, image_path) "
                "VALUES (?,?,?,?)",
                (
                    number,
                    plate["confidence"],
                    datetime.now().astimezone().isoformat(),
                    image,
                ),
            )
            connection.execute(
                "INSERT INTO detection_tracks VALUES (?,?,?,?,?,?)",
                (*key, cursor.lastrowid, quality, number, image),
            )
        else:
            detection_id, best_quality, old_number, best_image = saved
            improved = quality > best_quality + 0.08
            if improved or number != old_number:
                best_image = image or best_image
                connection.execute(
                    "UPDATE detections SET plate=?, confidence=?, "
                    "image_path=? "
                    "WHERE id=?",
                    (number, plate["confidence"], best_image, detection_id),
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
        return True
