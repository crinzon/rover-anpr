
import sqlite3
from datetime import datetime
from pathlib import Path

import os

DB_PATH = Path(
    os.getenv(
        "DB_PATH",
        str(Path(__file__).resolve().parent / "plates.db")
    )
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
        cursor = connection.execute("""
            INSERT INTO detections (
                plate, confidence, detected_at, image_path
            )
            VALUES (?, ?, ?, ?)
        """, (
            plate,
            confidence,
            datetime.now().astimezone().isoformat(),
            image_path
        ))

        return cursor.lastrowid


def update_detection(detection_id, plate, confidence, image_path=None):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            UPDATE detections
            SET plate = ?,
                confidence = ?,
                image_path = COALESCE(?, image_path)
            WHERE id = ?
        """, (plate, confidence, image_path, detection_id))


def get_detections():
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT *
            FROM detections
            ORDER BY id DESC
        """).fetchall()

        return [dict(row) for row in rows]
