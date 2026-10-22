import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from database import init_db, save_detection, update_detection, get_detections

app = FastAPI()

CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")

init_db()

latest_plates = []
saved_tracks = {}
last_detection_time = 0.0

TIMEOUT = 2.0
QUALITY_IMPROVEMENT = 0.08


@app.get("/plates")
def get_plates():
    if time.monotonic() - last_detection_time > TIMEOUT:
        return {"plates": []}

    return {"plates": latest_plates}


@app.get("/history")
def get_history():
    return {"detections": get_detections()}


@app.post("/plates")
def update_plates(data: dict):
    global latest_plates, last_detection_time

    plates = data.get("plates", [])
    session_id = data.get("session_id")

    if not session_id:
        return {"status": "error", "message": "session_id required"}

    latest_plates = plates
    last_detection_time = time.monotonic()

    for plate in plates:
        track_id = plate.get("track_id")

        if track_id is None:
            continue

        key = (session_id, track_id)

        number = plate["plate"]
        confidence = plate["confidence"]
        quality = plate["quality"]
        image_path = plate.get("image_path")

        if key not in saved_tracks:
            detection_id = save_detection(
                number,
                confidence,
                image_path
            )

            saved_tracks[key] = {
                "id": detection_id,
                "quality": quality,
                "number": number,
                "best_image_path": image_path
            }

            print(f"Сохранён объект {track_id}: {number}")
            continue

        saved = saved_tracks[key]

        number_changed = number != saved["number"]

        quality_improved = (
            quality > saved["quality"] + QUALITY_IMPROVEMENT
        )

        if quality_improved:
            saved["quality"] = quality

            if image_path:
                saved["best_image_path"] = image_path

        if number_changed:
            saved["number"] = number

            if image_path:
                saved["best_image_path"] = image_path

            update_detection(
                saved["id"],
                number,
                confidence,
                saved["best_image_path"]
            )

            print(f"Исправлен номер объекта {track_id}: {number}")

        elif quality_improved:
            update_detection(
                saved["id"],
                saved["number"],
                confidence,
                saved["best_image_path"]
            )

            print(
                f"Улучшено качество объекта {track_id}: "
                f"{saved['number']}"
            )

    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)