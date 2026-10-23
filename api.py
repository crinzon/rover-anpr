import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from database import apply_event, get_detections, init_db

app = FastAPI()

CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")

init_db()

latest_plates = []
last_detection_time = 0.0

TIMEOUT = 2.0
QUALITY_IMPROVEMENT = 0.08


@app.get("/health")
def health():
    return {"status": "ok"}


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

    accepted = []
    for plate in plates:
        if plate.get("track_id") is None:
            continue
        if apply_event(session_id, plate):
            accepted.append(plate)
    if accepted:
        latest_plates = accepted
        last_detection_time = time.monotonic()

    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
