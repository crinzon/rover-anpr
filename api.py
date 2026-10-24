import asyncio
import json
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from anpr_diagnostics import read_diagnostics
from database import apply_event, get_detections, init_db
from live_preview import read_preview

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


@app.get("/live")
def live_page():
    return FileResponse(Path(__file__).parent / "static" / "live.html")


@app.get("/live/status")
def live_status():
    jpeg, meta = read_preview()
    return dict(available=jpeg is not None, **meta)


@app.get("/live/diagnostics")
def live_diagnostics():
    return Response(
        content=json.dumps(read_diagnostics()),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/live/frame")
def live_frame(after: float = 0):
    jpeg, meta = read_preview()
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "X-Accel-Buffering": "no",
    }
    if jpeg is None:
        return Response(status_code=503, headers=headers)
    if meta["timestamp"] <= after:
        return Response(status_code=204, headers=headers)
    for field in (
        "timestamp",
        "frame_age_ms",
        "jpeg_ms",
        "api_read_ms",
        "transfer_to_api_ms",
        "preview_fps",
        "fps",
        "processing_ms",
        "volume_write_ms",
        "capture_ipc_ms",
        "preview_wait_ms",
        "inference_queue_ms",
        "inference_total_ms",
        "preview_render_ms",
    ):
        headers["X-" + field.replace("_", "-")] = str(meta.get(field, 0))
    return Response(jpeg, media_type="image/jpeg", headers=headers)


@app.get("/live/stream")
async def live_stream():
    async def frames():
        timestamp = None
        while True:
            jpeg, meta = await asyncio.to_thread(read_preview)
            if jpeg is not None and meta["timestamp"] != timestamp:
                timestamp = meta["timestamp"]
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(1 / 30)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
