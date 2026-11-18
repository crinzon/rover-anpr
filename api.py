import asyncio
import base64
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    FileResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from anpr_diagnostics import read_diagnostics
from current_plates import CurrentPlates
from database import (
    apply_event,
    get_detections,
    get_vehicle_observations,
    get_vehicle_registry,
    init_db,
)
from live_preview import (
    DEFAULT_PATH,
    PreviewReader,
    read_preview,
    system_metrics,
)

app = FastAPI()

CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")

init_db()

TIMEOUT = 2.0
current_plates = CurrentPlates(ttl=TIMEOUT)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/plates")
def get_plates(include_partial: bool = False):
    rows = current_plates.snapshot()
    if not include_partial:
        rows = [p for p in rows if p.get("status", "CONFIRMED") == "CONFIRMED"]
    return {"plates": rows}


@app.get("/history")
def get_history(include_partial: bool = False):
    rows = get_detections()
    if not include_partial:
        rows = [p for p in rows if p["status"] == "CONFIRMED"]
    return {"detections": rows}


@app.post("/plates")
def update_plates(data: dict):
    plates = data.get("plates", [])
    session_id = data.get("session_id")

    if not session_id:
        return {"status": "error", "message": "session_id required"}

    try:
        current_plates.update(session_id, plates, apply_event)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {"status": "ok"}


CAMERA_READERS = {
    name: PreviewReader(DEFAULT_PATH.parent / (".live-preview-" + name))
    for name in ("Front", "Back")
}

VIDEO_DEMO_ENABLED = os.getenv("VIDEO_DEMO_ENABLED") == "1"
if VIDEO_DEMO_ENABLED:
    for name in ("Front", "Back"):
        CAMERA_READERS["Demo" + name] = PreviewReader(
            Path(__file__).parent / ".run/video-demo" / name / "preview",
            max_age=86400,
        )
    CAMERA_READERS["Demo"] = CAMERA_READERS["DemoFront"]


def demo_channel(camera):
    return {"Demo": "Front", "DemoFront": "Front", "DemoBack": "Back"}.get(
        camera
    )


@app.get("/live/demo")
def demo_status(channel: str = "Front"):
    if not VIDEO_DEMO_ENABLED:
        return {"enabled": False}
    from video_demo import snapshot

    try:
        return snapshot(channel)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/live/demo")
def demo_control(data: dict):
    if not VIDEO_DEMO_ENABLED:
        raise HTTPException(status_code=409, detail="VIDEO DEMO disabled")
    from video_demo import control

    try:
        return control(
            data.get("action"),
            data.get("video_id"),
            data.get("channel", "Front"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/live/demo/both")
def demo_both(data: dict):
    if not VIDEO_DEMO_ENABLED:
        raise HTTPException(status_code=409, detail="VIDEO DEMO disabled")
    from video_demo import control_both

    try:
        return control_both(data.get("action"), data.get("videos"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/live/demo/all")
def demo_all_status():
    if not VIDEO_DEMO_ENABLED:
        return {"enabled": False}
    from video_demo import sequence_snapshot

    return dict(enabled=True, **sequence_snapshot())


@app.post("/live/demo/all")
def demo_all_control(data: dict):
    if not VIDEO_DEMO_ENABLED:
        raise HTTPException(status_code=409, detail="VIDEO DEMO disabled")
    from video_demo import sequence_control

    try:
        return dict(enabled=True, **sequence_control(data.get("action")))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.on_event("startup")
def restore_paused_video_demo():
    if VIDEO_DEMO_ENABLED:
        from video_demo import restore_paused_sequence

        restore_paused_sequence()


@app.on_event("shutdown")
def stop_video_demo():
    if VIDEO_DEMO_ENABLED:
        from video_demo import shutdown

        shutdown()


def camera_preview(camera=None):
    if camera is None:
        return read_preview()
    if camera not in CAMERA_READERS:
        raise HTTPException(status_code=404, detail="Unknown camera")
    jpeg, meta = CAMERA_READERS[camera].read()
    if demo_channel(camera):
        from video_demo import STATE, read_json

        if meta.get("session_id") != read_json(
            STATE / demo_channel(camera) / "control.json"
        ).get("run_id"):
            return None, {}
    return jpeg, meta


@app.get("/live/system")
def live_system():
    return system_metrics.snapshot()


@app.get("/live/cameras")
def live_cameras():
    return {"cameras": list(CAMERA_READERS)}


@app.get("/live")
def live_page(camera: str | None = None, dual: bool = False):
    if camera is not None and camera not in CAMERA_READERS:
        raise HTTPException(status_code=404, detail="Unknown camera")
    if dual or camera is None:
        return FileResponse(
            Path(__file__).parent / "static" / "live_dual.html"
        )
    return FileResponse(Path(__file__).parent / "static" / "live.html")


@app.get("/live/status")
def live_status(camera: str | None = None):
    jpeg, meta = camera_preview(camera)
    result = dict(available=jpeg is not None, **meta)
    path = CAMERA_READERS[camera].path if camera else DEFAULT_PATH
    try:
        summary_path = Path(str(path) + ".readings.json")
        if summary_path.stat().st_size <= 1_000_000:
            readings = json.loads(summary_path.read_text(encoding="utf-8"))
            if demo_channel(camera):
                from video_demo import STATE, read_json

                if readings.get("session_id") != read_json(
                    STATE / demo_channel(camera) / "control.json"
                ).get("run_id"):
                    readings = None
            if readings is not None:
                result["readings"] = readings
    except (OSError, ValueError):
        pass
    return result


@app.get("/live/diagnostics")
def live_diagnostics():
    return Response(
        content=json.dumps(read_diagnostics()),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/live/frame")
def live_frame(
    after: float = 0,
    camera: str | None = None,
    bundle: bool = False,
    after_frame: int | None = None,
    session: str = "",
):
    jpeg, meta = camera_preview(camera)
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "X-Accel-Buffering": "no",
    }
    if jpeg is None:
        return Response(status_code=503, headers=headers)
    if after_frame is not None and isinstance(meta.get("frame_id"), int):
        unchanged = (
            meta.get("session_id") == session
            and meta["frame_id"] <= after_frame
        )
    else:
        unchanged = meta["timestamp"] <= after
    if unchanged:
        return Response(status_code=204, headers=headers)
    for field in (
        "timestamp",
        "frame_id",
        "result_frame_id",
        "frame_age_ms",
        "jpeg_ms",
        "api_read_ms",
        "transfer_to_api_ms",
        "capture_fps",
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
        headers["X-" + field.replace("_", "-")] = (
            str(meta[field]) if meta.get(field) is not None else ""
        )
    if bundle:
        return Response(
            json.dumps(
                dict(
                    meta,
                    image="data:image/jpeg;base64,"
                    + base64.b64encode(jpeg).decode("ascii"),
                )
            ),
            media_type="application/json",
            headers=headers,
        )
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


@app.get("/vehicles")
def vehicles():
    """Separate evidence encounters; not a visually validated fleet count."""
    return {"encounters": get_vehicle_registry()}


@app.get("/observations")
def observations(limit: int = 100):
    return {"observations": get_vehicle_observations(limit)}
