"""Launch existing camera.main; secrets are read from ignored .env only."""

import argparse
from pathlib import Path
from urllib.parse import quote

from video_settings import load_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("camera", choices=("Front", "Back"))
    args = parser.parse_args()
    settings = load_env()
    user = settings.get("CAMERA_USER")
    password = settings.get("CAMERA_PASSWORD")
    if not user or not password:
        raise SystemExit(
            "Missing CAMERA_USER or CAMERA_PASSWORD in local .env"
        )
    host = {"Front": "192.168.77.202", "Back": "192.168.77.201"}[args.camera]
    source = (
        "rtsp://"
        + quote(user, safe="")
        + ":"
        + quote(password, safe="")
        + "@"
        + host
        + ":554/cam/realmonitor?channel=1&subtype=0"
    )
    root = Path(__file__).resolve().parent
    settings.update(
        VIDEO_SOURCE=source,
        DEVICE="cuda",
        SHOW_WINDOW="false",
        CAMERA_ID=args.camera,
        API_URL="http://172.22.192.1:8000/plates",
        RTSP_TRANSPORT="tcp",
        VIDEO_OPERATION_TIMEOUT="60",
        LIVE_MATCHED_ONLY="true",
        LIVE_PREVIEW_PATH=str(
            root / "captures" / (".live-preview-" + args.camera)
        ),
        STATS_PATH=str(
            root
            / "results/dual_rtsp_2026-11-17"
            / (args.camera + "_stats.json")
        ),
    )
    import os

    os.environ["ROVER_RTSP_LOW_LATENCY"] = "true"
    import camera

    camera.main(settings)


if __name__ == "__main__":
    main()
