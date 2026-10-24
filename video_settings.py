"""Shared configuration without model or OpenCV imports."""

import os
from pathlib import Path


def load_env():
    settings = {}
    path = Path(__file__).resolve().parent / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip().strip("\"'")
    settings.update(os.environ)
    return settings


def stream_options(settings):
    return dict(
        initial=float(settings.get("VIDEO_RETRY_INITIAL", "0.5")),
        maximum=float(settings.get("VIDEO_RETRY_MAX", "10")),
        timeout=float(settings.get("VIDEO_OPERATION_TIMEOUT", "5")),
        backend=settings.get("VIDEO_BACKEND", "auto"),
        transport=settings.get("RTSP_TRANSPORT", "tcp"),
    )
