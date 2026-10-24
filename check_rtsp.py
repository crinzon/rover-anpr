"""Bounded video probe. Never imports the ANPR models or prints credentials."""

import argparse
import math
import signal
import threading
import time

from video_settings import load_env, stream_options
from video_stream import VideoStream


def probe(stream, duration, logger=print):
    started = time.monotonic()
    # Interrupt a pending open/read/backoff at the overall deadline.
    timer = threading.Timer(duration, stream.stop.set)
    timer.start()
    count = 0
    first = last = None
    sizes = set()
    read_seconds = 0.0
    try:
        while not stream.stop.is_set():
            before = time.monotonic()
            success, frame = stream.read()
            read_seconds += time.monotonic() - before
            if success:
                now = time.monotonic()
                first = now if first is None else first
                last = now
                count += 1
                sizes.add((frame.shape[1], frame.shape[0]))
    finally:
        timer.cancel()
        stream.release()
    elapsed = time.monotonic() - started
    fps = (count - 1) / (last - first) if count > 1 and last > first else 0.0
    dimensions = ", ".join(f"{w}x{h}" for w, h in sorted(sizes)) or "unknown"
    logger(
        f"Stream: {'available' if count else 'unavailable'}; "
        f"resolution={dimensions}; frames={count}; FPS={fps:.2f}; "
        f"elapsed={elapsed:.2f}s; read_wait={read_seconds:.2f}s; "
        f"first_frame={(first - started) if first else 0:.2f}s"
    )
    return 0 if count else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")
    try:
        settings = load_env()
        source = settings.get("VIDEO_SOURCE", "0")
        source = int(source) if source.isdigit() else source
        stream = VideoStream(source, **stream_options(settings))
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: stream.stop.set())
        return probe(stream, args.duration)
    except Exception:
        print(
            (
                "Video probe failed; check configuration and camera access "
                "locally."
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
