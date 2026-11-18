"""Small file-playback adapter around camera.main; no second ANPR/tracker."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".run/video-demo"
PREVIEW = STATE / "preview"
CHANNELS = ("Front", "Back")
_processes = {}
_locks = {name: threading.Lock() for name in CHANNELS}


def read_json(path, default=None):
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if attempt < 2:
                time.sleep(0.01)
    return default or {}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    try:
        for attempt in range(30):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 29:
                    raise
                time.sleep(0.02)
    finally:
        tmp.unlink(missing_ok=True)


def videos():
    paths = sorted(
        p
        for p in (ROOT / "test_videos").rglob("*")
        if p.is_file() and p.suffix.lower() in (".mp4", ".mov")
    )
    return [
        dict(
            id=str(i),
            name=str(p.relative_to(ROOT / "test_videos")).replace("\\", "/"),
        )
        for i, p in enumerate(paths)
    ]


def channel_state(channel):
    if channel not in CHANNELS:
        raise ValueError("Unknown channel")
    return STATE / channel


def snapshot(channel="Front"):
    directory = channel_state(channel)
    current = read_json(directory / "control.json")
    status = read_json(directory / "status.json")
    if status.get("run_id") != current.get("run_id"):
        status = {}
    process = _processes.get(channel)
    return dict(
        enabled=True,
        channel=channel,
        videos=videos(),
        control=current,
        status=status,
        worker_alive=process is not None and process.poll() is None,
    )


def _ensure_worker(channel):
    process = _processes.get(channel)
    if process is not None and process.poll() is None:
        return
    directory = channel_state(channel)
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        linux_root = (
            "/mnt/" + ROOT.drive[0].lower() + str(ROOT)[2:].replace("\\", "/")
        )
        python = os.environ.get(
            "VIDEO_DEMO_PYTHON",
            "/home/user/.venvs/rover-anpr-cu128-test-20261117/bin/python",
        )
        command = [
            "wsl",
            "-d",
            os.environ.get("VIDEO_DEMO_DISTRO", "Ubuntu"),
            "--cd",
            linux_root,
            "-e",
            python,
            "-u",
            "video_demo.py",
            "--worker",
            "--channel",
            channel,
        ]
    else:
        command = [
            os.environ.get("VIDEO_DEMO_PYTHON", sys.executable),
            "-u",
            str(ROOT / "video_demo.py"),
            "--worker",
            "--channel",
            channel,
        ]
    kwargs = dict(cwd=ROOT, stdin=subprocess.DEVNULL)
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    with (directory / "worker.log").open("a", encoding="utf-8") as log:
        _processes[channel] = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, **kwargs
        )


def _control(action, video_id=None, channel="Front"):
    directory = channel_state(channel)
    with _locks[channel]:
        choices = videos()
        if not choices:
            raise ValueError("Нет файлов в test_videos")
        if action not in ("start", "pause", "restart", "select"):
            raise ValueError("Unknown action")
        current = read_json(directory / "control.json")
        default = next(
            (v["id"] for v in choices if v["name"].startswith(channel + "/")),
            choices[0]["id"],
        )
        selected = (
            video_id
            if video_id is not None
            else current.get("video_id", default)
        )
        if selected not in {v["id"] for v in choices}:
            raise ValueError("Unknown video")
        status = read_json(directory / "status.json")
        if status.get("run_id") != current.get("run_id"):
            status = {}
        process = _processes.get(channel)
        alive = (process is not None and process.poll() is None) or (
            status.get("phase") in ("playing", "paused")
            and time.time() - status.get("updated_at", 0) < 3
        )
        resume = (
            action == "start"
            and alive
            and current.get("video_id") == selected
            and current.get("state") in ("playing", "paused")
            and status.get("phase") not in ("ended", "error")
        )
        if action == "pause":
            if current.get("state") == "playing":
                current["state"] = "paused"
        elif resume:
            current["state"] = "playing"
        else:
            current = dict(
                run_id=uuid.uuid4().hex,
                video_id=selected,
                state="idle" if action == "select" else "playing",
            )
        current.pop("sync_group", None)
        current["updated_at"] = time.time()
        atomic_json(directory / "control.json", current)
        if current.get("state") == "playing":
            _ensure_worker(channel)
        return snapshot(channel)


def _control_both(action, selected=None):
    if action not in ("start", "restart", "pause", "stop"):
        raise ValueError("Unknown paired action")
    selected = selected or {}
    with _locks["Front"], _locks["Back"]:
        choices = videos()
        valid = {v["id"] for v in choices}
        current = {
            c: read_json(channel_state(c) / "control.json") for c in CHANNELS
        }
        ids = {
            c: selected.get(c)
            or current[c].get("video_id")
            or next(
                (v["id"] for v in choices if v["name"].startswith(c + "/")),
                None,
            )
            for c in CHANNELS
        }
        if any(i not in valid for i in ids.values()):
            raise ValueError("Unknown video")
        grouped = current["Front"].get("sync_group") and current["Front"].get(
            "sync_group"
        ) == current["Back"].get("sync_group")
        resumable = grouped and all(
            current[c].get("video_id") == ids[c]
            and current[c].get("state") in ("playing", "paused")
            and snapshot(c)["status"].get("phase") not in ("ended", "error")
            for c in CHANNELS
        )
        group = (
            current["Front"].get("sync_group")
            if resumable and action != "restart"
            else uuid.uuid4().hex
        )
        for c in CHANNELS:
            if action in ("pause", "stop"):
                value = current[c]
                value["state"] = "paused" if action == "pause" else "idle"
            elif action == "start" and resumable:
                value = current[c]
                value["state"] = "playing"
            else:
                value = dict(
                    run_id=uuid.uuid4().hex,
                    video_id=ids[c],
                    state="playing",
                    sync_group=group,
                )
            value["updated_at"] = time.time()
            atomic_json(channel_state(c) / "control.json", value)
        for c in CHANNELS:
            if action not in ("pause", "stop"):
                _ensure_worker(c)
        return {c: snapshot(c) for c in CHANNELS}


def shutdown():
    with _sequence_lock:
        _stop_sequence("Сервер остановлен")
    for channel, process in _processes.items():
        directory = channel_state(channel)
        current = read_json(directory / "control.json")
        current["state"] = "shutdown"
        atomic_json(directory / "control.json", current)
    for process in _processes.values():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


# One coordinator for the existing paired workers; no additional inference
# path.
_sequence_lock = threading.RLock()
_sequence_thread = None
_sequence = None
_TOTAL_FIELDS = (
    "processed_frames",
    "detection_count",
    "ocr_readings",
    "unique_events",
    "unique_confirmed_events",
    "skipped_processing_frames",
    "dropped_preview",
)


def demo_pairs():
    import re

    grouped = {}
    for video in videos():
        name = video["name"]
        camera = name.split("/")[0]
        match = re.search(r"(\d{2}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2})", name)
        if camera in CHANNELS and match:
            pair = grouped.setdefault(match[1], {})
            if camera in pair:
                raise ValueError("Ambiguous video pair: " + match[1])
            pair[camera] = video["id"]
    slots = ("12.30.00-12.33.00", "12.33.00-12.36.00", "12.36.00-12.39.00")
    if any(set(grouped.get(slot, {})) != set(CHANNELS) for slot in slots):
        raise ValueError("All three Front/Back video pairs are required")
    return [dict(label=slot, videos=grouped[slot]) for slot in slots]


def _pair_status(sequence):
    result = {}
    for camera in CHANNELS:
        value = read_json(channel_state(camera) / "status.json")
        result[camera] = (
            value
            if value.get("run_id") == sequence["run_ids"].get(camera)
            else {}
        )
    return result


def _sequence_view():
    if _sequence is None:
        saved = read_json(STATE / "sequence.json")
        if saved.get("state") in ("playing", "paused"):
            # An API restart never silently resumes an old playlist.
            saved = dict(
                saved,
                state="interrupted",
                message="Запуск прерван перезапуском сервера;"
                " нажмите Start All",
            )
        return saved or dict(
            state="idle",
            pair_number=0,
            pair_count=3,
            progress_percent=0,
            totals={},
        )
    value = json.loads(json.dumps(_sequence))
    statuses = value.get("terminal_channels") or _pair_status(value)
    for c in CHANNELS:
        if not statuses[c]:
            cached = value.get("last_status", {}).get(c, {})
            if cached.get("run_id") == value["run_ids"].get(c):
                statuses[c] = cached
    totals = {
        key: sum(
            pair["totals"].get(key, 0) for pair in value["completed_pairs"]
        )
        for key in _TOTAL_FIELDS
    }
    finished = value["pair_index"] < len(value["completed_pairs"])
    current = {
        key: sum(s.get(key, 0) for s in statuses.values())
        for key in _TOTAL_FIELDS
    }
    if not finished:
        for key in _TOTAL_FIELDS:
            totals[key] += current[key]
    fraction = (
        sum(
            min(1, s.get("processed_frames", 0) / s["source_frames"])
            if s.get("source_frames")
            else 0
            for s in statuses.values()
        )
        / 2
    )
    value.update(
        pair_number=value["pair_index"] + 1,
        pair_count=3,
        current_pair_totals=current,
        totals=totals,
        progress_percent=min(
            100,
            (len(value["completed_pairs"]) + (0 if finished else fraction))
            / 3
            * 100,
        ),
    )
    return value


def restore_paused_sequence():
    """Keep a deliberately paused run paused across an API-only restart."""
    global _sequence, _sequence_thread
    with _sequence_lock:
        if _sequence is not None:
            return
        saved = read_json(STATE / "sequence.json")
        if saved.get("state") == "error":
            # Restore diagnostics for an explicit retry; never start it here.
            _sequence = saved
            return
        if saved.get("state") != "paused":
            return
        if not all(
            read_json(channel_state(c) / "control.json").get("run_id")
            == saved.get("run_ids", {}).get(c)
            and read_json(channel_state(c) / "control.json").get("state")
            == "paused"
            for c in CHANNELS
        ):
            return
        _sequence = saved
        _sequence_thread = threading.Thread(
            target=_sequence_monitor, args=(saved["sequence_id"],), daemon=True
        )
        _sequence_thread.start()


def sequence_snapshot():
    with _sequence_lock:
        return _sequence_view()


def _save_sequence():
    if (
        _sequence["state"] in ("completed", "stopped", "error")
        and "terminal_channels" not in _sequence
    ):
        _sequence["terminal_channels"] = _pair_status(_sequence)
    value = _sequence_view()
    atomic_json(STATE / "sequence.json", value)
    atomic_json(ROOT / _sequence["report_path"], value)


def _launch_sequence_pair():
    pair = _sequence["pairs"][_sequence["pair_index"]]
    result = _control_both("restart", pair["videos"])
    _sequence["run_ids"] = {
        camera: result[camera]["control"]["run_id"] for camera in CHANNELS
    }
    _sequence["pair_started_at"] = time.time()
    _sequence["pair_started_mono"] = time.monotonic()
    _sequence["ready_runs"] = {}
    _sequence["last_status"] = {}


def _stop_sequence(reason="Остановлено пользователем"):
    if _sequence is not None and _sequence["state"] in ("playing", "paused"):
        _sequence.update(
            state="stopped", message=reason, finished_at=time.time()
        )
        _save_sequence()


def _sequence_tick():
    # All callers hold the same lock: EOF, Pause, Stop and manual commands
    # cannot race a next-pair launch. Run UUIDs guard repeated/stale EOFs.
    if _sequence is None or _sequence["state"] != "playing":
        return
    statuses = _pair_status(_sequence)
    for c, s in statuses.items():
        if s:
            _sequence.setdefault("ready_runs", {})[c] = s["run_id"]
            _sequence.setdefault("last_status", {})[c] = s
    if any(s.get("phase") in ("error", "stopped") for s in statuses.values()):
        _sequence.update(
            state="error",
            message="Ошибка потока; автоматический переход отменён",
            finished_at=time.time(),
        )
        _control_both("pause")
        _save_sequence()
        return
    if time.monotonic() - _sequence.get(
        "pair_started_mono", time.monotonic()
    ) > 120 and any(
        not s
        and _sequence.get("ready_runs", {}).get(c) != _sequence["run_ids"][c]
        for c, s in statuses.items()
    ):
        _sequence.update(
            state="error",
            message="Не получен статус запуска потока",
            finished_at=time.time(),
        )
        _control_both("stop")
        _save_sequence()
        return
    if not all(s.get("phase") == "ended" for s in statuses.values()):
        return
    if not all(
        s.get("source_frames", 0) > 0
        and s.get("processed_frames") == s["source_frames"]
        and s.get("skipped_processing_frames") == 0
        for s in statuses.values()
    ):
        _sequence.update(
            state="error",
            message="Неполная обработка пары; переход отменён",
            finished_at=time.time(),
        )
        _save_sequence()
        return
    # finish() writes status before its archive. Wait for both archives too.
    if not all(
        (
            channel_state(c) / "runs" / (_sequence["run_ids"][c] + ".json")
        ).exists()
        for c in CHANNELS
    ):
        return
    _sequence["completed_pairs"].append(
        dict(
            pair_number=_sequence["pair_index"] + 1,
            label=_sequence["pairs"][_sequence["pair_index"]]["label"],
            run_ids=dict(_sequence["run_ids"]),
            totals={
                k: sum(s.get(k, 0) for s in statuses.values())
                for k in _TOTAL_FIELDS
            },
            channels=statuses,
        )
    )
    if _sequence["pair_index"] == 2:
        _sequence.update(
            state="completed",
            message="Демонстрация завершена",
            finished_at=time.time(),
        )
    else:
        _sequence["pair_index"] += 1
        _launch_sequence_pair()
    _save_sequence()


def _sequence_monitor(sequence_id):
    while True:
        with _sequence_lock:
            if (
                _sequence is None
                or _sequence["sequence_id"] != sequence_id
                or _sequence["state"] not in ("playing", "paused")
            ):
                return
            try:
                _sequence_tick()
            except Exception as error:
                _sequence.update(
                    state="error",
                    message="Ошибка управления: " + type(error).__name__,
                    finished_at=time.time(),
                )
                _control_both("stop")
                _save_sequence()
                return
        time.sleep(0.25)


def sequence_control(action):
    global _sequence, _sequence_thread
    with _sequence_lock:
        if action == "start":
            if _sequence is not None and _sequence["state"] in (
                "playing",
                "paused",
            ):
                return (
                    _sequence_view()
                )  # double click does not restart the pair
            pairs = demo_pairs()
            identifier = uuid.uuid4().hex
            _sequence = dict(
                sequence_id=identifier,
                state="playing",
                pair_index=0,
                pairs=pairs,
                completed_pairs=[],
                run_ids={},
                started_at=time.time(),
                message="Общий таймлайн PTS; без пропуска кадров",
                report_path=f"results/video_demo_all/{identifier}/report.json",
                existing_excel="results/anpr_final/unique_plates.xlsx",
            )
            try:
                _launch_sequence_pair()
            except Exception:
                _sequence.update(
                    state="error",
                    message="Не удалось запустить пару",
                    finished_at=time.time(),
                )
                _control_both("stop")
                _save_sequence()
                raise
            _save_sequence()
            _sequence_thread = threading.Thread(
                target=_sequence_monitor, args=(identifier,), daemon=True
            )
            _sequence_thread.start()
        elif action == "retry":
            if not _sequence or _sequence["state"] != "error":
                raise ValueError("Only a failed pair can be retried")
            previous = _sequence_view()
            identifier = uuid.uuid4().hex
            _sequence = json.loads(json.dumps(previous))
            _sequence.update(
                sequence_id=identifier,
                parent_sequence_id=previous["sequence_id"],
                state="playing",
                message="Повтор текущей пары; завершённые пары сохранены",
                report_path=f"results/video_demo_all/{identifier}/report.json",
            )
            for key in ("terminal_channels", "finished_at"):
                _sequence.pop(key, None)
            _sequence.setdefault("failed_attempts", []).append(
                dict(
                    sequence_id=previous["sequence_id"],
                    report_path=previous["report_path"],
                    pair_number=previous["pair_number"],
                    run_ids=previous["run_ids"],
                    totals=previous.get("current_pair_totals", {}),
                )
            )
            _launch_sequence_pair()
            _save_sequence()
            _sequence_thread = threading.Thread(
                target=_sequence_monitor, args=(identifier,), daemon=True
            )
            _sequence_thread.start()
        elif action == "pause":
            if _sequence and _sequence["state"] == "playing":
                _sequence["state"] = "paused"
                _control_both("pause")
                _save_sequence()
        elif action == "resume":
            if _sequence and _sequence["state"] == "paused":
                # A paused EOF remains EOF; do not restart that finished pair.
                for c in CHANNELS:
                    value = read_json(channel_state(c) / "control.json")
                    if value.get("run_id") != _sequence["run_ids"][c]:
                        raise ValueError("Run changed while paused")
                    value["state"] = "playing"
                    atomic_json(channel_state(c) / "control.json", value)
                _sequence["state"] = "playing"
                _sequence_tick()
                _save_sequence()
        elif action == "stop":
            _stop_sequence()
            _control_both("stop")
        else:
            raise ValueError("Unknown sequence action")
        return _sequence_view()


def control(action, video_id=None, channel="Front"):
    with _sequence_lock:
        _stop_sequence("Очередь отменена ручным управлением потоком")
        return _control(action, video_id, channel)


def control_both(action, selected=None):
    with _sequence_lock:
        if (
            _sequence
            and _sequence["state"] in ("playing", "paused")
            and action in ("pause", "start")
        ):
            sequence_control("pause" if action == "pause" else "resume")
            return {c: snapshot(c) for c in CHANNELS}
        _stop_sequence("Очередь отменена ручным запуском пары")
        return _control_both(action, selected)


class DemoEvents:
    """Count existing admitted events by scoped track; never vote or
    merge text."""

    def __init__(self):
        self.events = {}
        self.confirmed = set()
        self.confirmed_records = {}
        self.first_seen = {}

    def submit(self, session, event):
        key = (session, event["track_id"])
        self.first_seen.setdefault(key, event.get("video_time_s"))
        self.events[key] = dict(event)
        if event.get("status") == "CONFIRMED":
            self.confirmed.add(key)
            self.confirmed_records[key] = dict(event)
        return True

    def close(self):
        pass


def build_replay_class():
    from process_video import VideoReplay

    class DemoReplay(VideoReplay):
        def __init__(self, path, metadata, cv, run):
            super().__init__(
                path, STATE, metadata, cv, live=True, preview_path=PREVIEW
            )
            self.session_id = run["run_id"]
            self.run = run
            self.publisher.keep_frame = True
            self.publisher.prefer_confirmed = True
            self.publisher.readings.update(
                session_id=self.session_id, unique_confirmed_events=0
            )
            self.publisher._write_readings()
            self.events = DemoEvents()
            self.sender_factory = lambda *a, **k: self.events
            self.owns_preview = True
            self.previous_start = None
            self.previous_start_pts = None
            self.anchor = None
            self.paused_seconds = 0.0
            self.pause_started = None
            self.last_status = 0.0
            self.frame_lag_ms = 0.0
            self.completed_count = 0
            self.started_processing = None
            self.initialized_mono = time.monotonic()
            self.completed_media_time = 0.0
            self.sync_active = False
            self.sync_wait_ms = 0.0
            self.write_status("playing")

        def active_control(self):
            started = time.monotonic()
            while not self.stop.is_set():
                value = read_json(STATE / "control.json")
                if value.get("run_id") and value.get("state"):
                    break
                if time.monotonic() - started > 120:
                    raise RuntimeError("Control file unavailable")
                self.stop.wait(0.01)
            else:
                return None
            if value.get("run_id") != self.session_id or value.get(
                "state"
            ) in ("idle", "shutdown"):
                self.stop.set()
                return None
            return value

        def wait_playing(self):
            while not self.stop.is_set():
                value = self.active_control()
                if value is None:
                    return False
                if value.get("state") == "playing":
                    if self.pause_started is not None:
                        delta = time.monotonic() - self.pause_started
                        self.paused_seconds += delta
                        if self.anchor is not None:
                            self.anchor += delta
                        if self.previous_start is not None:
                            self.previous_start += delta
                        self.pause_started = None
                    return True
                if self.pause_started is None:
                    self.pause_started = time.monotonic()
                self.write_status("paused")
                self.stop.wait(0.04)
            return False

        def sync_peer(self):
            own = read_json(STATE / "control.json")
            group = own.get("sync_group")
            if not group:
                return None
            peer = STATE.parent / (
                "Back" if STATE.name == "Front" else "Front"
            )
            control = read_json(peer / "control.json")
            if control.get("sync_group") != group:
                return None
            status = read_json(peer / "status.json")
            if status.get("run_id") != control.get("run_id"):
                return {}
            if status.get("phase") == "error":
                raise RuntimeError("Paired stream failed")
            return status

        def wait_sync(self, startup=False):
            started = time.monotonic()
            while not self.stop.is_set():
                if not self.wait_playing():
                    return False
                peer = self.sync_peer()
                self.sync_active = peer is not None
                if peer is None:
                    return True
                if peer.get("initialized_mono") is not None:
                    if startup:
                        self.anchor = max(
                            self.initialized_mono, peer["initialized_mono"]
                        )
                        return True
                    if peer.get("phase") == "ended":
                        return True
                    delta = self.media_time - (
                        self.previous_start_pts
                        if self.previous_start_pts is not None
                        else self.media_time
                    )
                    tolerance = max(delta, 1 / self.source_fps) + 1 / max(
                        peer.get("source_fps", self.source_fps), 1
                    )
                    # One in-flight frame per worker: the
                    # horizon is the sum of source frame
                    # periods, not an arbitrary delay.
                    if (
                        self.media_time
                        <= peer.get("completed_media_time_s", 0)
                        + tolerance
                        + 1e-6
                    ):
                        self.sync_wait_ms += (
                            time.monotonic() - started
                        ) * 1000
                        return True
                if time.monotonic() - started > 120:
                    raise RuntimeError("Paired stream readiness timeout")
                self.stop.wait(0.01)
            return False

        def read(self):
            if not self.wait_playing():
                return False, None
            if self.anchor is None and not self.wait_sync(startup=True):
                return False, None
            ok, frame = super().read()
            if not ok:
                return False, None
            if self.anchor is None:
                self.anchor = time.monotonic() - self.media_time
                self.started_processing = time.monotonic()
            if not self.wait_sync():
                return False, None
            # Each frame waits for its PTS. If ANPR falls behind, retain order,
            # slow playback and report lateness; never silently skip frames.
            while not self.stop.is_set():
                if not self.wait_playing():
                    return False, None
                target = self.anchor + self.media_time
                if self.previous_start is not None:
                    target = max(
                        target,
                        self.previous_start
                        + self.media_time
                        - self.previous_start_pts,
                    )
                remaining = target - time.monotonic()
                if remaining <= 0:
                    break
                self.stop.wait(min(0.02, remaining))
            self.last_frame_timestamp = time.monotonic()
            self.previous_start = self.last_frame_timestamp
            self.previous_start_pts = self.media_time
            return not self.stop.is_set(), frame

        def record_frame(
            self, frame_id, media_time, output, plates, confirmed, total_ms
        ):
            if frame_id != self.frame_count or media_time != self.media_time:
                raise RuntimeError("Frame mismatch")
            self.completed_count += 1
            self.completed_media_time = media_time
            self.statistics.add(
                output.get("worker_timing_ms", output["timing_ms"])
            )
            self.frame_lag_ms = max(
                0.0, (time.monotonic() - self.anchor - media_time) * 1000
            )
            self.publisher.readings.update(
                unique_confirmed_events=len(self.events.confirmed)
            )
            latest = self.publisher.readings.get("last_reading")
            if latest and latest.get("status") == "CONFIRMED":
                self.publisher.readings["last_confirmed_reading"] = dict(
                    latest
                )
            # Same source frame and same result, no overlay from
            # future/older frames.
            self.publisher.submit(
                self.current,
                self.last_frame_timestamp,
                self.capture_fps,
                self.capture_read_ms,
                0,
                frame_id,
            )
            self.write_status("playing", self.sync_active)
            with self.condition:
                self.last_completed = frame_id
                self.condition.notify_all()

        def write_status(self, phase, force=False):
            if not force and time.monotonic() - self.last_status < 0.25:
                return
            self.last_status = time.monotonic()
            atomic_json(
                STATE / "status.json",
                dict(
                    run_id=self.session_id,
                    video_id=self.run["video_id"],
                    phase=phase,
                    source_fps=self.source_fps,
                    media_time_s=self.completed_media_time,
                    completed_media_time_s=self.completed_media_time,
                    initialized_mono=self.initialized_mono,
                    sync_mode=self.sync_active,
                    sync_wait_ms=self.sync_wait_ms,
                    duration_s=self.metadata["duration_s"],
                    source_frames=self.metadata["frames"],
                    processed_frames=self.completed_count,
                    frame_id=self.completed_count,
                    anpr_fps=self.anpr_fps,
                    lag_ms=self.frame_lag_ms,
                    skipped_processing_frames=0,
                    dropped_preview=self.publisher.dropped,
                    unique_confirmed_events=len(self.events.confirmed),
                    unique_events=len(self.events.events),
                    detection_count=self.publisher.readings["detections"],
                    ocr_readings=self.publisher.readings["ocr_readings"],
                    updated_at=time.time(),
                    device="cuda",
                ),
            )

        def record_worker_error(self):
            import traceback

            (STATE / "worker_error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )

        def finish(self):
            current = read_json(STATE / "control.json")
            if current.get("run_id") == self.session_id:
                phase = (
                    "error"
                    if self.worker_error
                    or self.completed_count != self.metadata["frames"]
                    else "ended"
                )
                if current.get("state") in ("idle", "shutdown"):
                    phase = "stopped"
                self.write_status(phase, True)
                state = read_json(STATE / "status.json")
                events = []
                for key, event in self.events.events.items():
                    full = self.events.confirmed_records.get(key)
                    events.append(
                        dict(
                            track_id=key[1],
                            plate=event.get("plate"),
                            status=event.get("status"),
                            confirmed_plate=full.get("plate")
                            if full
                            else None,
                            first_seen_s=self.events.first_seen[key],
                            last_seen_s=event.get("video_time_s"),
                            box=event.get("position"),
                        )
                    )
                atomic_json(
                    STATE / "runs" / (self.session_id + ".json"),
                    dict(
                        state=state,
                        source=str(self.path.relative_to(ROOT)),
                        metadata=self.metadata,
                        readings={
                            k: v
                            for k, v in self.publisher.readings.items()
                            if k
                            not in ("last_reading", "last_confirmed_reading")
                        },
                        events=events,
                    ),
                )

    return DemoReplay


def worker():
    # WSL process can outlive a force-stopped Windows API. Never load a second
    # model/tracker against the same preview/control channel.
    import fcntl

    STATE.mkdir(parents=True, exist_ok=True)
    singleton = (STATE / "worker.lock").open("a")
    try:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("VIDEO DEMO worker already exists", flush=True)
        return
    import signal

    run_id = None
    try:
        import cv2
        import torch

        import camera
        from full_stand.config import StandConfig
        from full_stand.pipeline import PlatePipeline
        from process_video import probe_video

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        print("VIDEO DEMO CUDA:", torch.cuda.get_device_name(0), flush=True)
        config = StandConfig(device="cuda", confidence=0.5)
        config.validate()
        pipeline = PlatePipeline(config)
        replay_class = build_replay_class()
        while True:
            request = read_json(STATE / "control.json")
            if request.get("state") == "shutdown":
                break
            if (
                request.get("state") != "playing"
                or request.get("run_id") == run_id
            ):
                time.sleep(0.08)
                continue
            run_id = request["run_id"]
            selected = next(
                v for v in videos() if v["id"] == request["video_id"]
            )
            path = ROOT / "test_videos" / selected["name"]
            metadata = probe_video(path, cv2)
            replay = replay_class(path, metadata, cv2, request)
            print("VIDEO DEMO RUN", run_id, selected["name"], flush=True)
            handlers = {
                sig: signal.getsignal(sig)
                for sig in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                camera.main(
                    dict(
                        DEVICE="cuda",
                        SHOW_WINDOW="false",
                        API_URL="",
                        VIDEO_SOURCE=str(path),
                    ),
                    replay=replay,
                    pipeline_override=pipeline,
                )
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
                replay.release()
                replay.finish()
    except Exception as exc:
        # No .env or camera URL is used by this module.
        current = read_json(STATE / "control.json")
        atomic_json(
            STATE / "status.json",
            dict(
                run_id=current.get("run_id"),
                phase="error",
                error="VIDEO_DEMO_FAILED: " + type(exc).__name__,
                updated_at=time.time(),
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--channel", choices=CHANNELS, default="Front")
    args = parser.parse_args()
    if args.worker:
        STATE = STATE / args.channel
        PREVIEW = STATE / "preview"
        worker()
