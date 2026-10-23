"""Bounded, best-effort delivery; drop newest on overflow."""

import json
import queue
import threading
import urllib.request
import uuid


class EventSender:
    def __init__(
        self,
        url,
        capacity=256,
        attempts=5,
        initial=0.5,
        maximum=10.0,
        timeout=2.0,
        transport=None,
        logger=print,
    ):
        self.url = url
        self.queue = queue.Queue(capacity)
        if capacity < 1 or attempts < 1:
            raise ValueError("capacity and attempts must be positive")
        import math

        if (
            not all(
                math.isfinite(v) and v > 0 for v in (initial, maximum, timeout)
            )
            or maximum < initial
        ):
            raise ValueError("invalid delivery delays")
        self.attempts, self.initial = attempts, initial
        self.maximum, self.timeout = maximum, timeout
        self.transport = transport or self._post
        self.logger = logger
        self.stop = threading.Event()
        self.guard = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _post(self, payload):
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if json.load(response).get("status") != "ok":
                raise OSError("API did not acknowledge event")

    def submit(self, session_id, plate):
        payload = {
            "session_id": session_id,
            "plates": [dict(plate, event_id=uuid.uuid4().hex)],
        }
        with self.guard:
            if self.stop.is_set():
                self.logger("Delivery loss: sender closed")
                return False
            try:
                self.queue.put_nowait(payload)
                return True
            except queue.Full:
                self.logger("Delivery loss: queue full (drop newest)")
                return False

    def _run(self):
        while not self.stop.is_set():
            try:
                payload = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            delivered = False
            delay = self.initial
            try:
                for attempt in range(self.attempts):
                    if self.stop.is_set():
                        break
                    try:
                        self.transport(payload)
                        delivered = True
                        break
                    except Exception:
                        self.logger("Delivery attempt failed")
                        if attempt + 1 < self.attempts:
                            if self.stop.wait(delay):
                                break
                            delay = min(self.maximum, delay * 2)
                if not delivered:
                    self.logger("Delivery loss: retries exhausted or shutdown")
            finally:
                self.queue.task_done()

    def close(self):
        with self.guard:
            self.stop.set()
        self.thread.join(timeout=self.timeout + 0.2)
        count = 0
        while True:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                count += 1
            except queue.Empty:
                break
        if count:
            self.logger(f"Delivery loss: shutdown discarded {count} events")
