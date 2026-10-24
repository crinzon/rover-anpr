import importlib
import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import database
from event_sender import EventSender


class DeliveryTest(unittest.TestCase):
    def test_recovery_keeps_event_id(self):
        calls = []

        def transport(payload):
            calls.append(payload["plates"][0]["event_id"])
            if len(calls) < 3:
                raise OSError("secret")

        logs = []
        sender = EventSender(
            "unused",
            initial=0.001,
            maximum=0.002,
            transport=transport,
            logger=logs.append,
        )
        try:
            sender.submit("session", {"plate": "A123AA77"})
            sender.queue.join()
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(set(calls)), 1)
            self.assertNotIn("secret", "".join(logs))
        finally:
            sender.close()

    def test_overflow_and_shutdown(self):
        entered, release = threading.Event(), threading.Event()

        def transport(payload):
            entered.set()
            release.wait(1)

        logs = []
        sender = EventSender(
            "unused",
            capacity=1,
            timeout=0.01,
            transport=transport,
            logger=logs.append,
        )
        try:
            self.assertTrue(sender.submit("s", {}))
            self.assertTrue(entered.wait(1))
            self.assertTrue(sender.submit("s", {}))
            self.assertFalse(sender.submit("s", {}))
            sender.close()
            self.assertTrue(any("queue full" in line for line in logs))
            self.assertTrue(any("shutdown discarded" in line for line in logs))
        finally:
            release.set()
            sender.thread.join(1)

    def test_exhaustion(self):
        calls = []

        def transport(payload):
            calls.append(1)
            raise OSError

        logs = []
        sender = EventSender(
            "unused",
            attempts=2,
            initial=0.001,
            transport=transport,
            logger=logs.append,
        )
        try:
            sender.submit("s", {})
            sender.queue.join()
            self.assertEqual(len(calls), 2)
            self.assertTrue(any("Delivery loss" in line for line in logs))
        finally:
            sender.close()

    def test_sqlite_restart_and_legacy_history(self):
        original = database.DB_PATH
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.db"
            try:
                database.DB_PATH = path
                database.init_db()
                legacy = database.save_detection("A111AA77", 0.9)
                plate = dict(
                    event_id="unique",
                    track_id=1,
                    plate="A123AA77",
                    confidence=0.95,
                    quality=0.5,
                )
                self.assertTrue(database.apply_event("s", plate))
                self.assertFalse(database.apply_event("s", plate))
                importlib.reload(database)
                database.DB_PATH = path
                database.init_db()
                self.assertFalse(database.apply_event("s", plate))
                plate.update(event_id="next", quality=0.9)
                self.assertTrue(database.apply_event("s", plate))
                rows = database.get_detections()
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[-1]["id"], legacy)
            finally:
                database.DB_PATH = original


@unittest.skipUnless(
    importlib.util.find_spec("fastapi"), "FastAPI unavailable"
)
class ApiContractTest(unittest.TestCase):
    def test_confirmed_event_history_position_and_expiry(self):
        import api

        previous = api.latest_plates, api.last_detection_time
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(database, "DB_PATH", Path(directory) / "api.db"):
                database.init_db()
                event = dict(
                    event_id="api-event",
                    track_id=1,
                    plate="A123AA77",
                    confidence=0.95,
                    quality=0.7,
                    position=dict(x=0.1, y=-0.2, direction="CENTER"),
                )
                try:
                    self.assertEqual(api.update_plates({})["status"], "error")
                    with patch("api.time.monotonic", return_value=10):
                        for _ in range(2):
                            self.assertEqual(
                                api.update_plates(
                                    dict(session_id="test", plates=[event])
                                ),
                                {"status": "ok"},
                            )
                        self.assertEqual(api.get_plates()["plates"], [event])
                    with patch("api.time.monotonic", return_value=13):
                        self.assertEqual(api.get_plates(), {"plates": []})
                    history = api.get_history()["detections"]
                    self.assertEqual(len(history), 1)
                    self.assertEqual(history[0]["plate"], "A123AA77")
                    self.assertNotIn("position", history[0])
                finally:
                    api.latest_plates, api.last_detection_time = previous
