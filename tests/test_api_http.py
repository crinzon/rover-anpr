"""Actual HTTP over localhost to Uvicorn/FastAPI; no models or camera."""

import importlib.util
import json
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from current_plates import CurrentPlates
from event_sender import EventSender
from plate_validator import plate_position


@unittest.skipUnless(
    all(importlib.util.find_spec(n) for n in ("fastapi", "uvicorn")),
    "FastAPI/Uvicorn are not installed",
)
class ApiHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import uvicorn

        import database

        cls.temp = tempfile.TemporaryDirectory()
        cls.original_db = database.DB_PATH
        database.DB_PATH = Path(cls.temp.name) / "http.db"
        import api

        cls.api, cls.database = api, database
        cls.original_store = api.current_plates
        database.init_db()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        cls.base = f"http://127.0.0.1:{sock.getsockname()[1]}"
        cls.server = uvicorn.Server(
            uvicorn.Config(api.app, log_level="error", lifespan="off")
        )
        cls.thread = threading.Thread(
            target=cls.server.run, kwargs={"sockets": [sock]}, daemon=True
        )
        cls.thread.start()
        deadline = time.monotonic() + 5
        while not cls.server.started and cls.thread.is_alive():
            if time.monotonic() > deadline:
                raise RuntimeError("HTTP server did not start")
            time.sleep(0.01)
        if not cls.server.started:
            raise RuntimeError("HTTP server stopped during startup")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(5)
        cls.api.current_plates = cls.original_store
        cls.database.DB_PATH = cls.original_db
        cls.temp.cleanup()

    def setUp(self):
        self.wall, self.mono = 1000.0, 10.0
        self.api.current_plates = CurrentPlates(
            wall_clock=lambda: self.wall, monotonic_clock=lambda: self.mono
        )
        self.session = self.id()

    def http(self, path, data=None):
        payload = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request(
            self.base + path,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def event(self, track=1, **changes):
        row = dict(
            track_id=track,
            plate="A123AA77",
            confidence=0.95,
            quality=0.8,
            captured_at=self.wall,
            event_id=f"{self.session}-{track}-{self.wall}",
        )
        row.update(changes)
        return row

    def post(self, *events):
        self.assertEqual(
            self.http(
                "/plates", dict(session_id=self.session, plates=list(events))
            ),
            (200, {"status": "ok"}),
        )

    def test_two_tracks_independent_updates_and_disappearance(self):
        self.post(self.event(1))
        self.wall += 0.5
        self.mono += 0.5
        self.post(self.event(2))
        self.assertEqual(len(self.http("/plates")[1]["plates"]), 2)
        self.wall += 1
        self.mono += 1
        self.post(
            self.event(
                2, position=plate_position([400, 100, 440, 140], 640, 480)
            )
        )
        self.mono += 0.6
        rows = self.http("/plates")[1]["plates"]
        self.assertEqual([p["track_id"] for p in rows], [2])
        self.assertEqual(rows[0]["position"]["offset_x_px"], 100)

    def test_delivery_delay_reduces_ttl_and_old_event_cannot_revive(self):
        self.post(self.event(captured_at=998.5))
        row = self.http("/plates")[1]["plates"][0]
        self.assertEqual(row["captured_at"], 998.5)
        self.assertEqual(row["received_at"], 1000)
        self.mono += 0.5
        self.assertEqual(self.http("/plates")[1]["plates"], [])
        self.post(self.event(event_id="late", captured_at=997))
        self.assertEqual(self.http("/plates")[1]["plates"], [])

    def test_out_of_order_and_duplicate_do_not_replace_or_refresh(self):
        current = self.event(
            position=plate_position([400, 100, 440, 140], 640, 480)
        )
        self.post(current)
        self.wall += 1
        self.mono += 1
        self.post(
            self.event(event_id="older", captured_at=999.5, plate="B123BB77")
        )
        self.post(current)
        row = self.http("/plates")[1]["plates"][0]
        self.assertEqual(row["plate"], "A123AA77")
        self.assertEqual(row["received_at"], 1000)
        history = self.http("/history")[1]["detections"]
        self.assertNotIn("B123BB77", {p["plate"] for p in history})
        self.mono += 1
        self.assertEqual(self.http("/plates")[1]["plates"], [])

    def test_real_sender_transfers_pixel_coordinate_contract(self):
        sender = EventSender(self.base + "/plates", timeout=1)
        try:
            for track, box in enumerate(
                (
                    [400, 100, 440, 140],
                    [0, 0, 40, 40],
                    [600, 440, 640, 480],
                    [300, 220, 340, 260],
                ),
                1,
            ):
                sender.submit(
                    self.session,
                    self.event(track, position=plate_position(box, 640, 480)),
                )
            sender.queue.join()
        finally:
            sender.close()
        status, data = self.http("/plates")
        self.assertEqual(status, 200)
        rows = {p["track_id"]: p for p in data["plates"]}
        self.assertEqual(set(rows), {1, 2, 3, 4})
        self.assertEqual(rows[1]["position"]["offset_x_px"], 100)
        self.assertEqual(rows[1]["position"]["offset_y_px"], -120)
        self.assertEqual(rows[2]["position"]["offset_x_px"], -300)
        self.assertEqual(rows[2]["position"]["offset_y_px"], -220)
        self.assertEqual(rows[3]["position"]["offset_x_px"], 300)
        self.assertEqual(rows[3]["position"]["offset_y_px"], 220)
        self.assertEqual(rows[4]["position"]["offset_x_px"], 0)
        self.assertEqual(rows[4]["position"]["offset_y_px"], 0)
        self.assertEqual(rows[1]["position"]["x"], 0.156)
        self.assertEqual(rows[1]["position"]["y"], -0.25)

    def test_same_track_id_in_different_sessions_and_legacy_sender(self):
        legacy = self.event()
        del legacy["captured_at"]
        self.post(legacy)
        self.assertEqual(
            self.http(
                "/plates",
                dict(
                    session_id="another-session",
                    plates=[
                        dict(legacy, event_id=legacy["event_id"] + "-other")
                    ],
                ),
            )[0],
            200,
        )
        rows = self.http("/plates")[1]["plates"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(
            all(p["timestamp_source"] == "received_at" for p in rows)
        )
        self.assertTrue(all(p["captured_at"] is None for p in rows))

    def test_pending_partial_confirmed_lifecycle_and_live_page(self):
        for status, text in (
            ("PENDING", "A123AA77"),
            ("CONFIRMED", "A123AA77"),
            ("PARTIAL", "A123#A77"),
            ("CONFIRMED", "A123AA77"),
        ):
            self.wall += 0.1
            self.post(
                self.event(
                    plate=text,
                    subtype="type1",
                    status=status,
                    confirmed=status == "CONFIRMED",
                )
            )
            current = self.http("/plates?include_partial=true")[1]["plates"]
            self.assertEqual(len(current), 1)
            self.assertEqual(
                (current[0]["plate"], current[0]["status"]), (text, status)
            )
            history = self.http("/history?include_partial=true")[1][
                "detections"
            ]
            self.assertEqual(
                (history[0]["plate"], history[0]["status"]), (text, status)
            )
            self.assertEqual(
                len(self.http("/plates")[1]["plates"]),
                1 if status == "CONFIRMED" else 0,
            )
        with urllib.request.urlopen(
            self.base + "/live", timeout=3
        ) as response:
            page = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            for camera in ("Front", "Back"):
                self.assertIn(f'src="/live?camera={camera}"', page)
        for camera in ("Front", "Back"):
            with urllib.request.urlopen(
                self.base + f"/live?camera={camera}", timeout=3
            ) as response:
                panel = response.read().decode("utf-8")
                self.assertEqual(response.status, 200)
                self.assertIn("/live/status", panel)
                self.assertIn("p.status", panel)
                self.assertIn("p.track_id", panel)
            code, status = self.http(f"/live/status?camera={camera}")
            self.assertEqual(code, 200)
            self.assertIsInstance(status["available"], bool)
        code, status = self.http("/live/status")
        self.assertEqual(code, 200)
        self.assertIsInstance(status["available"], bool)

    def test_dual_status_keeps_partial_and_confirmed_cameras_separate(self):
        with (
            patch.object(
                self.api.CAMERA_READERS["Front"],
                "read",
                return_value=(
                    b"jpeg-front",
                    {
                        "plates": [
                            dict(track_id=1, text="A123#A77", status="PARTIAL")
                        ]
                    },
                ),
            ),
            patch.object(
                self.api.CAMERA_READERS["Back"],
                "read",
                return_value=(
                    b"jpeg-back",
                    {
                        "plates": [
                            dict(
                                track_id=1, text="B456CC77", status="CONFIRMED"
                            )
                        ]
                    },
                ),
            ),
        ):
            front = self.http("/live/status?camera=Front")
            back = self.http("/live/status?camera=Back")
        self.assertEqual(front[0], 200)
        self.assertEqual(back[0], 200)
        self.assertTrue(front[1]["available"])
        self.assertTrue(back[1]["available"])
        self.assertEqual(
            front[1]["plates"],
            [dict(track_id=1, text="A123#A77", status="PARTIAL")],
        )
        self.assertEqual(
            back[1]["plates"],
            [dict(track_id=1, text="B456CC77", status="CONFIRMED")],
        )
        self.assertEqual(self.http("/live/status?camera=Unknown")[0], 404)

    def test_invalid_coordinate_or_future_timestamp_returns_422(self):
        for event in (
            self.event(position={"offset_x_px": 4}),
            self.event(captured_at=1002),
        ):
            code, _ = self.http(
                "/plates", dict(session_id=self.session, plates=[event])
            )
            self.assertEqual(code, 422)
        self.assertEqual(self.http("/plates")[1]["plates"], [])


if __name__ == "__main__":
    unittest.main()
