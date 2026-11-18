import unittest

from current_plates import CurrentPlates
from plate_validator import plate_position


class CurrentPlatesTest(unittest.TestCase):
    def setUp(self):
        self.wall, self.mono = 1000.0, 10.0
        self.store = CurrentPlates(
            wall_clock=lambda: self.wall, monotonic_clock=lambda: self.mono
        )
        self.saved = []

    def event(self, **changes):
        return dict(
            dict(
                track_id=1,
                plate="A123AA77",
                confidence=0.95,
                quality=0.8,
                captured_at=self.wall,
            ),
            **changes,
        )

    def save(self, session, plate):
        self.saved.append((session, plate))
        return True

    def test_multiple_tracks_and_individual_expiry(self):
        self.store.update("s", [self.event()], self.save)
        self.wall += 1
        self.mono += 1
        self.store.update("s", [self.event(track_id=2)], self.save)
        self.assertEqual(len(self.store.snapshot()), 2)
        self.mono += 1
        self.assertEqual([p["track_id"] for p in self.store.snapshot()], [2])

    def test_capture_age_reduces_remaining_ttl(self):
        self.store.update("s", [self.event(captured_at=998.5)], self.save)
        self.assertEqual(self.store.snapshot()[0]["received_at"], 1000)
        self.mono += 0.5
        self.assertEqual(self.store.snapshot(), [])

    def test_expired_and_out_of_order_do_not_persist(self):
        self.store.update("s", [self.event()], self.save)
        self.store.update("s", [self.event(captured_at=999.5)], self.save)
        self.store.update(
            "s", [self.event(track_id=2, captured_at=998)], self.save
        )
        self.assertEqual(len(self.saved), 1)
        self.assertEqual(len(self.store.snapshot()), 1)

    def test_duplicate_does_not_refresh_deadline(self):
        self.store.update("s", [self.event(event_id="a")], self.save)
        self.wall += 1
        self.mono += 1
        self.store.update("s", [self.event(event_id="a")], lambda *args: False)
        self.mono += 1
        self.assertEqual(self.store.snapshot(), [])

    def test_session_scoping_and_legacy_payload(self):
        legacy = self.event()
        del legacy["captured_at"]
        self.store.update("one", [legacy], self.save)
        self.store.update("two", [legacy], self.save)
        self.assertEqual(len(self.store.snapshot()), 2)
        row = self.store.snapshot()[0]
        self.assertIsNone(row["captured_at"])
        self.assertEqual(row["timestamp_source"], "received_at")

    def test_invalid_batch_is_not_partially_applied(self):
        with self.assertRaises(ValueError):
            self.store.update(
                "s",
                [
                    self.event(),
                    self.event(track_id=2, position={"offset_x_px": 1}),
                ],
                self.save,
            )
        self.assertEqual(self.saved, [])
        for stamp in (float("nan"), float("inf"), True, 1002, 10**1000):
            with self.assertRaises(ValueError):
                self.store.update(
                    "s", [self.event(captured_at=stamp)], self.save
                )

    def test_wall_clock_jump_does_not_extend_lifetime(self):
        self.store.update("s", [self.event()], self.save)
        self.wall -= 100
        self.mono += 2
        self.assertEqual(self.store.snapshot(), [])

    def test_pixel_offsets_and_legacy_normalized_coordinates(self):
        cases = [
            ([300, 220, 340, 260], 640, 480, 0, 0),
            ([400, 100, 440, 140], 640, 480, 100, -120),
            ([0, 0, 40, 40], 640, 480, -300, -220),
            ([600, 440, 640, 480], 640, 480, 300, 220),
            ([2000, 900, 2096, 960], 4096, 1860, 0, 0),
        ]
        for box, width, height, dx, dy in cases:
            with self.subTest(box=box):
                position = plate_position(box, width, height)
                self.assertEqual(position["offset_x_px"], dx)
                self.assertEqual(position["offset_y_px"], dy)
                self.assertEqual(position["x"], round(dx / width, 3))
                self.assertEqual(position["y"], round(dy / height, 3))


if __name__ == "__main__":
    unittest.main()
