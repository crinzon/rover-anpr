import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from live_preview import PreviewReader


class ClockTests(unittest.TestCase):
    def test_future_wsl_frame_is_visible_but_still_expires(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preview"
            p.write_bytes(
                json.dumps(
                    dict(
                        timestamp=100.2,
                        published_at=100.3,
                        frame_id=7,
                        session_id="s",
                    )
                ).encode()
                + b"\n\xff\xd8test\xff\xd9"
            )
            reader = PreviewReader(p, max_age=2)
            with (
                patch("live_preview.time.time", return_value=100),
                patch("live_preview.time.monotonic", return_value=10),
            ):
                image, meta = reader.read()
                self.assertIsNotNone(image)
                self.assertAlmostEqual(meta["frame_age_ms"], 100)
            with (
                patch("live_preview.time.time", return_value=100),
                patch("live_preview.time.monotonic", return_value=12),
            ):
                self.assertEqual(reader.read(), (None, {}))

    def test_old_frame_not_revived_by_first_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preview"
            p.write_bytes(
                json.dumps(dict(timestamp=90, published_at=90.1)).encode()
                + b"\n\xff\xd8test\xff\xd9"
            )
            with patch("live_preview.time.time", return_value=100):
                self.assertEqual(PreviewReader(p).read(), (None, {}))

    def test_legacy_future_frame_without_producer_time_stays_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preview"
            p.write_bytes(
                json.dumps(dict(timestamp=100.2)).encode()
                + b"\n\xff\xd8test\xff\xd9"
            )
            with patch("live_preview.time.time", return_value=100):
                self.assertEqual(PreviewReader(p).read(), (None, {}))


if __name__ == "__main__":
    unittest.main()
