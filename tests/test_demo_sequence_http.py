import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from . import test_api_http as api_tests
except ImportError:
    import test_api_http as api_tests
import video_demo as d


@unittest.skipUnless(
    all(api_tests.importlib.util.find_spec(n) for n in ("fastapi", "uvicorn")),
    "FastAPI/Uvicorn unavailable",
)
class SequenceHttpTest(unittest.TestCase):
    setUpClass = classmethod(api_tests.ApiHttpTest.setUpClass.__func__)
    tearDownClass = classmethod(api_tests.ApiHttpTest.tearDownClass.__func__)
    setUp = api_tests.ApiHttpTest.setUp
    http = api_tests.ApiHttpTest.http

    def test_playlist_http_pause_stop_and_existing_endpoints(self):
        choices = [
            dict(id=str(2 * i + j), name=f"{c}/{slot}[R].mp4")
            for i, slot in enumerate(
                ("12.30.00-12.33.00", "12.33.00-12.36.00", "12.36.00-12.39.00")
            )
            for j, c in enumerate(d.CHANNELS)
        ]
        with (
            patch.object(self.api, "VIDEO_DEMO_ENABLED", True),
            patch.object(d, "ROOT", Path(self.temp.name)),
            patch.object(d, "STATE", Path(self.temp.name) / "demo"),
            patch.object(d, "_sequence", None),
            patch.object(d, "videos", return_value=choices),
            patch.object(d, "_ensure_worker"),
            patch.object(d.threading.Thread, "start"),
        ):
            self.assertEqual(self.http("/live/demo/all")[1]["state"], "idle")
            code, start = self.http("/live/demo/all", dict(action="start"))
            self.assertEqual(code, 200)
            self.assertEqual(start["pair_number"], 1)
            self.assertEqual(
                self.http("/live/demo/all", dict(action="pause"))[1]["state"],
                "paused",
            )
            self.assertEqual(
                self.http("/live/demo/all", dict(action="resume"))[1][
                    "run_ids"
                ],
                start["run_ids"],
            )
            self.assertEqual(
                self.http("/live/demo/all", dict(action="stop"))[1]["state"],
                "stopped",
            )
            self.assertEqual(
                self.http("/live/demo/all", dict(action="bogus"))[0], 422
            )
            self.assertEqual(self.http("/health"), (200, dict(status="ok")))
            self.assertTrue(self.http("/live/demo")[1]["enabled"])
            self.assertEqual(
                self.http("/live/demo/both", dict(action="pause"))[0], 200
            )

    def test_playlist_disabled_is_not_started(self):
        with patch.object(self.api, "VIDEO_DEMO_ENABLED", False):
            self.assertEqual(
                self.http("/live/demo/all"), (200, dict(enabled=False))
            )
            self.assertEqual(
                self.http("/live/demo/all", dict(action="start"))[0], 409
            )
