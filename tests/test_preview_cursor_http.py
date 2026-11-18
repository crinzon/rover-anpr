import unittest
import urllib.request
from unittest.mock import patch

try:
    from . import test_api_http as api_tests
except ImportError:
    import test_api_http as api_tests


@unittest.skipUnless(
    all(api_tests.importlib.util.find_spec(n) for n in ("fastapi", "uvicorn")),
    "FastAPI/Uvicorn unavailable",
)
class CursorHttpTests(unittest.TestCase):
    setUpClass = classmethod(api_tests.ApiHttpTest.setUpClass.__func__)
    tearDownClass = classmethod(api_tests.ApiHttpTest.tearDownClass.__func__)
    setUp = api_tests.ApiHttpTest.setUp
    http = api_tests.ApiHttpTest.http

    def test_frame_cursor_ignores_backward_wall_clock_and_scopes_session(self):
        meta = dict(
            timestamp=400.0, frame_id=8, result_frame_id=8, session_id="A"
        )
        with patch.object(
            self.api,
            "camera_preview",
            return_value=(b"\xff\xd8test\xff\xd9", meta),
        ):
            code, data = self.http(
                "/live/frame?bundle=1&after=500&after_frame=7&session=A"
            )
            self.assertEqual(code, 200)
            self.assertEqual(data["frame_id"], 8)
            with urllib.request.urlopen(
                self.base + "/live/frame?bundle=1&after_frame=8&session=A"
            ) as r:
                self.assertEqual(r.status, 204)
            self.assertEqual(
                self.http("/live/frame?bundle=1&after_frame=100&session=old")[
                    0
                ],
                200,
            )

    def test_legacy_timestamp_cursor_remains_supported(self):
        with patch.object(
            self.api,
            "camera_preview",
            return_value=(
                b"\xff\xd8test\xff\xd9",
                dict(timestamp=500.0, frame_id=8, session_id="A"),
            ),
        ):
            self.assertEqual(
                self.http("/live/frame?bundle=1&after=499")[0], 200
            )
            with urllib.request.urlopen(
                self.base + "/live/frame?bundle=1&after=500"
            ) as r:
                self.assertEqual(r.status, 204)
