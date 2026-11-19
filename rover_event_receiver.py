"""Local test receiver for current ANPR events; NOT a rover protocol.

Run: python -B rover_event_receiver.py --output results/receiver_demo
"""

import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from current_plates import validate_plate


class Receiver(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, output, drop_ack_once=False):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.guard = threading.Lock()
        self.seen = set()
        self.drop_ack_once = drop_ack_once
        events = self.output / "events.jsonl"
        if events.exists():
            for line in events.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.seen.add((row["session_id"], row["event"]["event_id"]))
        super().__init__(address, Handler)

    def append(self, name, row):
        with (self.output / name).open("a", encoding="utf-8") as target:
            target.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def response(self, code, row):
        data = json.dumps(row).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/health":
            return self.response(404, {"status": "error"})
        return self.response(
            200,
            {
                "status": "ok",
                "test_receiver": True,
                "events": len(self.server.seen),
            },
        )

    def do_POST(self):
        if self.path != "/plates":
            return self.response(404, {"status": "error"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1_000_000:
                raise ValueError("invalid body size")
            body = json.loads(self.rfile.read(size))
            session = body["session_id"]
            plates = body["plates"]
            if (
                not isinstance(session, str)
                or not session
                or not isinstance(plates, list)
            ):
                raise ValueError("existing ANPR session_id/plates required")
            for event in plates:
                validate_plate(event, time.time())
                if (
                    not isinstance(event.get("event_id"), str)
                    or not event["event_id"]
                ):
                    raise ValueError(
                        "event_id required for test deduplication"
                    )
            duplicate = 0
            with self.server.guard:
                for event in plates:
                    key = (session, event["event_id"])
                    if key in self.server.seen:
                        duplicate += 1
                    else:
                        self.server.append(
                            "events.jsonl",
                            dict(
                                received_at=time.time(),
                                session_id=session,
                                event=event,
                            ),
                        )
                        self.server.seen.add(key)
                self.server.append(
                    "requests.jsonl",
                    dict(
                        received_at=time.time(),
                        duplicate_events=duplicate,
                        payload=body,
                    ),
                )
                drop = self.server.drop_ack_once
                self.server.drop_ack_once = False
            if drop:
                self.close_connection = True
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.connection.close()
                return
            self.response(200, {"status": "ok", "duplicate_events": duplicate})
        except (
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            self.response(422, {"status": "error", "detail": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--drop-ack-once",
        action="store_true",
        help="Test ACK loss after accepting first event",
    )
    args = parser.parse_args()
    server = Receiver((args.host, args.port), args.output, args.drop_ack_once)
    print(
        f"TEST receiver: http://{args.host}:{server.server_port}/plates (not rover protocol)",  # noqa: E501
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
