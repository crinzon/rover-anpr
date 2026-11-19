"""Experimental ANPR -> ROS 2 String bridge, NOT a rover protocol.

Uses existing GET /plates snapshots or saved receiver events.jsonl.
No rclpy import until real ROS mode is explicitly requested.
"""

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def message(session_id, event, camera_id=None):
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Existing session_id required")
    if not isinstance(event, dict) or event.get("track_id") is None:
        raise ValueError("Existing track_id required")
    if event.get("status") not in (
        "PENDING",
        "PARTIAL",
        "CONFIRMED",
        "REJECTED",
        "UNREADABLE",
        "UNTRACKED",
    ):
        raise ValueError("Unknown ANPR status")
    if not isinstance(event.get("plate"), str):
        raise ValueError("Existing plate string required")
    captured = event.get("captured_at")
    if captured is not None and (
        isinstance(captured, bool)
        or not isinstance(captured, (float, int))
        or not math.isfinite(captured)
    ):
        raise ValueError("Invalid captured_at")
    row = dict(
        schema="rover_anpr.experimental.v1",
        not_official_rover_protocol=True,
        session_id=session_id,
        camera_id=event.get("camera_id") or camera_id,
        camera_id_source=(
            "ANPR_EVENT"
            if event.get("camera_id")
            else "ADAPTER_CONFIGURATION"
            if camera_id
            else "UNAVAILABLE"
        ),
        event=deepcopy(event),
    )
    # Keep the original field names/units; no invented X/Y, confidence, or
    # time.
    json.dumps(row, allow_nan=False)
    return row


class Bridge:
    def __init__(self, publish, camera_id=None, limit=4096):
        self.publish = publish
        self.camera_id = camera_id
        self.seen = OrderedDict()
        self.limit = limit

    def accept(self, session, event):
        row = message(session, event, self.camera_id)
        key = (
            session,
            event.get("event_id")
            or json.dumps(event, sort_keys=True, allow_nan=False),
        )
        if key in self.seen:
            return False
        self.publish(
            row
        )  # failed publication is not acknowledged by this bridge
        self.seen[key] = None
        if len(self.seen) > self.limit:
            self.seen.popitem(last=False)
        return True


def poll_api(bridge, address, timeout=2):
    with urlopen(address, timeout=timeout) as response:
        body = json.load(response)
    if not isinstance(body.get("plates"), list):
        raise ValueError("Existing /plates response expected")
    count = 0
    for event in body["plates"]:
        count += bridge.accept(event.get("session_id"), event)
    return count


def replay(bridge, path):
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if "event" in row:  # existing receiver journal
                count += bridge.accept(row["session_id"], row["event"])
            else:  # existing EventSender envelope, no alternative event protocol  # noqa: E501
                for event in row["plates"]:
                    count += bridge.accept(row["session_id"], event)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input", type=Path, help="Saved ANPR receiver/EventSender JSONL"
    )
    source.add_argument(
        "--api-url",
        default=None,
        help="Existing GET /plates?include_partial=true",
    )
    source.add_argument(
        "--self-test",
        action="store_true",
        help="Explicit synthetic transport fixture, no model test",
    )
    parser.add_argument(
        "--camera-id",
        help="Only set for a known single-camera input; never inferred from track_id",  # noqa: E501
    )
    parser.add_argument("--topic", default="/anpr/experimental/events")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write JSON to stdout; no ROS/DDS publication claimed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new dry-run JSONL, never overwrite",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--duration", type=float, default=0, help="0 = until Ctrl+C"
    )
    parser.add_argument("--poll-interval", type=float, default=0.2)
    args = parser.parse_args()
    if args.poll_interval <= 0 or args.duration < 0:
        parser.error("Invalid duration/interval")
    if args.output and not args.dry_run:
        parser.error("--output is a dry-run journal")
    stream = args.output.open("x", encoding="utf-8") if args.output else None
    node = ros = None
    if not args.dry_run:
        try:
            import rclpy as ros
            from rclpy.node import Node
            from std_msgs.msg import String
        except ImportError:
            parser.exit(
                2,
                "ROS 2/rclpy unavailable. Source Humble on Ubuntu; use --dry-run to test serialization. No ROS publication performed.\n",  # noqa: E501
            )
        ros.init(args=None)
        node = Node("rover_anpr_experimental_bridge")
        publisher = node.create_publisher(String, args.topic, 10)

    def publish(row):
        payload = json.dumps(
            row, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        if args.dry_run:
            print(payload, flush=True)
            if stream:
                stream.write(payload + "\n")
                stream.flush()
        else:
            packet = String()
            packet.data = payload
            publisher.publish(packet)
            ros.spin_once(node, timeout_sec=0)

    bridge = Bridge(publish, args.camera_id)
    try:
        if args.self_test:
            # Explicit test event, not a detected or confirmed real car.
            bridge.accept(
                "SELF_TEST_NOT_A_CAMERA",
                dict(
                    event_id="transport-self-test",
                    track_id="TEST_ONLY",
                    plate="K477##797",
                    status="PARTIAL",
                    captured_at=time.time(),
                    test_event=True,
                ),
            )
            if node:
                end = time.monotonic() + 2
                while time.monotonic() < end:
                    ros.spin_once(node, timeout_sec=0.1)
        elif args.input:
            replay(bridge, args.input)
            if node:
                end = time.monotonic() + 1
                while time.monotonic() < end:
                    ros.spin_once(node, timeout_sec=0.1)
        else:
            start = time.monotonic()
            while (
                not args.duration or time.monotonic() - start < args.duration
            ):
                try:
                    poll_api(
                        bridge,
                        args.api_url
                        or "http://127.0.0.1:8000/plates?include_partial=true",
                    )
                except (OSError, URLError, ValueError) as error:
                    print(
                        "ANPR snapshot unavailable; retrying: " + str(error),
                        file=sys.stderr,
                        flush=True,
                    )
                if args.once:
                    break
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        if stream:
            stream.close()
        if node:
            node.destroy_node()
            ros.shutdown()


if __name__ == "__main__":
    main()
