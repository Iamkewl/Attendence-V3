#!/usr/bin/env python3
"""Capture laptop webcam frames and stream them to the V2 inference endpoint."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from uuid import UUID

import cv2
import requests


def _is_uuid(value: str) -> bool:
    """Return True when the input can be parsed as a UUID."""
    try:
        UUID(value)
        return True
    except ValueError:
        return False


def _build_headers(auth_token: str | None) -> dict[str, str]:
    """Build optional request headers for authenticated local testing."""
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers


def _send_frame(
    *,
    endpoint: str,
    frame,
    timeout_seconds: float,
    auth_token: str | None,
    request_id: str,
    camera_id: str,
    course_id: str,
    interval_seconds: int,
) -> None:
    """Encode and upload one frame using multipart form data."""
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        print("[warn] failed to encode frame to JPEG; skipping this interval")
        return

    jpeg_bytes = encoded.tobytes()

    frame_height, frame_width = frame.shape[:2]
    form_data = {
        "frame_id": f"webcam-{int(time.time())}",
        "width": str(frame_width),
        "height": str(frame_height),
        "channels": "3",
        "dtype": "uint8",
        "normalize": "false",
        "request_id": request_id,
        "camera_id": camera_id,
        "confidence_threshold": "0.25",
        "liveness_threshold": "0.5",
        "include_embeddings": "false",
        "priority": "false",
        # Keep a visible dummy metadata marker for local debugging.
        "course_label": course_id,
    }

    # Backend currently expects a UUID for course_id. When the provided value is not
    # a UUID, we keep it as metadata only via course_label instead of sending invalid input.
    if _is_uuid(course_id):
        form_data["course_id"] = course_id

    files = {
        "frame_file": ("webcam_frame.jpg", jpeg_bytes, "image/jpeg"),
    }

    try:
        response = requests.post(
            endpoint,
            data=form_data,
            files=files,
            headers=_build_headers(auth_token),
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        print(f"[error] upload failed: {exc}")
        return

    timestamp = datetime.now(tz=UTC).isoformat()
    if response.ok:
        print(f"[{timestamp}] uploaded frame successfully ({response.status_code})")
        print(response.text)
    else:
        print(f"[{timestamp}] upload returned {response.status_code}")
        print(response.text)
        if response.status_code == 401:
            print("[hint] set ATTENDANCE_BEARER_TOKEN for authenticated endpoint access")

    # Informative heartbeat for visible cadence in local terminal logs.
    print(f"[info] next frame upload in ~{interval_seconds}s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open webcam preview and send one frame every few seconds to FastAPI inference.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/api/v1/inference/stream",
        help="Inference stream endpoint URL.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=5,
        help="Upload interval in seconds.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--camera-id",
        default="laptop-webcam",
        help="Camera identifier sent to backend.",
    )
    parser.add_argument(
        "--request-id",
        default="webcam-local-test",
        help="Request ID marker sent to backend.",
    )
    parser.add_argument(
        "--course-id",
        default="test-course-123",
        help="Dummy metadata marker (or UUID if you want backend to parse course_id).",
    )
    args = parser.parse_args()

    if args.interval_seconds < 1:
        print("[error] --interval-seconds must be >= 1")
        return 1

    auth_token = os.getenv("ATTENDANCE_BEARER_TOKEN")

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("[error] unable to open webcam device index 0")
        return 1

    window_name = "Attendance Webcam Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 360)

    print("[info] webcam opened")
    print(f"[info] streaming to: {args.endpoint}")
    print("[info] press q in the preview window to exit")

    next_send_time = time.monotonic()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("[warn] failed to capture webcam frame")
                time.sleep(0.2)
                continue

            cv2.imshow(window_name, frame)

            if time.monotonic() >= next_send_time:
                _send_frame(
                    endpoint=args.endpoint,
                    frame=frame,
                    timeout_seconds=args.timeout_seconds,
                    auth_token=auth_token,
                    request_id=args.request_id,
                    camera_id=args.camera_id,
                    course_id=args.course_id,
                    interval_seconds=args.interval_seconds,
                )
                next_send_time = time.monotonic() + args.interval_seconds

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
