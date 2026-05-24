"""Synthetic sighting emitter for demo-mode dashboard flickering."""

from __future__ import annotations

import logging
import os
import random
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select

from app.core.database import get_session_factory
from app.domain.models import Student
from app.services.attendance_service import AttendanceNotFoundError, AttendanceService, AttendanceValidationError


LOGGER = logging.getLogger(__name__)

_DEMO_CAMERA_ID = "demo-camera-overhead-01"


def _read_demo_flags() -> tuple[bool, bool, int, UUID | None]:
    """Return (demo_mode_enabled, triton_demo_enabled, interval_seconds, course_id)."""
    demo_mode = os.getenv("ATTENDANCE_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
    triton_demo = os.getenv("ATTENDANCE_TRITON_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}

    raw_interval = os.getenv("ATTENDANCE_DEMO_SIGHTING_INTERVAL_SECONDS", "5").strip()
    try:
        interval_seconds = max(1, int(raw_interval))
    except ValueError:
        interval_seconds = 5

    raw_course_id = os.getenv("ATTENDANCE_DEMO_COURSE_ID", "").strip()
    course_id: UUID | None = None
    if raw_course_id:
        try:
            course_id = UUID(raw_course_id)
        except ValueError:
            LOGGER.warning("ATTENDANCE_DEMO_COURSE_ID is not a valid UUID: %r", raw_course_id)

    return demo_mode, triton_demo, interval_seconds, course_id


async def emit_one_synthetic_sighting() -> int:
    """Persist one synthetic Sighting for a random student in the demo course.

    Returns 1 on success, 0 when no students are found (seed hasn't run yet).
    Does not raise; non-fatal errors are logged and return 0.
    """
    _, _, _, course_id = _read_demo_flags()

    if course_id is None:
        LOGGER.warning(
            "ATTENDANCE_DEMO_COURSE_ID is not set; cannot emit synthetic sighting."
        )
        return 0

    session_factory = get_session_factory()

    async with session_factory() as session:
        student_rows = (
            await session.execute(
                select(Student.id)
                .where(Student.is_active.is_(True))
            )
        ).scalars().all()

    if not student_rows:
        LOGGER.warning(
            "No active students found in DB; demo seed may not have run yet."
        )
        return 0

    student_id = random.choice(student_rows)
    confidence_score = round(random.uniform(0.78, 0.97), 4)
    timestamp = datetime.now(tz=UTC)

    async with session_factory() as session:
        attendance_service = AttendanceService(session=session)
        try:
            await attendance_service.log_sighting(
                student_id=student_id,
                camera_id=_DEMO_CAMERA_ID,
                course_id=course_id,
                timestamp=timestamp,
                room_id=None,
                confidence_score=confidence_score,
                embedding_reference=None,
            )
        except (AttendanceNotFoundError, AttendanceValidationError) as exc:
            LOGGER.warning(
                "Synthetic sighting skipped for student_id=%s course_id=%s: %s",
                student_id,
                course_id,
                exc,
            )
            return 0

    return 1


__all__ = ["emit_one_synthetic_sighting", "_read_demo_flags"]
