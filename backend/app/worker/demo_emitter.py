"""Synthetic sighting emitter for demo-mode dashboard flickering."""

from __future__ import annotations

import logging
import os
import random
import uuid
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select

from app.core.database import get_session_factory
from app.domain.models import Student
from app.services.attendance_service import (
    AttendanceNotFoundError,
    AttendanceService,
    AttendanceValidationError,
)

LOGGER = logging.getLogger(__name__)

_DEMO_CAMERA_ID = "demo-camera-overhead-01"

# Fixed UUIDs of the demo-seeded students. These MUST stay in lock-step with
# scripts/seed_demo_data.py:DEMO_STUDENT_IDS — the demo emitter is only
# permitted to write Sighting rows for students seeded into the demo course,
# so periodic-emit sightings cannot be attributed to non-demo students.
# Hardcoded twice (here and in seed_demo_data.py) rather than imported because
# `scripts/seed_demo_data.py` is not a package import target for `app.worker`
# (it is a standalone management script guarded by `if __name__ == "__main__"`).
_DEMO_STUDENT_IDS: tuple[UUID, ...] = (
    uuid.UUID("00000000-0000-4000-a000-000000000010"),
    uuid.UUID("00000000-0000-4000-a000-000000000011"),
    uuid.UUID("00000000-0000-4000-a000-000000000012"),
    uuid.UUID("00000000-0000-4000-a000-000000000013"),
    uuid.UUID("00000000-0000-4000-a000-000000000014"),
)


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
        # Restrict the random pick to demo-seeded students only. The demo
        # course (ATTENDANCE_DEMO_COURSE_ID) is seeded with a fixed roster
        # (seed_demo_data.DEMO_STUDENT_IDS) and we must never write a Sighting
        # row for a student outside that roster; otherwise the nightly
        # task_evaluate_daily_attendance aggregation will mark non-demo
        # students PRESENT on the demo course.
        student_rows = (
            await session.execute(
                select(Student.id)
                .where(Student.is_active.is_(True))
                .where(Student.id.in_(_DEMO_STUDENT_IDS))
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


__all__ = ["_read_demo_flags", "emit_one_synthetic_sighting"]
