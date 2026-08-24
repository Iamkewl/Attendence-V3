"""ATT-012 regression — demo emitter must only write sightings for demo-seeded students.

Without the fix in ATT-012, ``emit_one_synthetic_sighting()`` queried
``Student`` with ``where(is_active=True)`` and *no* filter for demo-seeded
students, so a Sighting row could be written against the demo course for a
non-demo student. The nightly ``task_evaluate_daily_attendance`` aggregation
would then mark that non-demo student PRESENT on the demo course's roster.

This test asserts the fix by seeding a demo course plus only non-demo-ID
active students, then requesting one synthetic sighting. With the fix the
empty demo-roster query returns 0 and writes no Sighting; without the fix a
random non-demo student is picked and a Sighting row appears.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

_DEMO_COURSE_ID = uuid.UUID("00000000-0000-4000-a000-000000000001")
# Two non-demo student IDs — deliberately NOT in _DEMO_STUDENT_IDS so the only
# way emit_one_synthetic_sighting picks them is if the course filter is missing.
_NON_DEMO_STUDENT_IDS = (
    uuid.UUID("aaaaaaaa-0000-4000-a000-000000000aaa"),
    uuid.UUID("bbbbbbbb-0000-4000-a000-000000000bbb"),
)


async def _seed_demo_course_and_non_demo_students(engine: AsyncEngine) -> None:
    """Seed the demo Course and two non-demo Student rows (with backing User rows)."""
    from app.core.security import hash_password
    from app.domain.models import Course, Student, User, UserRole

    factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    async with factory() as session:
        course = Course(
            id=_DEMO_COURSE_ID,
            code="DEMO-101",
            title="Demo Course — ATT-012 regression",
            credits=3,
            is_active=True,
        )
        session.add(course)
        await session.flush()

        for i, student_id in enumerate(_NON_DEMO_STUDENT_IDS):
            user_id = uuid.uuid4()
            user = User(
                id=user_id,
                email=f"non-demo-{i}@att012.test",
                full_name=f"Non-Demo Student {i}",
                password_hash=hash_password("TestPass1!"),
                role=UserRole.AUDITOR,
                is_active=True,
            )
            session.add(user)
            await session.flush()

            student = Student(
                id=student_id,
                user_id=user_id,
                student_number=f"NON-DEMO-{i:02d}",
                program="Non-Demo Program",
                enrollment_year=2024,
                is_active=True,
            )
            session.add(student)

        await session.commit()


@pytest.mark.asyncio
async def test_demo_emitter_refuses_non_demo_students(test_engine: AsyncEngine) -> None:
    """ATT-012: a non-demo student must never be linked to the demo course."""
    from app.domain.models import Sighting
    from app.worker.demo_emitter import emit_one_synthetic_sighting

    await _seed_demo_course_and_non_demo_students(test_engine)

    prior_course_id = os.environ.get("ATTENDANCE_DEMO_COURSE_ID", "")
    prior_demo_mode = os.environ.get("ATTENDANCE_DEMO_MODE", "")
    prior_triton_demo = os.environ.get("ATTENDANCE_TRITON_DEMO_MODE", "")
    os.environ["ATTENDANCE_DEMO_COURSE_ID"] = str(_DEMO_COURSE_ID)
    os.environ["ATTENDANCE_DEMO_MODE"] = "1"
    os.environ["ATTENDANCE_TRITON_DEMO_MODE"] = "1"
    try:
        result = await emit_one_synthetic_sighting()
    finally:
        os.environ.pop("ATTENDANCE_DEMO_COURSE_ID", None)
        os.environ.pop("ATTENDANCE_DEMO_MODE", None)
        os.environ.pop("ATTENDANCE_TRITON_DEMO_MODE", None)
        if prior_course_id:
            os.environ["ATTENDANCE_DEMO_COURSE_ID"] = prior_course_id
        if prior_demo_mode:
            os.environ["ATTENDANCE_DEMO_MODE"] = prior_demo_mode
        if prior_triton_demo:
            os.environ["ATTENDANCE_TRITON_DEMO_MODE"] = prior_triton_demo

    # With the ATT-012 fix: the emitter's query (Student.id.in_(_DEMO_STUDENT_IDS))
    # matches no rows (we seeded only non-demo IDs), so emit returns 0 and writes
    # no Sighting. Without the fix: a random non-demo student would be picked and
    # a Sighting row for course_id == _DEMO_COURSE_ID would land in DB.
    assert result == 0, (
        f"emit_one_synthetic_sighting should return 0 (no demo-seeded students "
        f"in this test's DB); got {result} — the ATT-012 course filter is missing."
    )

    factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    async with factory() as session:
        rows = (
            await session.execute(
                select(Sighting).where(Sighting.course_id == _DEMO_COURSE_ID)
            )
        ).scalars().all()

    assert rows == [], (
        f"No Sighting rows should exist for the demo course when only non-demo "
        f"students are seeded; found {len(rows)} — ATT-012 fix regressed."
    )

    # Sanity: the test DB did see the non-demo Students we seeded, so the
    # 'no sighting' outcome is due to the filter, not "no students" masking it.
    from app.domain.models import Student
    async with factory() as session:
        non_demo_students = (
            await session.execute(
                select(Student).where(Student.id.in_(_NON_DEMO_STUDENT_IDS))
            )
        ).scalars().all()
    assert len(non_demo_students) == len(_NON_DEMO_STUDENT_IDS), (
        "Test setup failed: non-demo Student rows not present in DB."
    )
