"""Flow 7 — daily attendance aggregation marks a seeded student PRESENT."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


async def _seed_course_student_sightings(engine: AsyncEngine, *, sighting_count: int = 3):
    """Insert Course, User, Student, and sighting_count Sighting rows for today."""
    from app.domain.models import Course, Sighting, Student, User, UserRole
    from app.core.security import hash_password

    factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    now = datetime.now(tz=UTC)

    async with factory() as session:
        user = User(
            id=uuid.uuid4(),
            email="agg_student@test.example",
            full_name="Aggregation Student",
            password_hash=hash_password("TestPass1!"),
            role=UserRole.AUDITOR,
            is_active=True,
        )
        session.add(user)
        await session.flush()

        course = Course(
            id=uuid.uuid4(),
            code="AGG101",
            title="Aggregation Test Course",
            credits=3,
            is_active=True,
        )
        session.add(course)
        await session.flush()

        student = Student(
            id=uuid.uuid4(),
            user_id=user.id,
            student_number="AGG0001",
            program="Test Program",
            enrollment_year=2024,
            is_active=True,
        )
        session.add(student)
        await session.flush()

        for i in range(sighting_count):
            sighting = Sighting(
                id=uuid.uuid4(),
                student_id=student.id,
                course_id=course.id,
                room_id=None,
                timestamp=now,
                camera_id=f"cam-agg-{i}",
                confidence_score=0.9,
            )
            session.add(sighting)

        await session.commit()

    return course, student


@pytest.mark.asyncio
async def test_daily_evaluation_marks_present(test_engine: AsyncEngine) -> None:
    from app.domain.models import AttendanceStatus, ClassSessionRecord
    # Call the underlying async function directly — Celery tasks use asyncio.run()
    # which cannot be called from within an already-running event loop.
    from app.worker.tasks import _evaluate_daily_attendance

    course, student = await _seed_course_student_sightings(test_engine, sighting_count=3)

    summary = await _evaluate_daily_attendance(required_sightings_threshold=3)

    assert summary.get("records_upserted", 0) >= 1, (
        f"Expected at least one upserted record, got: {summary}"
    )

    factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    async with factory() as session:
        row = (
            await session.execute(
                select(ClassSessionRecord).where(
                    ClassSessionRecord.student_id == student.id,
                    ClassSessionRecord.course_id == course.id,
                )
            )
        ).scalar_one_or_none()

    assert row is not None, "No ClassSessionRecord found after aggregation"
    assert row.status == AttendanceStatus.PRESENT, (
        f"Expected PRESENT, got {row.status}"
    )
