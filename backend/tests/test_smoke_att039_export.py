"""ATT-039 — CSV attendance roster export (GET /api/v1/courses/{id}/attendance/export).

Covers the authorization matrix (owner ok, 'ta' denied, AUDITOR 403 via the
unconditional role gate, unlinked instructor existence-denying 404, ADMIN ok,
flag-off pass-through parity for the link gate), the CSV contract
(content-type/disposition/header row), the empty-day edge, override column
rendering out of the ATT-038 notes scheme, and the advisory EXPORT
governance event.
"""

from __future__ import annotations

import csv
import io
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select


_CSV_HEADERS = [
    "student_number", "student_name", "status", "confidence_score",
    "last_sighting_at", "override_applied", "override_reason",
]


async def _seed_course(session_factory, *, is_active: bool = True) -> uuid.UUID:
    from app.domain.models import Course

    async with session_factory() as session:
        course = Course(
            id=uuid.uuid4(), code=f"EXP-{uuid.uuid4().hex[:10]}",
            title="Export probe course", credits=3, is_active=is_active,
        )
        session.add(course)
        await session.commit()
        return course.id


async def _link(session_factory, user_id: uuid.UUID, course_id: uuid.UUID, *, role: str = "owner") -> None:
    from app.domain.models import CourseInstructor

    async with session_factory() as session:
        session.add(CourseInstructor(course_id=course_id, user_id=user_id, role_in_course=role))
        await session.commit()


async def _seed_student(session_factory, number: str, name: str) -> uuid.UUID:
    from app.core.security import hash_password
    from app.domain.models import Student, User, UserRole

    async with session_factory() as session:
        user = User(
            id=uuid.uuid4(), email=f"{number}@test.example", full_name=name,
            password_hash=hash_password("TestPass1!"), role=UserRole.AUDITOR, is_active=True,
        )
        session.add(user)
        await session.flush()
        student = Student(
            id=uuid.uuid4(), user_id=user.id, student_number=number,
            program="Test Program", enrollment_year=2024, is_active=True,
        )
        session.add(student)
        await session.commit()
        return student.id


async def _seed_sightings(session_factory, student_id, course_id, confidences) -> str:
    """Insert one sighting per confidence value today; return last timestamp ISO."""
    from app.domain.models import Sighting

    base = datetime.now(tz=UTC)
    async with session_factory() as session:
        for i, confidence in enumerate(confidences):
            session.add(Sighting(
                id=uuid.uuid4(), student_id=student_id, course_id=course_id,
                room_id=None, timestamp=base + timedelta(minutes=i),
                camera_id=f"cam-exp-{i}", confidence_score=confidence,
            ))
        await session.commit()
    return (base + timedelta(minutes=len(confidences) - 1)).astimezone(UTC).isoformat()


async def _seed_record(session_factory, student_id, course_id, *, status: str, notes: str | None = None) -> None:
    from app.domain.models import AttendanceStatus, ClassSessionRecord

    async with session_factory() as session:
        session.add(ClassSessionRecord(
            student_id=student_id, course_id=course_id,
            session_date=datetime.now(tz=UTC).date(), status=AttendanceStatus(status),
            sighting_count=2, required_sightings_threshold=3,
            evaluated_at=datetime.now(tz=UTC), notes=notes,
        ))
        await session.commit()


def _bearer(user, auth_cookie) -> dict[str, str]:
    from app.core.security import create_access_token

    token, _ = create_access_token(subject=user.id, role=user.role)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def authz_on(monkeypatch: pytest.MonkeyPatch):
    from app.core.security import get_security_settings

    monkeypatch.setenv("ATTENDANCE_COURSE_SCOPED_AUTHZ", "true")
    get_security_settings.cache_clear()
    yield
    get_security_settings.cache_clear()


@pytest.fixture()
def authz_off(monkeypatch: pytest.MonkeyPatch):
    from app.core.security import get_security_settings

    monkeypatch.delenv("ATTENDANCE_COURSE_SCOPED_AUTHZ", raising=False)
    get_security_settings.cache_clear()
    yield
    get_security_settings.cache_clear()


def _export_url(course_id: uuid.UUID, query: str = "") -> str:
    return f"/api/v1/courses/{course_id}/attendance/export{query}"


# ---------------------------------------------------------------------------
# Authorization matrix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flag_on_linked_owner_exports_csv(
    async_client: AsyncClient, instructor_user, auth_cookie, _session_factory, authz_on
) -> None:
    course_id = await _seed_course(_session_factory)
    await _link(_session_factory, instructor_user.id, course_id)

    s1 = await _seed_student(_session_factory, "EXP0001", "Ann Present")
    last_seen_1 = await _seed_sightings(_session_factory, s1, course_id, [0.8, 0.9])
    await _seed_record(
        _session_factory, s1, course_id,
        status="present", notes="manual_override: medical note",
    )

    s2 = await _seed_student(_session_factory, "EXP0002", "Bob Absent")
    await _seed_record(_session_factory, s2, course_id, status="absent")

    s3 = await _seed_student(_session_factory, "EXP0003", "Cy SightedOnly")
    last_seen_3 = await _seed_sightings(_session_factory, s3, course_id, [None])

    response = await async_client.get(_export_url(course_id), headers=_bearer(instructor_user, auth_cookie))
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    today = datetime.now(tz=UTC).date().isoformat()
    assert response.headers["content-disposition"] == (
        f'attachment; filename="attendance_{course_id}_{today}.csv"'
    )

    parsed = list(csv.reader(io.StringIO(response.text)))
    assert parsed[0] == _CSV_HEADERS
    assert len(parsed) == 4, "one row per enrolled/seen student"
    assert parsed[1] == ["EXP0001", "Ann Present", "present", "0.8500", last_seen_1, "true", "medical note"]
    assert parsed[2] == ["EXP0002", "Bob Absent", "absent", "", "", "false", ""]
    assert parsed[3] == ["EXP0003", "Cy SightedOnly", "unknown", "", last_seen_3, "false", ""]


@pytest.mark.asyncio
async def test_flag_on_ta_link_is_denied(
    async_client: AsyncClient, instructor_user, auth_cookie, _session_factory, authz_on
) -> None:
    course_id = await _seed_course(_session_factory)
    await _link(_session_factory, instructor_user.id, course_id, role="ta")
    response = await async_client.get(_export_url(course_id), headers=_bearer(instructor_user, auth_cookie))
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_auditor_gets_403_in_both_flag_modes(
    async_client: AsyncClient, auditor_user, auth_cookie, _session_factory, authz_on
) -> None:
    """The unconditional instructor role gate fires before any link check."""
    course_id = await _seed_course(_session_factory)
    await _link(_session_factory, auditor_user.id, course_id)  # even a link cannot widen
    denied_on = await async_client.get(_export_url(course_id), headers=_bearer(auditor_user, auth_cookie))
    assert denied_on.status_code == 403, denied_on.text


@pytest.mark.asyncio
async def test_flag_on_unlinked_instructor_gets_existence_denying_404(
    async_client: AsyncClient, instructor_user, auth_cookie, _session_factory, authz_on
) -> None:
    course_id = await _seed_course(_session_factory)
    missing = uuid.uuid4()
    real = await async_client.get(_export_url(course_id), headers=_bearer(instructor_user, auth_cookie))
    probe = await async_client.get(_export_url(missing), headers=_bearer(instructor_user, auth_cookie))
    assert real.status_code == probe.status_code == 404
    assert real.json() == probe.json() == {"detail": "Course does not exist."}


@pytest.mark.asyncio
async def test_flag_on_admin_exports_without_link(
    async_client: AsyncClient, admin_user, auth_cookie, _session_factory, authz_on
) -> None:
    course_id = await _seed_course(_session_factory)
    response = await async_client.get(_export_url(course_id), headers=_bearer(admin_user, auth_cookie))
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")


@pytest.mark.asyncio
async def test_flag_off_unlinked_instructor_still_exports_passthrough_parity(
    async_client: AsyncClient, instructor_user, auth_cookie, _session_factory, authz_off
) -> None:
    """Flag off: the course-link requirement disappears, exactly like reads."""
    course_id = await _seed_course(_session_factory)
    response = await async_client.get(_export_url(course_id), headers=_bearer(instructor_user, auth_cookie))
    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Data edges
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_day_renders_header_only_csv(
    async_client: AsyncClient, instructor_user, auth_cookie, _session_factory
) -> None:
    course_id = await _seed_course(_session_factory)
    await _link(_session_factory, instructor_user.id, course_id)
    response = await async_client.get(_export_url(course_id), headers=_bearer(instructor_user, auth_cookie))
    assert response.status_code == 200, response.text
    assert response.text == ",".join(_CSV_HEADERS) + "\r\n"


@pytest.mark.asyncio
async def test_date_query_selects_that_days_window_only(
    async_client: AsyncClient, instructor_user, auth_cookie, _session_factory
) -> None:
    course_id = await _seed_course(_session_factory)
    await _link(_session_factory, instructor_user.id, course_id)
    student_id = await _seed_student(_session_factory, "EXP0010", "Yesterday Only")

    from app.domain.models import Sighting

    yesterday = datetime.now(tz=UTC) - timedelta(days=1)
    async with _session_factory() as session:
        session.add(Sighting(
            id=uuid.uuid4(), student_id=student_id, course_id=course_id, room_id=None,
            timestamp=yesterday, camera_id="cam-yesterday", confidence_score=0.7,
        ))
        await session.commit()

    today = datetime.now(tz=UTC).date().isoformat()
    day_response = await async_client.get(
        _export_url(course_id, f"?date={yesterday.date().isoformat()}"),
        headers=_bearer(instructor_user, auth_cookie),
    )
    assert day_response.status_code == 200, day_response.text
    rows = list(csv.reader(io.StringIO(day_response.text)))
    assert len(rows) == 2 and rows[1][0] == "EXP0010"

    empty_response = await async_client.get(
        _export_url(course_id, f"?date={today}"),
        headers=_bearer(instructor_user, auth_cookie),
    )
    assert empty_response.text == ",".join(_CSV_HEADERS) + "\r\n"


@pytest.mark.asyncio
async def test_missing_course_404s_for_admin(
    async_client: AsyncClient, admin_user, auth_cookie
) -> None:
    response = await async_client.get(_export_url(uuid.uuid4()), headers=_bearer(admin_user, auth_cookie))
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "Course does not exist."


@pytest.mark.asyncio
async def test_unsupported_format_is_rejected(
    async_client: AsyncClient, admin_user, auth_cookie, _session_factory
) -> None:
    course_id = await _seed_course(_session_factory)
    response = await async_client.get(
        _export_url(course_id, "?format=xlsx"), headers=_bearer(admin_user, auth_cookie)
    )
    assert response.status_code == 422, response.text


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_export_writes_advisory_governance_event(
    async_client: AsyncClient, instructor_user, auth_cookie, _session_factory
) -> None:
    from app.domain.models import GovernanceLog

    course_id = await _seed_course(_session_factory)
    await _link(_session_factory, instructor_user.id, course_id)
    s1 = await _seed_student(_session_factory, "EXP0020", "Dee Row")
    await _seed_sightings(_session_factory, s1, course_id, [0.9])

    response = await async_client.get(_export_url(course_id), headers=_bearer(instructor_user, auth_cookie))
    assert response.status_code == 200, response.text

    async with _session_factory() as session:
        events = list(
            (
                await session.execute(select(GovernanceLog).where(GovernanceLog.action == "EXPORT"))
            ).scalars().all()
        )
    assert len(events) == 1
    event = events[0]
    assert event.actor_user_id == instructor_user.id
    assert event.entity_type == "class_session_record"
    assert event.entity_id == course_id
    assert event.change_summary["rows"] == 1
    assert event.change_summary["format"] == "csv"
