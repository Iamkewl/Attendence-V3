"""ATT-016 course-scoped authorization for GET /api/v1/attendance/sessions.

Covers both modes of the ``ATTENDANCE_COURSE_SCOPED_AUTHZ`` flag (decisions
D8-D10):

* flag ON — owner link allows, 'ta' rows are stored-but-denied, unlinked or
  unprivileged callers get an existence-denying 404 (link is checked BEFORE
  any course lookup, so there is no course-ID oracle), ADMIN bypasses the
  link check (and alone can learn that a bogus id is a 404 from the
  service), revoked links fail closed on the very next request, and an
  inactive course yields 409.
* flag OFF — byte-identical legacy behavior: any authenticated user may read
  any roster; only truly missing courses 404.

The settings object hosting the flag is lru_cached, so each mode fixture
clears the cache when entering AND leaving (monkeypatch undo runs after the
fixture finalizer because it was instantiated first).
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient


async def _create_course(
    session_factory,
    *,
    is_active: bool = True,
) -> uuid.UUID:
    from app.domain.models import Course

    async with session_factory() as session:
        course = Course(
            id=uuid.uuid4(),
            code=f"ATT016-{uuid.uuid4().hex[:10]}",
            title="Course-scoped authz probe course",
            credits=3,
            is_active=is_active,
        )
        session.add(course)
        await session.commit()
        return course.id


async def _link_owner(session_factory, user_id: uuid.UUID, course_id: uuid.UUID, *, role: str = "owner") -> None:
    from app.domain.models import CourseInstructor

    async with session_factory() as session:
        session.add(CourseInstructor(course_id=course_id, user_id=user_id, role_in_course=role))
        await session.commit()


async def _unlink(session_factory, user_id: uuid.UUID, course_id: uuid.UUID) -> None:
    from sqlalchemy import delete

    from app.domain.models import CourseInstructor

    async with session_factory() as session:
        await session.execute(
            delete(CourseInstructor).where(
                CourseInstructor.user_id == user_id,
                CourseInstructor.course_id == course_id,
            )
        )
        await session.commit()


@pytest.fixture()
def authz_on(monkeypatch: pytest.MonkeyPatch):
    """Enable the course-scoped authz flag for this test (cache-cleared)."""
    from app.core.security import get_security_settings

    monkeypatch.setenv("ATTENDANCE_COURSE_SCOPED_AUTHZ", "true")
    get_security_settings.cache_clear()
    yield
    get_security_settings.cache_clear()


@pytest.fixture()
def authz_off(monkeypatch: pytest.MonkeyPatch):
    """Pin the flag explicitly off (its documented default)."""
    from app.core.security import get_security_settings

    monkeypatch.delenv("ATTENDANCE_COURSE_SCOPED_AUTHZ", raising=False)
    get_security_settings.cache_clear()
    yield
    get_security_settings.cache_clear()


def _bearer(user, auth_cookie) -> dict[str, str]:
    from app.core.security import create_access_token

    token, _ = create_access_token(subject=user.id, role=user.role)
    return {"Authorization": f"Bearer {token}"}


async def _get_roster(async_client: AsyncClient, course_id: uuid.UUID, headers: dict | None = None):
    return await async_client.get(
        f"/api/v1/attendance/sessions?course_id={course_id}",
        headers=headers or {},
    )


# ---------------------------------------------------------------------------
# Flag ON — authorization matrix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flag_on_linked_owner_reads_roster(
    async_client, instructor_user, auth_cookie, _session_factory, authz_on
) -> None:
    course_id = await _create_course(_session_factory)
    await _link_owner(_session_factory, instructor_user.id, course_id)
    response = await _get_roster(async_client, course_id, _bearer(instructor_user, auth_cookie))
    assert response.status_code == 200, response.text
    assert response.json()["course_id"] == str(course_id)


@pytest.mark.asyncio
async def test_flag_on_ta_link_is_stored_but_denied(
    async_client, instructor_user, auth_cookie, _session_factory, authz_on
) -> None:
    """D10 phase 1: 'ta' rows exist in the DB but grant nothing (fail closed)."""
    course_id = await _create_course(_session_factory)
    await _link_owner(_session_factory, instructor_user.id, course_id, role="ta")
    response = await _get_roster(async_client, course_id, _bearer(instructor_user, auth_cookie))
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_flag_on_unlinked_instructor_gets_existence_denying_404(
    async_client, instructor_user, auth_cookie, _session_factory, authz_on
) -> None:
    """Real course id, no link: denied without revealing that the course exists."""
    course_id = await _create_course(_session_factory)
    response = await _get_roster(async_client, course_id, _bearer(instructor_user, auth_cookie))
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "Course does not exist."


@pytest.mark.asyncio
async def test_flag_on_unlinked_caller_cannot_probe_missing_courses(
    async_client, instructor_user, auth_cookie, _session_factory, authz_on
) -> None:
    """No course-ID oracle: nonexistent id produces the SAME denial as a real one."""
    real_course_id = await _create_course(_session_factory)
    missing_course_id = uuid.uuid4()
    real = await _get_roster(async_client, real_course_id, _bearer(instructor_user, auth_cookie))
    missing = await _get_roster(async_client, missing_course_id, _bearer(instructor_user, auth_cookie))
    assert real.status_code == missing.status_code == 404
    assert real.json() == missing.json()


@pytest.mark.asyncio
async def test_flag_on_admin_bypasses_link_check(
    async_client, admin_user, auth_cookie, _session_factory, authz_on
) -> None:
    course_id = await _create_course(_session_factory)
    response = await _get_roster(async_client, course_id, _bearer(admin_user, auth_cookie))
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_flag_on_admin_alone_learns_missing_course_is_404(
    async_client, admin_user, auth_cookie, _session_factory, authz_on
) -> None:
    """Existence information is reserved for principals who would otherwise pass."""
    response = await _get_roster(async_client, uuid.uuid4(), _bearer(admin_user, auth_cookie))
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "Course does not exist."


@pytest.mark.asyncio
async def test_flag_on_operator_denied_even_with_row(
    async_client, operator_user, auth_cookie, _session_factory, authz_on
) -> None:
    """OPERATOR keeps exactly its current surface; rows cannot widen it."""
    course_id = await _create_course(_session_factory)
    await _link_owner(_session_factory, operator_user.id, course_id)
    response = await _get_roster(async_client, course_id, _bearer(operator_user, auth_cookie))
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_flag_on_auditor_denied(
    async_client, auditor_user, auth_cookie, _session_factory, authz_on
) -> None:
    """Phase 1 narrows AUDITOR global read on this route (Q5 revisitable)."""
    course_id = await _create_course(_session_factory)
    response = await _get_roster(async_client, course_id, _bearer(auditor_user, auth_cookie))
    assert response.status_code == 404, response.text


# ---------------------------------------------------------------------------
# Flag ON — revocation, inactive courses, schema constraint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flag_on_revoked_link_fails_closed_on_next_request(
    async_client, instructor_user, auth_cookie, _session_factory, authz_on
) -> None:
    """Link re-read per request: deleting it mid-session denies immediately."""
    course_id = await _create_course(_session_factory)
    await _link_owner(_session_factory, instructor_user.id, course_id)
    before = await _get_roster(async_client, course_id, _bearer(instructor_user, auth_cookie))
    assert before.status_code == 200, before.text
    await _unlink(_session_factory, instructor_user.id, course_id)
    after = await _get_roster(async_client, course_id, _bearer(instructor_user, auth_cookie))
    assert after.status_code == 404, after.text


@pytest.mark.asyncio
async def test_flag_on_inactive_course_yields_409_not_500(
    async_client, instructor_user, auth_cookie, _session_factory, authz_on
) -> None:
    course_id = await _create_course(_session_factory, is_active=False)
    await _link_owner(_session_factory, instructor_user.id, course_id)
    response = await _get_roster(async_client, course_id, _bearer(instructor_user, auth_cookie))
    assert response.status_code == 409, response.text


@pytest.mark.asyncio
async def test_role_check_constraint_rejects_unknown_roles(_session_factory) -> None:
    """The CHECK constraint backs the stored-but-denied TA policy surface."""
    from sqlalchemy.exc import IntegrityError

    from app.domain.models import CourseInstructor

    async with _session_factory() as session:
        session.add(
            CourseInstructor(
                course_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                role_in_course="professor",
            )
        )
        with pytest.raises(IntegrityError) as excinfo:
            await session.commit()
        assert "course_instructor_role_valid" in str(excinfo.value.orig)


# ---------------------------------------------------------------------------
# Flag OFF — regression: behavior byte-identical to pre-ATT-016
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flag_off_unlinked_instructor_still_reads_any_roster(
    async_client, instructor_user, auth_cookie, _session_factory, authz_off
) -> None:
    course_id = await _create_course(_session_factory)
    response = await _get_roster(async_client, course_id, _bearer(instructor_user, auth_cookie))
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_flag_off_operator_still_reads_any_roster(
    async_client, operator_user, auth_cookie, _session_factory, authz_off
) -> None:
    course_id = await _create_course(_session_factory)
    response = await _get_roster(async_client, course_id, _bearer(operator_user, auth_cookie))
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_flag_off_auditor_still_reads_any_roster(
    async_client, auditor_user, auth_cookie, _session_factory, authz_off
) -> None:
    course_id = await _create_course(_session_factory)
    response = await _get_roster(async_client, course_id, _bearer(auditor_user, auth_cookie))
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_flag_off_missing_course_still_404s_for_admin(
    async_client, admin_user, auth_cookie, _session_factory, authz_off
) -> None:
    response = await _get_roster(async_client, uuid.uuid4(), _bearer(admin_user, auth_cookie))
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_inactive_course_maps_to_409_in_both_modes(
    async_client, admin_user, auth_cookie, _session_factory, authz_off
) -> None:
    """Q6 bug fix is unconditional: inactive course never resurfaces as 500."""
    course_id = await _create_course(_session_factory, is_active=False)
    response = await _get_roster(async_client, course_id, _bearer(admin_user, auth_cookie))
    assert response.status_code == 409, response.text


def test_flag_defaults_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rollout safety (D9): absent env var must leave legacy behavior on."""
    from app.core.security import get_security_settings

    monkeypatch.delenv("ATTENDANCE_COURSE_SCOPED_AUTHZ", raising=False)
    get_security_settings.cache_clear()
    try:
        assert get_security_settings().course_scoped_authz_enabled is False
    finally:
        get_security_settings.cache_clear()


def test_flag_parses_truthy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.security import get_security_settings

    for raw in ("true", "1", "YES"):
        monkeypatch.setenv("ATTENDANCE_COURSE_SCOPED_AUTHZ", raw)
        get_security_settings.cache_clear()
        try:
            assert get_security_settings().course_scoped_authz_enabled is True, raw
        finally:
            get_security_settings.cache_clear()
