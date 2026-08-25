"""Attendance session endpoints: roster queries and manual overrides.

Route authorization surface:

``GET /api/v1/attendance/sessions`` (ATT-016 phase 1)
    Guarded by ``CourseScopedPrincipal``. With ``ATTENDANCE_COURSE_SCOPED_AUTHZ``
    unset/false the dependency is a pass-through and this route behaves exactly
    as before ATT-016 (any authenticated user may query any course). When the
    flag is on, ADMIN passes unconditionally while every other role must hold
    an active ``course_instructors`` owner row for the requested course;
    callers without one receive 404 with an existence-denying detail — the link
    check runs before any course lookup so the route cannot be used to probe
    which course ids exist (see ``app.api.deps.get_course_scoped_principal``).

``POST /api/v1/attendance/sessions/{session_id}/override`` (ATT-038)
    A WRITE path, so the role gate is unconditional (independent of the
    ATT-016 flag): only ADMIN or INSTRUCTOR may override, everyone else gets
    403. When the flag is on, INSTRUCTORs additionally need the owner link
    for the target course (reusing the SAME check via
    ``ensure_course_scoped_principal`` — one implementation, no drift) and
    fail closed with the identical existence-denying 404.

``{session_id}`` identifies the attendance session by its course id: a
session in this domain is a (course, date) pair and the roster route already
keys sessions by course id. The override always targets TODAY's aggregated
row for (student, course) and emits the mandatory OVERRIDE_APPLY governance
event with the row's id (see ``AttendanceService.apply_manual_override``).

Error mapping on the roster route: ``AttendanceNotFoundError`` -> 404 and,
fixing a latent bug where an inactive course escaped as HTTP 500,
``AttendanceValidationError`` (e.g. "Course is inactive.") -> 409 in both
flag modes.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    CourseScopedPrincipal,
    CurrentInstructorUser,
    ensure_course_scoped_principal,
)
from app.core.database import get_async_session
from app.domain.models import AttendanceStatus
from app.domain.schemas import (
    ClassSessionListResponse,
    ClassSessionOverrideRead,
    ClassSessionOverrideRequest,
)
from app.services.attendance_service import (
    AttendanceNotFoundError,
    AttendanceService,
    AttendanceValidationError,
)


router = APIRouter(prefix="/attendance", tags=["Attendance"])


@router.get(
    "/sessions",
    response_model=ClassSessionListResponse,
    summary="List Session Records",
    description="Return the attendance roster for a course on a given date.",
)
async def list_session_records(
    _: CourseScopedPrincipal,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    course_id: UUID,
    session_date: date | None = None,
) -> ClassSessionListResponse:
    """Return existing class session records for the requested course and date."""
    service = AttendanceService(session=session)
    try:
        return await service.list_session_records(
            course_id=course_id,
            session_date=session_date or datetime.now(tz=UTC).date(),
        )
    except AttendanceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except AttendanceValidationError as exc:
        # Inactive course previously surfaced as HTTP 500; clients need a
        # distinct retryable signal (design doc Q6).
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post(
    "/sessions/{session_id}/override",
    response_model=ClassSessionOverrideRead,
    summary="Apply Manual Attendance Override",
    description=(
        "Instructor/admin upsert of TODAY's attendance verdict for one "
        "student in the session's course ('present'/'absent', reason "
        "required). Last-write-wins; every application is audited."
    ),
)
async def apply_manual_override(
    session_id: UUID,
    payload: ClassSessionOverrideRequest,
    current_user: CurrentInstructorUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> ClassSessionOverrideRead:
    """Upsert today's class-session record for one student and audit it."""
    # Role gate first (403), then the course-link gate (existence-denying
    # 404 when the ATT-016 flag is on) — mirroring the roster route's
    # ordering so no oracle about course existence leaks to non-instructors.
    await ensure_course_scoped_principal(
        session, current_user, course_id=session_id
    )

    service = AttendanceService(session=session, actor=current_user)
    try:
        record, previous_status = await service.apply_manual_override(
            course_id=session_id,
            student_id=payload.student_id,
            status=AttendanceStatus(payload.status),
            reason=payload.reason,
        )
    except AttendanceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except AttendanceValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return ClassSessionOverrideRead(
        id=record.id,
        student_id=record.student_id,
        course_id=record.course_id,
        session_date=record.session_date,
        status=record.status,
        previous_status=previous_status,
        evaluated_at=record.evaluated_at,
    )


__all__ = ["router"]
