"""Attendance session endpoints for class roster queries.

Route authorization surface (ATT-016 phase 1):

``GET /api/v1/attendance/sessions``
    Guarded by ``CourseScopedPrincipal``. With ``ATTENDANCE_COURSE_SCOPED_AUTHZ``
    unset/false the dependency is a pass-through and this route behaves exactly
    as before ATT-016 (any authenticated user may query any course). When the
    flag is on, ADMIN passes unconditionally while every other role must hold
    an active ``course_instructors`` owner row for the requested course;
    callers without one receive 404 with an existence-denying detail — the link
    check runs before any course lookup so the route cannot be used to probe
    which course ids exist (see ``app.api.deps.get_course_scoped_principal``).

Error mapping on this route: ``AttendanceNotFoundError`` -> 404 and, fixing a
latent bug where an inactive course escaped as HTTP 500,
``AttendanceValidationError`` (e.g. "Course is inactive.") -> 409 in both flag
modes. Other routes (students, inference) are unchanged in phase 1 per the
design's affected-routes table.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CourseScopedPrincipal
from app.core.database import get_async_session
from app.domain.schemas import ClassSessionListResponse
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


__all__ = ["router"]
