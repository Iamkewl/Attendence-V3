"""Attendance session endpoints for class roster queries."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser
from app.core.database import get_async_session
from app.domain.schemas import ClassSessionListResponse
from app.services.attendance_service import AttendanceNotFoundError, AttendanceService


router = APIRouter(prefix="/attendance", tags=["Attendance"])


@router.get(
    "/sessions",
    response_model=ClassSessionListResponse,
    summary="List Session Records",
    description="Return the attendance roster for a course on a given date.",
)
async def list_session_records(
    current_user: CurrentUser,
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


__all__ = ["router"]
