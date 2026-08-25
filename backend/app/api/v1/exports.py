"""ATT-039 CSV attendance roster export.

Route authorization surface (mirrors the attendance session routes):

``GET /api/v1/courses/{course_id}/attendance/export``
    A sensitive data-egress READ, so it layers the SAME two gates the
    session routes already use — one implementation, no drift:

    * ``CurrentInstructorUser`` (unconditional role gate, exactly like the
      ATT-038 override write path): only ADMIN or INSTRUCTOR may export;
      AUDITOR/OPERATOR receive 403 in BOTH flag modes. Exports are strictly
      more exposing than roster views, so they never inherit the looser
      legacy read surface.
    * ``ensure_course_scoped_principal`` — the identical check
      ``GET /attendance/sessions`` uses, honoring
      ``ATTENDANCE_COURSE_SCOPED_AUTHZ`` exactly like reads today: flag off
      = pass-through; ADMIN bypasses; INSTRUCTORs need a live owner link;
      every other denial fails closed with the existence-denying 404.

Response is ``text/csv`` with an RFC 6266 attachment disposition. Rows come
from ``AttendanceService.export_daily_roster``, which reuses the daily
evaluation's population/verdict logic; embeddings are never selected,
logged, or serialized anywhere on this path.
"""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from io import StringIO
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentInstructorUser, ensure_course_scoped_principal
from app.core.database import get_async_session
from app.services.attendance_service import (
    AttendanceExportRow,
    AttendanceNotFoundError,
    AttendanceService,
    AttendanceValidationError,
)


router = APIRouter(tags=["Exports"])

_CSV_HEADERS = [
    "student_number",
    "student_name",
    "status",
    "confidence_score",
    "last_sighting_at",
    "override_applied",
    "override_reason",
]


def _render_csv(
    rows: list[AttendanceExportRow], course_id: UUID, session_date: date
) -> tuple[bytes, str]:
    """Serialize roster rows and build the attachment filename."""
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CSV_HEADERS)
    for row in rows:
        writer.writerow(
            [
                row.student_number,
                row.student_name,
                row.status,
                row.confidence_score,
                row.last_sighting_at,
                "true" if row.override_applied else "false",
                row.override_reason,
            ]
        )
    filename = f"attendance_{course_id}_{session_date.isoformat()}.csv"
    return buffer.getvalue().encode("utf-8"), filename


@router.get(
    "/courses/{course_id}/attendance/export",
    summary="Export Course Attendance CSV",
    description=(
        "Stream one CSV row per enrolled/seen student for the requested "
        "course and date (defaults to today). Advisory EXPORT governance "
        "event per download."
    ),
    response_class=Response,
)
async def export_course_attendance_csv(
    course_id: UUID,
    current_user: CurrentInstructorUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    export_date: Annotated[date | None, Query(alias="date")] = None,
    format: Literal["csv"] = "csv",
) -> Response:
    """Return the day's attendance roster as a downloadable CSV attachment."""
    # Role gate first (403), then the flag-gated course-link gate
    # (existence-denying 404 when ATTENDANCE_COURSE_SCOPED_AUTHZ is on) —
    # the same ordering and implementations the session routes use.
    await ensure_course_scoped_principal(session, current_user, course_id=course_id)

    service = AttendanceService(session=session, actor=current_user)
    session_date = export_date or datetime.now(tz=UTC).date()
    try:
        rows = await service.export_daily_roster(
            course_id=course_id, session_date=session_date
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

    payload, filename = _render_csv(rows, course_id, session_date)
    return Response(
        content=payload,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


__all__ = ["router"]
