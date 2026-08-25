"""Admin-only aggregate endpoints (ATT-045/coverage dashboard support).

``GET /api/v1/admin/enrollment-coverage``
    Per-student enrollment inventory: active template count, pose labels,
    last-enrolled timestamp, biometric consent status, and trailing-7-day
    sighting count. Built as ONE grouped query (embeddings via a LEFT JOIN
    aggregate, sightings via one correlated scalar subquery) so the payload
    is O(1) round-trips, never N+1. Read-only; ADMIN only, fail closed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentAdminUser
from app.core.database import get_async_session
from app.domain.models import Sighting, Student, StudentEmbedding, User
from app.domain.schemas import EnrollmentCoverageRow


router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get(
    "/enrollment-coverage",
    response_model=list[EnrollmentCoverageRow],
    summary="Enrollment Coverage",
    description=(
        "Per-student aggregate of active face templates, poses, last "
        "enrollment time, biometric consent state, and 7-day sighting "
        "volume. Single grouped query; powers the coverage dashboard."
    ),
)
async def get_enrollment_coverage(
    _: CurrentAdminUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[EnrollmentCoverageRow]:
    """Return the per-student enrollment-coverage aggregate."""
    cutoff = datetime.now(tz=UTC) - timedelta(days=7)

    active = StudentEmbedding.is_active.is_(True)
    # Correlated scalar subquery: evaluated once per output row INSIDE the
    # same statement — still a single round-trip, not application-side N+1.
    sightings_last_7d = (
        select(func.count(Sighting.id))
        .where(Sighting.student_id == Student.id)
        .where(Sighting.timestamp >= cutoff)
        .correlate(Student)
        .scalar_subquery()
    )

    stmt = (
        select(
            Student.id,
            Student.student_number,
            Student.biometric_consent_status,
            func.coalesce(User.full_name, "").label("full_name"),
            func.count(StudentEmbedding.id).filter(active).label("active_template_count"),
            func.array_agg(StudentEmbedding.pose_label)
            .filter(active)
            .label("poses"),
            func.max(StudentEmbedding.created_at)
            .filter(active)
            .label("last_enrolled_at"),
            sightings_last_7d.label("sightings_last_7d"),
        )
        .join(User, User.id == Student.user_id)
        .join(StudentEmbedding, StudentEmbedding.student_id == Student.id, isouter=True)
        .group_by(
            Student.id,
            Student.student_number,
            Student.biometric_consent_status,
            User.full_name,
        )
        .order_by(Student.student_number)
        .offset(offset)
        .limit(limit)
    )

    rows = (await session.execute(stmt)).all()
    return [
        EnrollmentCoverageRow(
            student_id=row.id,
            student_number=row.student_number,
            full_name=row.full_name or "",
            active_template_count=row.active_template_count,
            # A student with zero embeddings aggregates to NULL (no filtered
            # rows); normalize to an empty pose list for the dashboard.
            poses=[pose for pose in (row.poses or []) if pose],
            last_enrolled_at=row.last_enrolled_at,
            biometric_consent_status=row.biometric_consent_status,
            sightings_last_7d=row.sightings_last_7d,
        )
        for row in rows
    ]


__all__ = ["router"]
