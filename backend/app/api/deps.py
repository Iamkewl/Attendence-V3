"""Reusable FastAPI dependencies for authentication and authorization."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.security import SecurityError, get_security_settings, validate_token
from app.domain.models import CourseInstructor, User, UserRole


_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    bearer: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> User:
    """Resolve the currently authenticated and active user from bearer token or cookie."""
    settings = get_security_settings()
    token = bearer.credentials if bearer is not None else request.cookies.get(settings.access_cookie_name)

    if token is None or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
        )

    try:
        claims = await validate_token(token.strip(), expected_token_type="access")
    except SecurityError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
        ) from exc

    user = (
        await session.execute(
            select(User).where(User.id == claims.sub),
        )
    ).scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user does not exist.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authenticated user is inactive.",
        )

    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def _ensure_user_role(user: User, *, allowed_roles: set[UserRole], detail: str) -> User:
    """Enforce role membership for protected dependencies."""
    if user.role not in allowed_roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
    return user


async def get_current_admin_user(current_user: CurrentUser) -> User:
    """Require an authenticated user with administrator privileges."""
    return _ensure_user_role(
        current_user,
        allowed_roles={UserRole.ADMIN},
        detail="Administrator privileges are required for this operation.",
    )


async def get_current_instructor_user(current_user: CurrentUser) -> User:
    """Require an authenticated instructor or administrator user."""
    return _ensure_user_role(
        current_user,
        allowed_roles={UserRole.ADMIN, UserRole.INSTRUCTOR},
        detail="Instructor or administrator privileges are required for this operation.",
    )


async def get_current_ingest_user(current_user: CurrentUser) -> User:
    """Require an authenticated instructor or administrator to submit frames (ATT-075).

    The ingest endpoints (``/inference/stream``, ``/inference/batch``,
    ``/inference/photo``) are the intake of the biometric pipeline: they
    accept raw camera frames and produce embeddings/matches. Only roles that
    own the enrollment and attendance workflows may push frames — AUDITOR is
    deliberately read-only ("sees-all", decision D5) and OPERATOR is a
    worker-system principal, so neither has any legitimate ingest path.
    Mirrors :func:`get_current_governance_reader`: same fail-closed 403 shape,
    narrower role set.
    """
    return _ensure_user_role(
        current_user,
        allowed_roles={UserRole.ADMIN, UserRole.INSTRUCTOR},
        detail="Instructor or administrator privileges are required to submit frames for inference.",
    )


async def get_current_worker_system(current_user: CurrentUser) -> User:
    """Require an authenticated worker-system principal or administrator."""
    return _ensure_user_role(
        current_user,
        allowed_roles={UserRole.ADMIN, UserRole.OPERATOR},
        detail="Worker system or administrator privileges are required for this operation.",
    )


async def get_current_governance_reader(current_user: CurrentUser) -> User:
    """Require an authenticated auditor or administrator (decision D5).

    AUDITOR scope is deliberately "sees-all, read-only": auditors perform none
    of the logged actions, so a narrower scope would hand them an empty ledger.
    Separation of duties is preserved because AUDITOR holds no write power
    anywhere else in the API (those fail-closed denials are unchanged).
    """
    return _ensure_user_role(
        current_user,
        allowed_roles={UserRole.ADMIN, UserRole.AUDITOR},
        detail="Auditor or administrator privileges are required for this operation.",
    )


CurrentAdminUser = Annotated[User, Depends(get_current_admin_user)]
CurrentGovernanceReader = Annotated[User, Depends(get_current_governance_reader)]
CurrentIngestUser = Annotated[User, Depends(get_current_ingest_user)]
CurrentInstructorUser = Annotated[User, Depends(get_current_instructor_user)]
CurrentWorkerSystem = Annotated[User, Depends(get_current_worker_system)]


_COURSE_ROLE_OWNER = "owner"


async def ensure_course_scoped_principal(
    session: AsyncSession, user: User, *, course_id: UUID
) -> User:
    """Authorize ``user`` against a specific course (non-DI variant).

    ATT-038: extracted from :func:`get_course_scoped_principal` so routes
    whose path binds the course under a different name (e.g. the manual
    override route's ``session_id``) reuse ONE implementation instead of
    drifting. Same contract as the DI dependency: flag off = pass-through;
    ADMIN bypasses; INSTRUCTORs need a live owner row; everyone else and
    every missing link fails closed with an existence-denying 404 ('ta'
    rows stored but denied in phase 1, D10).
    """
    settings = get_security_settings()
    if not settings.course_scoped_authz_enabled:
        return user

    if user.role == UserRole.ADMIN:
        return user

    # Non-instructors never hold valid links; deny without touching the DB.
    # INSTRUCTORs fall through to the link check below.
    if user.role != UserRole.INSTRUCTOR:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course does not exist.",
        )

    linked = await session.scalar(
        select(CourseInstructor.id)
        .where(CourseInstructor.user_id == user.id)
        .where(CourseInstructor.course_id == course_id)
        .where(CourseInstructor.role_in_course == _COURSE_ROLE_OWNER)
    )
    if linked is None:
        # Fail closed: 'ta'-only or unlinked instructors get the identical
        # existence-denying 404 whether or not the course exists.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course does not exist.",
        )
    return user


async def get_course_scoped_principal(
    course_id: UUID,
    current_user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> User:
    """Authorize a principal against the specific course they are querying.

    ATT-016 phase 1 (decisions D8-D10). Gated by
    ``ATTENDANCE_COURSE_SCOPED_AUTHZ`` (default false): when off this is a
    pure pass-through so behavior stays byte-identical to the legacy
    ``CurrentUser``-only route.

    When on, ADMIN bypasses the link check; every other role must hold a
    live ``course_instructors`` row with ``role_in_course='owner'`` for the
    requested course. The link is re-read from the DB on every request (no
    memoization — caching would defeat mid-session revocation).

    The link lookup runs BEFORE any course-existence query and denials use
    404 with the same detail string the service uses for genuinely missing
    courses, so unauthorized callers learn nothing about which course ids
    exist (no course-ID oracle). Unknown state fails closed.
    """
    return await ensure_course_scoped_principal(
        session, current_user, course_id=course_id
    )


CourseScopedPrincipal = Annotated[User, Depends(get_course_scoped_principal)]


__all__ = [
    "CourseScopedPrincipal",
    "CurrentAdminUser",
    "CurrentGovernanceReader",
    "CurrentIngestUser",
    "CurrentInstructorUser",
    "CurrentUser",
    "CurrentWorkerSystem",
    "ensure_course_scoped_principal",
    "get_course_scoped_principal",
    "get_current_admin_user",
    "get_current_governance_reader",
    "get_current_ingest_user",
    "get_current_instructor_user",
    "get_current_user",
    "get_current_worker_system",
]
