"""Reusable FastAPI dependencies for authentication and authorization."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.security import SecurityError, get_security_settings, validate_token
from app.domain.models import User, UserRole


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


async def get_current_worker_system(current_user: CurrentUser) -> User:
    """Require an authenticated worker-system principal or administrator."""
    return _ensure_user_role(
        current_user,
        allowed_roles={UserRole.ADMIN, UserRole.OPERATOR},
        detail="Worker system or administrator privileges are required for this operation.",
    )


CurrentAdminUser = Annotated[User, Depends(get_current_admin_user)]
CurrentInstructorUser = Annotated[User, Depends(get_current_instructor_user)]
CurrentWorkerSystem = Annotated[User, Depends(get_current_worker_system)]


__all__ = [
    "CurrentAdminUser",
    "CurrentInstructorUser",
    "CurrentUser",
    "CurrentWorkerSystem",
    "get_current_admin_user",
    "get_current_instructor_user",
    "get_current_user",
    "get_current_worker_system",
]
