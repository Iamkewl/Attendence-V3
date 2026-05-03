"""User management endpoints protected by administrator role checks."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentAdminUser
from app.core.database import get_async_session
from app.domain.models import UserRole
from app.domain.schemas import UserCreate, UserRead, UserUpdate
from app.services.user_service import UserService


router = APIRouter(prefix="/users", tags=["Users"])


@router.post(
    "",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create User",
    description="Create a new platform user account.",
)
async def create_user(
    payload: UserCreate,
    _: CurrentAdminUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> UserRead:
    """Create one user record with hashed credentials."""
    service = UserService(session)

    try:
        return await service.create_user(payload)
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with the provided email already exists.",
        ) from exc


@router.get(
    "",
    response_model=list[UserRead],
    summary="List Users",
    description="List users with optional role and active-state filters.",
)
async def list_users(
    _: CurrentAdminUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    role: Annotated[UserRole | None, Query()] = None,
    is_active: Annotated[bool | None, Query()] = None,
) -> list[UserRead]:
    """Return a paginated collection of user records."""
    service = UserService(session)
    return await service.list_users(
        offset=offset,
        limit=limit,
        role=role,
        is_active=is_active,
    )


@router.get(
    "/{user_id}",
    response_model=UserRead,
    summary="Get User",
    description="Fetch one user by immutable identifier.",
)
async def get_user(
    user_id: UUID,
    _: CurrentAdminUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> UserRead:
    """Load one user record by UUID."""
    service = UserService(session)
    user = await service.get(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return user


@router.patch(
    "/{user_id}",
    response_model=UserRead,
    summary="Update User",
    description="Patch mutable user fields by identifier.",
)
async def update_user(
    user_id: UUID,
    payload: UserUpdate,
    _: CurrentAdminUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> UserRead:
    """Apply partial user updates for role and profile fields."""
    service = UserService(session)

    try:
        updated = await service.update_user(user_id, payload)
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The requested user update conflicts with an existing record.",
        ) from exc

    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    return updated


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete User",
    description="Delete one user account by identifier.",
)
async def delete_user(
    user_id: UUID,
    _: CurrentAdminUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> Response:
    """Delete a user and return an empty response when successful."""
    service = UserService(session)
    deleted = await service.delete(user_id)

    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
