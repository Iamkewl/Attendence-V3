"""User domain service with RBAC-oriented query and mutation helpers."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.domain.models import User, UserRole
from app.domain.schemas import UserCreate, UserUpdate
from app.services.base import AsyncCRUDService


class UserService(AsyncCRUDService[User, UserCreate, UserUpdate]):
    """Application service for transactional user CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session=session, model=User)

    async def get_by_email(self, email: str) -> User | None:
        """Load one user by normalized e-mail address."""
        normalized_email = email.strip().lower()
        stmt = select(User).where(func.lower(User.email) == normalized_email)
        return (await self._execute_read(stmt)).scalar_one_or_none()

    async def list_users(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        role: UserRole | None = None,
        is_active: bool | None = None,
    ) -> list[User]:
        """Return paginated users with optional role and activity filtering."""
        filters = []
        if role is not None:
            filters.append(User.role == role)
        if is_active is not None:
            filters.append(User.is_active.is_(is_active))

        return await self.list(
            offset=offset,
            limit=limit,
            filters=filters or None,
            order_by=User.created_at.desc(),
        )

    async def create_user(self, payload: UserCreate) -> User:
        """Create a user while hashing credentials and normalizing e-mail casing."""
        user_values = payload.model_dump(exclude={"password"})
        user_values["email"] = payload.email.lower()
        user_values["password_hash"] = hash_password(payload.password)
        return await self.create(user_values)

    async def update_user(self, user_id: UUID, payload: UserUpdate) -> User | None:
        """Patch mutable user fields and return the updated entity when found."""
        return await self.update(user_id, payload.model_dump(exclude_unset=True))


__all__ = ["UserService"]
