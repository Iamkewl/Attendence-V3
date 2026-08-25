"""User domain service with RBAC-oriented query and mutation helpers."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.domain.models import User, UserRole
from app.domain.schemas import UserCreate, UserUpdate
from app.services.audit_service import AuditEvent, GovernanceAction
from app.services.base import AsyncCRUDService


class UserService(AsyncCRUDService[User, UserCreate, UserUpdate]):
    """Application service for transactional user CRUD operations."""

    def __init__(self, session: AsyncSession, *, actor: User | None = None) -> None:
        super().__init__(session=session, model=User)
        self._actor = actor

    def _actor_id(self) -> UUID | None:
        return self._actor.id if self._actor is not None else None

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
        # USER_CREATE is mandatory (D1). The factory runs after the INSERT so
        # it can capture the server-generated id; only non-secret scalars are
        # summarized (never the password or its hash).
        self.queue_audit_event(
            lambda created: AuditEvent(
                action=GovernanceAction.USER_CREATE,
                entity_type="user",
                entity_id=created.id,
                actor_user_id=self._actor_id(),
                change_summary={"target_role": payload.role.value},
            )
        )
        return await self.create(user_values)

    async def update_user(self, user_id: UUID, payload: UserUpdate) -> User | None:
        """Patch mutable user fields and return the updated entity when found."""
        values = payload.model_dump(exclude_unset=True)
        # Summarize field NAMES only; a credential change is recorded as a
        # boolean, never as password material (privacy rule Q9).
        fields_changed = sorted(set(values) - {"password"})
        if "password" in values:
            fields_changed.append("password_changed")
        self.queue_audit_event(
            lambda _result: AuditEvent(
                action=GovernanceAction.USER_UPDATE,
                entity_type="user",
                entity_id=user_id,
                actor_user_id=self._actor_id(),
                change_summary={"fields_changed": fields_changed},
            )
        )
        return await self.update(user_id, values)

    async def delete_user(self, user_id: UUID) -> bool:
        """Delete a user account, recording its pre-delete role."""
        user = await self.get(user_id)
        if user is None:
            return False

        self.queue_audit_event(
            lambda _result: AuditEvent(
                action=GovernanceAction.USER_DELETE,
                entity_type="user",
                entity_id=user.id,
                actor_user_id=self._actor_id(),
                change_summary={
                    "target_role": user.role.value,
                    "target_email_domain": user.email.rsplit("@", 1)[-1],
                },
            )
        )
        return await self.delete(user_id)


__all__ = ["UserService"]
