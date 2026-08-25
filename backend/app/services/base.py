"""Generic asynchronous CRUD repository primitives for SQLAlchemy ORM models."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.services.audit_service import AuditEvent, AuditService


ModelT = TypeVar("ModelT")
CreateSchemaT = TypeVar("CreateSchemaT")
UpdateSchemaT = TypeVar("UpdateSchemaT")

# Zero-arg-result factories let subclasses defer building an AuditEvent until
# the CRUD statement has executed inside the transaction; the factory receives
# the operation result (inserted/updated row, deleted id, or None) so it can
# capture server-generated primary keys.
AuditEventFactory = Callable[[Any], AuditEvent]


class AsyncCRUDService(Generic[ModelT, CreateSchemaT, UpdateSchemaT]):
    """Typed CRUD building block for request-scoped asynchronous service classes."""

    def __init__(self, session: AsyncSession, model: type[ModelT]) -> None:
        self.session = session
        self.model = model
        self._pending_audit: list[AuditEvent | AuditEventFactory] = []

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Wrap a write workflow in commit-or-rollback transaction handling."""
        try:
            yield
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            self._pending_audit.clear()
            raise

    async def get(self, entity_id: UUID) -> ModelT | None:
        """Load one entity by primary key identifier."""
        stmt = select(self.model).where(getattr(self.model, "id") == entity_id)
        return (await self._execute_read(stmt)).scalar_one_or_none()

    async def list(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        filters: Sequence[ColumnElement[bool]] | None = None,
        order_by: Any | None = None,
    ) -> list[ModelT]:
        """Load a paginated entity collection with optional filtering and ordering."""
        if offset < 0:
            raise ValueError("offset must be greater than or equal to zero")
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        stmt = select(self.model)
        if filters is not None:
            for filter_clause in filters:
                stmt = stmt.where(filter_clause)

        if order_by is not None:
            stmt = stmt.order_by(order_by)

        stmt = stmt.offset(offset).limit(limit)
        return list((await self._execute_read(stmt)).scalars().all())

    async def create(
        self,
        payload: CreateSchemaT | BaseModel | Mapping[str, Any],
    ) -> ModelT:
        """Persist a new entity instance and return the inserted row."""
        values = self._normalize_payload(payload, partial=False)
        stmt = insert(self.model).values(**values).returning(self.model)

        async with self.transaction():
            created = (await self.session.execute(stmt)).scalar_one()
            await self._drain_pending_audit(created)

        return created

    async def update(
        self,
        entity_id: UUID,
        payload: UpdateSchemaT | BaseModel | Mapping[str, Any],
    ) -> ModelT | None:
        """Patch an existing entity and return the updated row when found."""
        values = self._normalize_payload(payload, partial=True)
        if not values:
            self._pending_audit.clear()
            return await self.get(entity_id)

        stmt = (
            update(self.model)
            .where(getattr(self.model, "id") == entity_id)
            .values(**values)
            .returning(self.model)
        )

        async with self.transaction():
            updated = (await self.session.execute(stmt)).scalar_one_or_none()
            if updated is not None:
                await self._drain_pending_audit(updated)

        return updated

    async def delete(self, entity_id: UUID) -> bool:
        """Delete one entity by identifier and report whether a row was removed."""
        stmt = delete(self.model).where(getattr(self.model, "id") == entity_id).returning(
            getattr(self.model, "id")
        )

        async with self.transaction():
            deleted_id = (await self.session.execute(stmt)).scalar_one_or_none()
            if deleted_id is not None:
                await self._drain_pending_audit(deleted_id)

        return deleted_id is not None

    def queue_audit_event(self, event: AuditEvent | AuditEventFactory) -> None:
        """Queue a governance event emitted inside the next successful write transaction."""
        self._pending_audit.append(event)

    async def _drain_pending_audit(self, result: Any = None) -> None:
        """Emit queued governance rows via flush() inside the active transaction."""
        while self._pending_audit:
            entry = self._pending_audit.pop(0)
            event = entry(result) if callable(entry) else entry
            await AuditService(self.session).emit(event)

    async def _execute_read(self, statement: Any):
        """Execute a read query and rollback the session if execution fails."""
        try:
            return await self.session.execute(statement)
        except Exception:
            await self.session.rollback()
            raise

    def _normalize_payload(
        self,
        payload: BaseModel | Mapping[str, Any] | CreateSchemaT | UpdateSchemaT,
        *,
        partial: bool,
    ) -> dict[str, Any]:
        """Normalize Pydantic or mapping payloads into mutable dictionaries."""
        if isinstance(payload, BaseModel):
            return payload.model_dump(exclude_unset=partial)

        if isinstance(payload, Mapping):
            return dict(payload)

        raise TypeError("payload must be a Pydantic model or mapping")


__all__ = ["AsyncCRUDService"]
