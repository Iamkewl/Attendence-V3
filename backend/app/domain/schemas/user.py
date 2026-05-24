"""User account schemas for registration, update, and read operations."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import EmailStr, StringConstraints, field_validator

from app.domain.models import UserRole

from .common import NameStr, SchemaModel


class UserCreate(SchemaModel):
    """Input schema used to register a new platform user."""

    email: EmailStr
    full_name: NameStr
    password: Annotated[str, StringConstraints(min_length=8, max_length=128)]
    role: UserRole = UserRole.INSTRUCTOR
    is_active: bool = True

    @field_validator("full_name")
    @classmethod
    def normalize_full_name(cls, value: str) -> str:
        """Collapse repeated internal spaces to maintain canonical names."""
        return " ".join(value.split())


class UserUpdate(SchemaModel):
    """Input schema used to patch mutable user fields."""

    full_name: NameStr | None = None
    role: UserRole | None = None
    is_active: bool | None = None
    last_login_at: datetime | None = None

    @field_validator("full_name")
    @classmethod
    def normalize_full_name(cls, value: str | None) -> str | None:
        """Collapse repeated internal spaces when a full name is provided."""
        if value is None:
            return None
        return " ".join(value.split())


class UserRead(SchemaModel):
    """Output schema representing a user record returned from persistence."""

    id: UUID
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool
    last_login_at: datetime | None
    created_at: datetime
    updated_at: datetime
