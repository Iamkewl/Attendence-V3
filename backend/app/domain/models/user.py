"""User ORM model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, DateTime, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import text

from ._base import Base, TimestampMixin, USER_ROLE_ENUM, UserRole, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from .student import Student
    from .governance import GovernanceLog


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Represents an authenticated actor who can access attendance workflows."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
        CheckConstraint("char_length(btrim(full_name)) >= 2", name="users_full_name_min_length"),
        CheckConstraint(
            "char_length(password_hash) >= 60",
            name="users_password_hash_min_length",
        ),
        Index("ix_users_role_active", "role", "is_active"),
    )

    email: Mapped[str] = mapped_column(String(320), nullable=False)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        USER_ROLE_ENUM,
        nullable=False,
        server_default=text("'instructor'::user_role"),
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    student_profile: Mapped[Student | None] = relationship(
        back_populates="user",
        lazy="select",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    governance_logs: Mapped[list[GovernanceLog]] = relationship(
        back_populates="actor_user",
        lazy="select",
        foreign_keys="GovernanceLog.actor_user_id",
    )
