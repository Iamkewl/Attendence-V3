"""Course ORM model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, Index, SmallInteger, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import text

from ._base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from .session import ClassSessionRecord
    from .sighting import Sighting


class Course(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Represents a teachable course against which attendance is recorded."""

    __tablename__ = "courses"
    __table_args__ = (
        UniqueConstraint("code", name="uq_courses_code"),
        CheckConstraint("credits BETWEEN 1 AND 12", name="courses_credits_range"),
        Index("ix_courses_active", "is_active"),
    )

    code: Mapped[str] = mapped_column(String(24), nullable=False)
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    credits: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        server_default=text("3"),
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )

    class_session_records: Mapped[list[ClassSessionRecord]] = relationship(
        back_populates="course",
        lazy="select",
    )
    sightings: Mapped[list[Sighting]] = relationship(
        back_populates="course",
        lazy="select",
    )
