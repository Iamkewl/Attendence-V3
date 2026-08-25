"""CourseInstructor ORM model — instructor↔course ownership link (ATT-016)."""

from __future__ import annotations

import uuid

from sqlalchemy import CheckConstraint, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import text

from ._base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class CourseInstructor(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Associates a user with a course they are authorized to operate.

    Phase 1 (ATT-016 / decision D8): ``role_in_course='owner'`` grants
    course-scoped roster access; ``'ta'`` rows are stored but DENIED —
    fail-closed until a TA policy lands. Cross-listed courses require an
    explicit row per course code (D10: no guessed links).
    """

    __tablename__ = "course_instructors"
    __table_args__ = (
        UniqueConstraint("course_id", "user_id", name="uq_course_instructors_course_user"),
        CheckConstraint(
            "role_in_course IN ('owner', 'ta')",
            name="course_instructor_role_valid",
        ),
        Index("ix_course_instructors_user_id", "user_id"),
    )

    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    role_in_course: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=text("'owner'"),
    )


__all__ = ["CourseInstructor"]
