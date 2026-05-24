"""Student, StudentEmbedding, and TemplateAuditLog ORM models (tightly coupled)."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func, text

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover - compatibility fallback for older pgvector builds.
    from pgvector.sqlalchemy import VECTOR as Vector

from ._base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Student(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Represents a student entity linked one-to-one with a platform user identity."""

    __tablename__ = "students"
    __table_args__ = (
        UniqueConstraint("user_id", name="uq_students_user_id"),
        UniqueConstraint("student_number", name="uq_students_student_number"),
        CheckConstraint("enrollment_year BETWEEN 2000 AND 2100", name="students_enrollment_year_range"),
        CheckConstraint(
            "graduation_year IS NULL OR graduation_year >= enrollment_year",
            name="students_graduation_year_valid",
        ),
        Index("ix_students_enrollment_active", "enrollment_year", "is_active"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    student_number: Mapped[str] = mapped_column(String(32), nullable=False)
    program: Mapped[str] = mapped_column(String(120), nullable=False)
    enrollment_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    graduation_year: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )

    user: Mapped[User] = relationship(back_populates="student_profile", lazy="select")
    class_session_records: Mapped[list[ClassSessionRecord]] = relationship(
        back_populates="student",
        lazy="select",
    )
    embeddings: Mapped[list[StudentEmbedding]] = relationship(
        back_populates="student",
        lazy="select",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    sightings: Mapped[list[Sighting]] = relationship(
        back_populates="student",
        lazy="select",
    )
    template_audit_logs: Mapped[list[TemplateAuditLog]] = relationship(
        back_populates="student",
        lazy="select",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class StudentEmbedding(UUIDPrimaryKeyMixin, Base):
    """Stores active and historical face template embeddings for a student across poses."""

    __tablename__ = "student_embeddings"
    __table_args__ = (
        CheckConstraint(
            "char_length(btrim(pose_label)) > 0",
            name="student_embeddings_pose_label_not_blank",
        ),
        CheckConstraint(
            "quality_score >= 0 AND quality_score <= 1",
            name="student_embeddings_quality_score_range",
        ),
        Index("ix_student_embeddings_student_pose_active", "student_id", "pose_label", "is_active"),
    )

    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False,
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(512), nullable=False)
    pose_label: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=text("'front'"),
    )
    quality_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        server_default=text("1.0"),
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    student: Mapped[Student] = relationship(back_populates="embeddings", lazy="select")
    audit_logs: Mapped[list[TemplateAuditLog]] = relationship(
        back_populates="student_embedding",
        lazy="select",
    )


class TemplateAuditLog(UUIDPrimaryKeyMixin, Base):
    """Tracks enrollment template lifecycle events such as create, archive, and updates."""

    __tablename__ = "template_audit_logs"
    __table_args__ = (
        CheckConstraint(
            "char_length(btrim(action)) > 0",
            name="template_audit_logs_action_not_blank",
        ),
        Index("ix_template_audit_logs_student_created_at", "student_id", "created_at"),
        Index("ix_template_audit_logs_embedding_created_at", "student_embedding_id", "created_at"),
    )

    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False,
    )
    student_embedding_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("student_embeddings.id", ondelete="SET NULL"),
        nullable=True,
    )
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    pose_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    event_metadata: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    student: Mapped[Student] = relationship(back_populates="template_audit_logs", lazy="select")
    student_embedding: Mapped[StudentEmbedding | None] = relationship(
        back_populates="audit_logs",
        lazy="select",
        foreign_keys=[student_embedding_id],
    )


# TYPE_CHECKING-style forward references resolved by SQLAlchemy string lookups above.
# These TYPE_CHECKING imports are only needed so that Mapped[User] etc. resolve at
# annotation evaluation time (from __future__ import annotations defers them).
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from .user import User
    from .session import ClassSessionRecord
    from .sighting import Sighting
