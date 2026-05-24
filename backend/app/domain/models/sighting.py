"""Sighting ORM model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ._base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from .student import Student
    from .course import Course
    from .room import Room


class Sighting(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Stores one raw AI heartbeat recognition event from a classroom camera."""

    __tablename__ = "sightings"
    __table_args__ = (
        CheckConstraint(
            "char_length(btrim(camera_id)) > 0",
            name="sightings_camera_id_not_blank",
        ),
        CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)",
            name="sightings_confidence_range",
        ),
        Index("ix_sightings_course_timestamp", "course_id", "timestamp"),
        Index("ix_sightings_student_timestamp", "student_id", "timestamp"),
        Index("ix_sightings_camera_timestamp", "camera_id", "timestamp"),
    )

    student_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("students.id", ondelete="SET NULL"),
        nullable=True,
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="RESTRICT"),
        nullable=False,
    )
    room_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("rooms.id", ondelete="SET NULL"),
        nullable=True,
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence_score: Mapped[float | None] = mapped_column(nullable=True)
    embedding_reference: Mapped[str | None] = mapped_column(String(255), nullable=True)

    student: Mapped[Student | None] = relationship(back_populates="sightings", lazy="select")
    course: Mapped[Course] = relationship(back_populates="sightings", lazy="select")
    room: Mapped[Room | None] = relationship(back_populates="sightings", lazy="select")
