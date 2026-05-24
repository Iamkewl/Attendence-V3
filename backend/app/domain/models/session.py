"""ClassSessionRecord ORM model."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func, text

from ._base import ATTENDANCE_STATUS_ENUM, AttendanceStatus, Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from .student import Student
    from .course import Course
    from .governance import GovernanceLog


class ClassSessionRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Stores one aggregated attendance outcome per student, course, and class date."""

    __tablename__ = "class_session_records"
    __table_args__ = (
        UniqueConstraint(
            "student_id",
            "course_id",
            "session_date",
            name="uq_class_session_student_course_date",
        ),
        CheckConstraint("sighting_count >= 0", name="class_session_sighting_count_non_negative"),
        CheckConstraint(
            "required_sightings_threshold >= 1",
            name="class_session_threshold_positive",
        ),
        Index("ix_class_session_course_date", "course_id", "session_date"),
        Index("ix_class_session_student_date", "student_id", "session_date"),
        Index("ix_class_session_status_date", "status", "session_date"),
    )

    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False,
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="RESTRICT"),
        nullable=False,
    )
    session_date: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[AttendanceStatus] = mapped_column(
        ATTENDANCE_STATUS_ENUM,
        nullable=False,
        server_default=text("'absent'::attendance_status"),
    )
    sighting_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    required_sightings_threshold: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("1"),
    )
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    student: Mapped[Student] = relationship(back_populates="class_session_records", lazy="select")
    course: Mapped[Course] = relationship(back_populates="class_session_records", lazy="select")
    governance_logs: Mapped[list[GovernanceLog]] = relationship(
        back_populates="class_session_record",
        lazy="select",
    )
