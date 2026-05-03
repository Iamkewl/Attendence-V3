"""SQLAlchemy ORM models for the V2 attendance domain."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover - compatibility fallback for older pgvector builds.
    from pgvector.sqlalchemy import VECTOR as Vector


MODEL_NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

MODEL_METADATA = MetaData(naming_convention=MODEL_NAMING_CONVENTION)


class Base(DeclarativeBase):
    """Base declarative class shared by all attendance domain models."""

    metadata = MODEL_METADATA


class UUIDPrimaryKeyMixin:
    """Mixin that provides a UUID primary key column for each table."""

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    """Mixin that provides creation and update timestamps for auditability."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class UserRole(str, Enum):
    """Enumerates all supported authorization roles for platform users."""

    ADMIN = "admin"
    INSTRUCTOR = "instructor"
    AUDITOR = "auditor"
    OPERATOR = "operator"


class AttendanceStatus(str, Enum):
    """Enumerates the canonical attendance states captured per student session."""

    PRESENT = "present"
    ABSENT = "absent"
    LATE = "late"
    EXCUSED = "excused"


class AttendanceSource(str, Enum):
    """Enumerates all accepted data sources for attendance check-in events."""

    MANUAL = "manual"
    QR_CODE = "qr_code"
    BIOMETRIC = "biometric"
    IMPORTED = "imported"


USER_ROLE_ENUM = SAEnum(UserRole, name="user_role", native_enum=True, validate_strings=True)
ATTENDANCE_STATUS_ENUM = SAEnum(
    AttendanceStatus,
    name="attendance_status",
    native_enum=True,
    validate_strings=True,
)
ATTENDANCE_SOURCE_ENUM = SAEnum(
    AttendanceSource,
    name="attendance_source",
    native_enum=True,
    validate_strings=True,
)


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


class Room(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Represents a physical learning space where attendance sessions can occur."""

    __tablename__ = "rooms"
    __table_args__ = (
        UniqueConstraint("code", name="uq_rooms_code"),
        CheckConstraint("capacity BETWEEN 1 AND 2000", name="rooms_capacity_range"),
        Index("ix_rooms_building_active", "building", "is_active"),
    )

    code: Mapped[str] = mapped_column(String(24), nullable=False)
    building: Mapped[str] = mapped_column(String(120), nullable=False)
    floor: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )

    sightings: Mapped[list[Sighting]] = relationship(
        back_populates="room",
        lazy="select",
    )


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


class GovernanceLog(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Captures auditable governance actions across attendance domain entities."""

    __tablename__ = "governance_logs"
    __table_args__ = (
        CheckConstraint("char_length(btrim(action)) > 0", name="governance_action_not_blank"),
        CheckConstraint(
            "char_length(btrim(entity_type)) > 0",
            name="governance_entity_type_not_blank",
        ),
        Index("ix_governance_actor_created_at", "actor_user_id", "created_at"),
        Index("ix_governance_entity_lookup", "entity_type", "entity_id"),
        Index("ix_governance_class_session_record_id", "class_session_record_id"),
        Index("ix_governance_request_id", "request_id"),
        Index("ix_governance_created_at", "created_at"),
    )

    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    class_session_record_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("class_session_records.id", ondelete="SET NULL"),
        nullable=True,
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    change_summary: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    request_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(INET, nullable=True)

    actor_user: Mapped[User | None] = relationship(
        back_populates="governance_logs",
        lazy="select",
        foreign_keys=[actor_user_id],
    )
    class_session_record: Mapped[ClassSessionRecord | None] = relationship(
        back_populates="governance_logs",
        lazy="select",
        foreign_keys=[class_session_record_id],
    )


__all__ = [
    "ClassSessionRecord",
    "AttendanceSource",
    "AttendanceStatus",
    "Base",
    "Course",
    "GovernanceLog",
    "Room",
    "Sighting",
    "Student",
    "StudentEmbedding",
    "TemplateAuditLog",
    "User",
    "UserRole",
]
