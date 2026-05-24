"""Shared declarative base, metadata, enums, and SA enum type objects."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SAEnum, MetaData
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


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


def _enum_values(enum_cls: type[Enum]) -> list[str]:
    """Return the string values of an Enum class for SQLAlchemy values_callable."""
    return [member.value for member in enum_cls]


USER_ROLE_ENUM = SAEnum(
    UserRole,
    name="user_role",
    native_enum=True,
    validate_strings=True,
    values_callable=_enum_values,
)
ATTENDANCE_STATUS_ENUM = SAEnum(
    AttendanceStatus,
    name="attendance_status",
    native_enum=True,
    validate_strings=True,
    values_callable=_enum_values,
)
ATTENDANCE_SOURCE_ENUM = SAEnum(
    AttendanceSource,
    name="attendance_source",
    native_enum=True,
    validate_strings=True,
    values_callable=_enum_values,
)
