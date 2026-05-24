"""models subpackage — re-exports every public symbol for backward compatibility."""

from ._base import (  # noqa: F401
    ATTENDANCE_SOURCE_ENUM,
    ATTENDANCE_STATUS_ENUM,
    USER_ROLE_ENUM,
    AttendanceSource,
    AttendanceStatus,
    Base,
    MODEL_METADATA,
    TimestampMixin,
    UserRole,
    UUIDPrimaryKeyMixin,
    _enum_values,
)
from .user import User  # noqa: F401
from .student import Student, StudentEmbedding, TemplateAuditLog  # noqa: F401
from .course import Course  # noqa: F401
from .room import Room  # noqa: F401
from .sighting import Sighting  # noqa: F401
from .session import ClassSessionRecord  # noqa: F401
from .governance import GovernanceLog  # noqa: F401

__all__ = [
    "ATTENDANCE_SOURCE_ENUM",
    "ATTENDANCE_STATUS_ENUM",
    "USER_ROLE_ENUM",
    "AttendanceSource",
    "AttendanceStatus",
    "Base",
    "ClassSessionRecord",
    "Course",
    "GovernanceLog",
    "MODEL_METADATA",
    "Room",
    "Sighting",
    "Student",
    "StudentEmbedding",
    "TemplateAuditLog",
    "TimestampMixin",
    "User",
    "UserRole",
    "UUIDPrimaryKeyMixin",
    "_enum_values",
]
