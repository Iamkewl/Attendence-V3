"""Pydantic V2 schemas for validating attendance domain input and output payloads."""

from __future__ import annotations

import base64
import binascii
from datetime import date, datetime
from typing import Any, Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from app.domain.models import AttendanceStatus, UserRole


NameStr = Annotated[str, StringConstraints(min_length=2, max_length=120)]
ProgramStr = Annotated[str, StringConstraints(min_length=2, max_length=120)]
CourseCodeStr = Annotated[
    str,
    StringConstraints(min_length=3, max_length=24, pattern=r"^[A-Z0-9][A-Z0-9_-]*$"),
]
RoomCodeStr = Annotated[
    str,
    StringConstraints(min_length=2, max_length=24, pattern=r"^[A-Z0-9][A-Z0-9_-]*$"),
]
StudentNumberStr = Annotated[
    str,
    StringConstraints(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$"),
]


class SchemaModel(BaseModel):
    """Base schema that enforces strict parsing and ORM interoperability."""

    model_config = ConfigDict(from_attributes=True, extra="forbid", str_strip_whitespace=True)


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


class StudentCreate(SchemaModel):
    """Input schema used to create a student profile."""

    user_id: UUID
    student_number: StudentNumberStr
    program: ProgramStr
    enrollment_year: Annotated[int, Field(ge=2000, le=2100)]
    graduation_year: Annotated[int | None, Field(ge=2000, le=2100)] = None
    date_of_birth: date | None = None
    is_active: bool = True

    @model_validator(mode="after")
    def validate_graduation_year(self) -> StudentCreate:
        """Ensure graduation year is not earlier than enrollment year."""
        if self.graduation_year is not None and self.graduation_year < self.enrollment_year:
            raise ValueError("graduation_year must be greater than or equal to enrollment_year")
        return self


class StudentUpdate(SchemaModel):
    """Input schema used to patch mutable student profile fields."""

    student_number: StudentNumberStr | None = None
    program: ProgramStr | None = None
    enrollment_year: Annotated[int | None, Field(ge=2000, le=2100)] = None
    graduation_year: Annotated[int | None, Field(ge=2000, le=2100)] = None
    date_of_birth: date | None = None
    is_active: bool | None = None

    @model_validator(mode="after")
    def validate_graduation_year(self) -> StudentUpdate:
        """Validate year ordering whenever both enrollment and graduation are provided."""
        if (
            self.enrollment_year is not None
            and self.graduation_year is not None
            and self.graduation_year < self.enrollment_year
        ):
            raise ValueError("graduation_year must be greater than or equal to enrollment_year")
        return self


class StudentRead(SchemaModel):
    """Output schema representing a student profile from persistence."""

    id: UUID
    user_id: UUID
    student_number: str
    program: str
    enrollment_year: int
    graduation_year: int | None
    date_of_birth: date | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class StudentEnrollmentRead(SchemaModel):
    """Output schema representing one persisted enrollment embedding template."""

    id: UUID
    student_id: UUID
    pose_label: str
    quality_score: float
    is_active: bool
    created_at: datetime


class CourseCreate(SchemaModel):
    """Input schema used to create a course catalog entry."""

    code: CourseCodeStr
    title: Annotated[str, StringConstraints(min_length=2, max_length=160)]
    description: Annotated[str | None, StringConstraints(max_length=4000)] = None
    credits: Annotated[int, Field(ge=1, le=12)] = 3
    is_active: bool = True


class CourseUpdate(SchemaModel):
    """Input schema used to patch mutable course fields."""

    code: CourseCodeStr | None = None
    title: Annotated[str | None, StringConstraints(min_length=2, max_length=160)] = None
    description: Annotated[str | None, StringConstraints(max_length=4000)] = None
    credits: Annotated[int | None, Field(ge=1, le=12)] = None
    is_active: bool | None = None


class CourseRead(SchemaModel):
    """Output schema representing a course catalog entity."""

    id: UUID
    code: str
    title: str
    description: str | None
    credits: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


class RoomCreate(SchemaModel):
    """Input schema used to create a physical room record."""

    code: RoomCodeStr
    building: Annotated[str, StringConstraints(min_length=2, max_length=120)]
    floor: Annotated[int | None, Field(ge=-10, le=200)] = None
    capacity: Annotated[int, Field(ge=1, le=2000)]
    is_active: bool = True


class RoomUpdate(SchemaModel):
    """Input schema used to patch mutable room fields."""

    code: RoomCodeStr | None = None
    building: Annotated[str | None, StringConstraints(min_length=2, max_length=120)] = None
    floor: Annotated[int | None, Field(ge=-10, le=200)] = None
    capacity: Annotated[int | None, Field(ge=1, le=2000)] = None
    is_active: bool | None = None


class RoomRead(SchemaModel):
    """Output schema representing a room entity from persistence."""

    id: UUID
    code: str
    building: str
    floor: int | None
    capacity: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


class SightingCreate(SchemaModel):
    """Input schema used to create one raw heartbeat recognition sighting."""

    student_id: UUID | None = None
    course_id: UUID
    room_id: UUID | None = None
    timestamp: datetime
    camera_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    confidence_score: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    embedding_reference: Annotated[str | None, StringConstraints(max_length=255)] = None


class SightingRead(SchemaModel):
    """Output schema representing one persisted sighting event."""

    id: UUID
    student_id: UUID | None
    course_id: UUID
    room_id: UUID | None
    timestamp: datetime
    camera_id: str
    confidence_score: float | None
    embedding_reference: str | None
    created_at: datetime
    updated_at: datetime


class ClassSessionRecordCreate(SchemaModel):
    """Input schema used to create one final aggregated class-session attendance record."""

    student_id: UUID
    course_id: UUID
    session_date: date
    status: AttendanceStatus = AttendanceStatus.ABSENT
    sighting_count: Annotated[int, Field(ge=0)] = 0
    required_sightings_threshold: Annotated[int, Field(ge=1)] = 1
    evaluated_at: datetime | None = None
    notes: Annotated[str | None, StringConstraints(max_length=4000)] = None


class ClassSessionRecordUpdate(SchemaModel):
    """Input schema used to patch mutable class-session attendance fields."""

    status: AttendanceStatus | None = None
    sighting_count: Annotated[int | None, Field(ge=0)] = None
    required_sightings_threshold: Annotated[int | None, Field(ge=1)] = None
    evaluated_at: datetime | None = None
    notes: Annotated[str | None, StringConstraints(max_length=4000)] = None


class ClassSessionRecordRead(SchemaModel):
    """Output schema representing one aggregated class-session attendance record."""

    id: UUID
    student_id: UUID
    course_id: UUID
    session_date: date
    status: AttendanceStatus
    sighting_count: int
    required_sightings_threshold: int
    evaluated_at: datetime
    notes: str | None
    created_at: datetime
    updated_at: datetime


class GovernanceLogCreate(SchemaModel):
    """Input schema used to append immutable governance audit events."""

    actor_user_id: UUID | None = None
    class_session_record_id: UUID | None = None
    action: Annotated[str, StringConstraints(min_length=2, max_length=64, pattern=r"^[A-Z_]+$")]
    entity_type: Annotated[str, StringConstraints(min_length=2, max_length=64, pattern=r"^[a-z_]+$")]
    entity_id: UUID | None = None
    reason: Annotated[str | None, StringConstraints(max_length=4000)] = None
    change_summary: dict[str, Any] = Field(default_factory=dict)
    request_id: UUID | None = None
    ip_address: Annotated[str | None, StringConstraints(max_length=45)] = None


class GovernanceLogRead(SchemaModel):
    """Output schema representing a governance audit log entry."""

    id: UUID
    actor_user_id: UUID | None
    class_session_record_id: UUID | None
    action: str
    entity_type: str
    entity_id: UUID | None
    reason: str | None
    change_summary: dict[str, Any]
    request_id: UUID | None
    ip_address: str | None
    created_at: datetime
    updated_at: datetime


class ImageTensorPayload(SchemaModel):
    """Validated frame payload carrying raw tensor bytes encoded as base64."""

    frame_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    data_base64: Annotated[str, StringConstraints(min_length=1)]
    height: Annotated[int, Field(ge=1, le=4096)]
    width: Annotated[int, Field(ge=1, le=4096)]
    channels: Annotated[int, Field(ge=1, le=4)] = 3
    dtype: Literal["uint8", "float32"] = "uint8"
    normalize: bool = True
    captured_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data_base64")
    @classmethod
    def validate_base64_data(cls, value: str) -> str:
        """Require valid non-empty base64 payload to prevent malformed tensor input."""
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("data_base64 must be valid base64 data.") from exc

        if len(decoded) == 0:
            raise ValueError("data_base64 must not decode to empty payload.")

        return value

    @model_validator(mode="after")
    def validate_tensor_shape(self) -> ImageTensorPayload:
        """Enforce byte-size consistency between payload and declared tensor dimensions."""
        raw = base64.b64decode(self.data_base64, validate=True)
        item_size = 1 if self.dtype == "uint8" else 4
        expected_size = self.height * self.width * self.channels * item_size

        if len(raw) != expected_size:
            raise ValueError(
                "data_base64 payload size does not match height * width * channels * dtype item size"
            )

        return self


class InferenceBatchRequest(SchemaModel):
    """Input schema used to enqueue one inference workload composed of one or more frames."""

    request_id: Annotated[str | None, StringConstraints(min_length=1, max_length=128)] = None
    session_id: UUID | None = None
    course_id: UUID | None = None
    room_id: UUID | None = None
    camera_id: Annotated[str | None, StringConstraints(min_length=1, max_length=128)] = None
    frames: Annotated[list[ImageTensorPayload], Field(min_length=1, max_length=128)]
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    liveness_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    include_embeddings: bool = False


class InferenceTaskAccepted(SchemaModel):
    """Response schema returned when an inference task is successfully queued."""

    task_id: str
    state: str
    queued_at: datetime
    frame_count: Annotated[int, Field(ge=1)]


class InferenceTaskStatus(SchemaModel):
    """Response schema exposing asynchronous task execution state and optional result data."""

    task_id: str
    state: str
    result: dict[str, Any] | None = None
    error: str | None = None


__all__ = [
    "ClassSessionRecordCreate",
    "ClassSessionRecordRead",
    "ClassSessionRecordUpdate",
    "CourseCreate",
    "CourseRead",
    "CourseUpdate",
    "GovernanceLogCreate",
    "GovernanceLogRead",
    "ImageTensorPayload",
    "InferenceBatchRequest",
    "InferenceTaskAccepted",
    "InferenceTaskStatus",
    "RoomCreate",
    "RoomRead",
    "RoomUpdate",
    "SchemaModel",
    "SightingCreate",
    "SightingRead",
    "StudentCreate",
    "StudentRead",
    "StudentUpdate",
    "UserCreate",
    "UserRead",
    "UserUpdate",
]
