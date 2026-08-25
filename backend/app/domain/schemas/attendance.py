"""Attendance sighting, class-session record, and governance audit schemas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any
from uuid import UUID

from pydantic import Field, StringConstraints

from app.domain.models import AttendanceStatus

from .common import SchemaModel


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


class ClassSessionRosterRecord(SchemaModel):
    """Output schema for one student's attendance record within a class session listing."""

    id: UUID
    student_id: UUID
    student_full_name: str
    status: str
    sightings_count: int
    required_sightings_threshold: int
    evaluated_at: datetime


class ClassSessionListResponse(SchemaModel):
    """Output schema for the class session roster endpoint."""

    course_id: UUID
    session_date: date
    required_sightings_threshold: int
    records: list[ClassSessionRosterRecord]


class ClassSessionOverrideRequest(SchemaModel):
    """Input schema for one manual attendance override (ATT-038).

    ``reason`` is REQUIRED — an override without its evidence defeats the
    governance ledger that records it (OVERRIDE_APPLY carries it verbatim).
    Only 'present' and 'absent' are settable by hand; 'late'/'excused'
    remain evaluator-managed states.
    """

    student_id: UUID
    status: Annotated[str, Field(pattern="^(present|absent)$")]
    reason: Annotated[str, StringConstraints(min_length=3, max_length=2000)]


class ClassSessionOverrideRead(SchemaModel):
    """Output schema for the applied manual override result."""

    id: UUID
    student_id: UUID
    course_id: UUID
    session_date: date
    status: AttendanceStatus
    previous_status: AttendanceStatus | None
    evaluated_at: datetime


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
