"""Student profile and enrollment embedding schemas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated
from uuid import UUID

from pydantic import Field, model_validator

from .common import ProgramStr, SchemaModel, StudentNumberStr


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
    biometric_consent_status: str
    biometric_consent_at: datetime | None
    created_at: datetime
    updated_at: datetime


_BIOMETRIC_CONSENT_STATUSES = ("granted", "denied", "withdrawn")


class StudentConsentUpdate(SchemaModel):
    """Input schema for recording one biometric consent decision (ATT-044).

    'pending' is intentionally NOT settable here — it is only the initial
    backfill state; every recorded decision is a definitive answer.
    """

    status: Annotated[str, Field(pattern="^(granted|denied|withdrawn)$")]
    reason: Annotated[str | None, Field(max_length=2000)] = None

    @model_validator(mode="after")
    def validate_status_vocabulary(self) -> StudentConsentUpdate:
        """Pin the exact consent vocabulary (defense in depth vs the regex)."""
        if self.status not in _BIOMETRIC_CONSENT_STATUSES:
            raise ValueError("status must be one of: granted, denied, withdrawn")
        return self


class EnrollmentCoverageRow(SchemaModel):
    """One per-student row of the admin enrollment-coverage aggregate.

    Powers the future coverage dashboard: template inventory, recency, and
    consent state side by side. Poses are active-template pose labels only;
    sightings_last_7d counts recognized sightings in the trailing week.
    """

    student_id: UUID
    student_number: str
    full_name: str
    active_template_count: int
    poses: list[str]
    last_enrolled_at: datetime | None
    biometric_consent_status: str
    sightings_last_7d: int


class StudentEnrollmentRead(SchemaModel):
    """Output schema representing one persisted enrollment embedding template."""

    id: UUID
    student_id: UUID
    pose_label: str
    quality_score: float
    is_active: bool
    created_at: datetime
