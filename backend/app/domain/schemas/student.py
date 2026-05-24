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
