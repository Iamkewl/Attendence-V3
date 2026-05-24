"""Course catalog and room schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import Field, StringConstraints

from .common import CourseCodeStr, RoomCodeStr, SchemaModel


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
