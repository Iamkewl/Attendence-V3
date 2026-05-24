"""Shared base class and constrained string type aliases used across all schema modules."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints


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
