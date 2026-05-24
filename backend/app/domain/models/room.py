"""Room ORM model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, Index, Integer, SmallInteger, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import text

from ._base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from .sighting import Sighting


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
