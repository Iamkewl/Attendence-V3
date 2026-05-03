"""Attendance domain service for heartbeat logging and temporal aggregation workflows."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time, timedelta
from uuid import UUID, uuid4

from sqlalchemy import func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.pubsub import LIVE_ATTENDANCE_CHANNEL, RedisPubSubManager
from app.domain.models import AttendanceStatus, ClassSessionRecord, Course, Room, Sighting, Student
from app.domain.schemas import ClassSessionRecordCreate, ClassSessionRecordUpdate
from app.services.base import AsyncCRUDService


LOGGER = logging.getLogger(__name__)


class AttendanceServiceError(Exception):
    """Base error type for attendance service-level failures."""


class AttendanceNotFoundError(AttendanceServiceError):
    """Raised when a required domain entity cannot be found."""


class AttendanceValidationError(AttendanceServiceError):
    """Raised when attendance input violates business constraints."""


class AttendanceService(
    AsyncCRUDService[ClassSessionRecord, ClassSessionRecordCreate, ClassSessionRecordUpdate]
):
    """Application service responsible for heartbeat logging and attendance aggregation."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        pubsub_manager: RedisPubSubManager | None = None,
    ) -> None:
        super().__init__(session=session, model=ClassSessionRecord)
        self._pubsub_manager = pubsub_manager or RedisPubSubManager()

    async def log_sighting(
        self,
        *,
        student_id: UUID | None,
        camera_id: str,
        course_id: UUID,
        timestamp: datetime,
        room_id: UUID | None = None,
        confidence_score: float | None = None,
        embedding_reference: str | None = None,
    ) -> Sighting:
        """Persist one raw heartbeat sighting and publish a realtime event for dashboards."""
        normalized_camera_id = camera_id.strip()
        if not normalized_camera_id:
            raise AttendanceValidationError("camera_id must not be blank.")

        if confidence_score is not None and not (0.0 <= confidence_score <= 1.0):
            raise AttendanceValidationError("confidence_score must be between 0 and 1.")

        sighting_timestamp = self._as_utc(timestamp)

        async with self.transaction():
            if student_id is not None:
                await self._require_student(student_id)
            await self._require_course(course_id)
            await self._require_room(room_id)

            created = (
                await self.session.execute(
                    insert(Sighting)
                    .values(
                        student_id=student_id,
                        course_id=course_id,
                        room_id=room_id,
                        timestamp=sighting_timestamp,
                        camera_id=normalized_camera_id,
                        confidence_score=confidence_score,
                        embedding_reference=embedding_reference,
                    )
                    .returning(Sighting)
                )
            ).scalar_one()

        await self._publish_live_sighting_event(created)
        return created

    async def evaluate_class_attendance(
        self,
        *,
        course_id: UUID,
        date: date,
        required_sightings_threshold: int,
    ) -> list[ClassSessionRecord]:
        """Aggregate daily sightings into final class-session attendance records."""
        if required_sightings_threshold < 1:
            raise AttendanceValidationError("required_sightings_threshold must be at least 1.")

        await self._require_course(course_id)

        window_start = datetime.combine(date, time.min, tzinfo=UTC)
        window_end = window_start + timedelta(days=1)
        evaluated_at = datetime.now(tz=UTC)

        sighting_rows = (
            await self.session.execute(
                select(
                    Sighting.student_id,
                    func.count(Sighting.id).label("sighting_count"),
                )
                .where(Sighting.course_id == course_id)
                .where(Sighting.timestamp >= window_start)
                .where(Sighting.timestamp < window_end)
                .where(Sighting.student_id.is_not(None))
                .group_by(Sighting.student_id)
            )
        ).all()
        sighting_counts = {student_id: int(sighting_count) for student_id, sighting_count in sighting_rows}

        existing_rows = (
            await self.session.execute(
                select(ClassSessionRecord.student_id, ClassSessionRecord.status)
                .where(ClassSessionRecord.course_id == course_id)
                .where(ClassSessionRecord.session_date == date)
            )
        ).all()
        existing_statuses = {student_id: status for student_id, status in existing_rows}

        student_ids = set(sighting_counts.keys()) | set(existing_statuses.keys())
        if not student_ids:
            return []

        upsert_rows: list[dict[str, object]] = []
        for student_id in sorted(student_ids, key=str):
            sighting_count = sighting_counts.get(student_id, 0)
            existing_status = existing_statuses.get(student_id)

            if existing_status == AttendanceStatus.EXCUSED:
                status = AttendanceStatus.EXCUSED
            elif sighting_count >= required_sightings_threshold:
                status = AttendanceStatus.PRESENT
            else:
                status = AttendanceStatus.ABSENT

            upsert_rows.append(
                {
                    "student_id": student_id,
                    "course_id": course_id,
                    "session_date": date,
                    "status": status,
                    "sighting_count": sighting_count,
                    "required_sightings_threshold": required_sightings_threshold,
                    "evaluated_at": evaluated_at,
                }
            )

        insert_stmt = pg_insert(ClassSessionRecord).values(upsert_rows)
        upsert_stmt = (
            insert_stmt.on_conflict_do_update(
                index_elements=[
                    ClassSessionRecord.student_id,
                    ClassSessionRecord.course_id,
                    ClassSessionRecord.session_date,
                ],
                set_={
                    "status": insert_stmt.excluded.status,
                    "sighting_count": insert_stmt.excluded.sighting_count,
                    "required_sightings_threshold": insert_stmt.excluded.required_sightings_threshold,
                    "evaluated_at": insert_stmt.excluded.evaluated_at,
                },
            )
            .returning(ClassSessionRecord)
        )

        async with self.transaction():
            updated_records = list((await self.session.execute(upsert_stmt)).scalars().all())

        return updated_records

    async def _publish_live_sighting_event(self, sighting: Sighting) -> None:
        """Publish a serialized sighting payload to realtime subscribers."""
        payload: dict[str, object] = {
            "event_id": str(uuid4()),
            "event_type": "sighting_logged",
            "emitted_at": datetime.now(tz=UTC).isoformat(),
            "sighting": {
                "id": str(sighting.id),
                "student_id": str(sighting.student_id) if sighting.student_id is not None else None,
                "course_id": str(sighting.course_id),
                "room_id": str(sighting.room_id) if sighting.room_id is not None else None,
                "timestamp": self._serialize_datetime(sighting.timestamp),
                "camera_id": sighting.camera_id,
                "confidence_score": sighting.confidence_score,
                "embedding_reference": sighting.embedding_reference,
                "created_at": self._serialize_datetime(sighting.created_at),
                "updated_at": self._serialize_datetime(sighting.updated_at),
            },
        }

        try:
            await self._pubsub_manager.publish_json(LIVE_ATTENDANCE_CHANNEL, payload)
        except Exception:
            LOGGER.exception(
                "Failed to publish sighting %s to Redis channel %s.",
                sighting.id,
                LIVE_ATTENDANCE_CHANNEL,
            )

    async def _require_student(self, student_id: UUID) -> None:
        """Ensure attendance sighting can only be recorded for an existing active student."""
        student = (
            await self.session.execute(
                select(Student).where(Student.id == student_id),
            )
        ).scalar_one_or_none()

        if student is None:
            raise AttendanceNotFoundError("Student does not exist.")

        if not student.is_active:
            raise AttendanceValidationError("Student is inactive.")

    async def _require_course(self, course_id: UUID) -> None:
        """Ensure attendance operations target an existing active course."""
        course = (
            await self.session.execute(
                select(Course).where(Course.id == course_id),
            )
        ).scalar_one_or_none()

        if course is None:
            raise AttendanceNotFoundError("Course does not exist.")

        if not course.is_active:
            raise AttendanceValidationError("Course is inactive.")

    async def _require_room(self, room_id: UUID | None) -> None:
        """Ensure optional room references point to an active room entity."""
        if room_id is None:
            return

        room = (
            await self.session.execute(
                select(Room).where(Room.id == room_id),
            )
        ).scalar_one_or_none()

        if room is None:
            raise AttendanceNotFoundError("Room does not exist.")

        if not room.is_active:
            raise AttendanceValidationError("Room is inactive.")

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        """Normalize naive and timezone-aware datetimes to UTC consistently."""
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @staticmethod
    def _serialize_datetime(value: datetime | None) -> str | None:
        """Serialize datetimes as ISO 8601 UTC timestamps for realtime payloads."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC).isoformat()
        return value.astimezone(UTC).isoformat()


__all__ = [
    "AttendanceNotFoundError",
    "AttendanceService",
    "AttendanceServiceError",
    "AttendanceValidationError",
]