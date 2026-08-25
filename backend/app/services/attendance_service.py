"""Attendance domain service for heartbeat logging and temporal aggregation workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from uuid import UUID, uuid4

from sqlalchemy import func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.pubsub import LIVE_ATTENDANCE_CHANNEL, RedisPubSubManager
from app.domain.models import AttendanceStatus, ClassSessionRecord, Course, Room, Sighting, Student, User
from app.domain.schemas import ClassSessionListResponse, ClassSessionRecordCreate, ClassSessionRecordUpdate, ClassSessionRosterRecord
from app.services.audit_service import AuditEvent, AuditService, GovernanceAction, emit
from app.services.base import AsyncCRUDService


LOGGER = logging.getLogger(__name__)

# Fresh rows created by a manual override carry the same default threshold
# the beat evaluator and roster listing use; any later aggregation run
# overwrites it with the course's configured threshold anyway.
_MANUAL_OVERRIDE_DEFAULT_THRESHOLD = 3

# Overrides persist as a notes-prefixed verdict row (ATT-038); the export
# renders them back out by recognizing this prefix — one source of truth,
# no duplicated persistence scheme.
_OVERRIDE_NOTES_PREFIX = "manual_override:"


@dataclass(frozen=True)
class AttendanceExportRow:
    """One CSV-ready roster row (ATT-039). Mirrors the daily evaluation's
    output shape plus per-day sighting aggregates; never carries embeddings."""

    student_number: str
    student_name: str
    status: str  # evaluated verdict, or "unknown" when only sighted
    confidence_score: str  # "" when the day's sightings carry no scores
    last_sighting_at: str  # ISO 8601 UTC, "" when never sighted that day
    override_applied: bool
    override_reason: str


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
        actor: User | None = None,
    ) -> None:
        super().__init__(session=session, model=ClassSessionRecord)
        self._pubsub_manager = pubsub_manager or RedisPubSubManager()
        # None (the default) means the system actor — e.g. the Celery
        # aggregation task. Governance rows then carry actor_user_id IS NULL.
        self._actor = actor

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

        # Insert is committed. Publish is best-effort: a failure here must not
        # propagate to the caller so that the persisted sighting is counted correctly.
        try:
            await self._publish_live_sighting_event(created)
        except Exception:
            LOGGER.exception(
                "Live sighting publish failed after sighting %s was persisted.",
                created.id,
            )

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
            # ATTENDANCE_EVALUATE is mandatory (D1): the aggregation run that
            # derives attendance verdicts is itself a governance event, written
            # in the SAME transaction as the upsert (system actor when the
            # Celery task constructs this service without an actor). Runs that
            # find nothing to evaluate return early above and emit no row.
            await AuditService(self.session).emit(
                AuditEvent(
                    action=GovernanceAction.ATTENDANCE_EVALUATE,
                    entity_type="class_session_record",
                    entity_id=course_id,
                    actor_user_id=self._actor.id if self._actor is not None else None,
                    change_summary={
                        "source": "celery",
                        "session_date": date.isoformat(),
                        "records_upserted": len(updated_records),
                        "threshold": required_sightings_threshold,
                    },
                )
            )

        return updated_records

    async def list_session_records(
        self,
        *,
        course_id: UUID,
        session_date: date,
        default_threshold: int = 3,
    ) -> ClassSessionListResponse:
        """Return the attendance roster for one course on one day without triggering evaluation."""
        await self._require_course(course_id)

        existing_records = (
            await self.session.execute(
                select(ClassSessionRecord)
                .where(ClassSessionRecord.course_id == course_id)
                .where(ClassSessionRecord.session_date == session_date)
            )
        ).scalars().all()

        if not existing_records:
            return ClassSessionListResponse(
                course_id=course_id,
                session_date=session_date,
                required_sightings_threshold=default_threshold,
                records=[],
            )

        threshold = existing_records[0].required_sightings_threshold

        student_ids = [r.student_id for r in existing_records]
        student_rows = (
            await self.session.execute(
                select(Student.id, Student.user_id)
                .where(Student.id.in_(student_ids))
            )
        ).all()
        student_user_ids = {row.id: row.user_id for row in student_rows}

        from app.domain.models import User as UserModel
        user_rows = (
            await self.session.execute(
                select(UserModel.id, UserModel.full_name)
                .where(UserModel.id.in_(student_user_ids.values()))
            )
        ).all()
        user_full_names = {row.id: row.full_name for row in user_rows}

        window_start = datetime.combine(session_date, time.min, tzinfo=UTC)
        window_end = window_start + timedelta(days=1)
        sighting_rows = (
            await self.session.execute(
                select(
                    Sighting.student_id,
                    func.count(Sighting.id).label("sighting_count"),
                )
                .where(Sighting.course_id == course_id)
                .where(Sighting.timestamp >= window_start)
                .where(Sighting.timestamp < window_end)
                .where(Sighting.student_id.in_(student_ids))
                .group_by(Sighting.student_id)
            )
        ).all()
        sighting_counts = {row.student_id: int(row.sighting_count) for row in sighting_rows}

        records: list[ClassSessionRosterRecord] = []
        for record in existing_records:
            user_id = student_user_ids.get(record.student_id)
            full_name = user_full_names.get(user_id, "") if user_id is not None else ""
            records.append(
                ClassSessionRosterRecord(
                    id=record.id,
                    student_id=record.student_id,
                    student_full_name=full_name,
                    status=record.status.value,
                    sightings_count=sighting_counts.get(record.student_id, 0),
                    required_sightings_threshold=record.required_sightings_threshold,
                    evaluated_at=record.evaluated_at,
                )
            )

        return ClassSessionListResponse(
            course_id=course_id,
            session_date=session_date,
            required_sightings_threshold=threshold,
            records=records,
        )

    async def export_daily_roster(
        self,
        *,
        course_id: UUID,
        session_date: date,
    ) -> list[AttendanceExportRow]:
        """Assemble one CSV-ready roster row per enrolled/seen student (ATT-039).

        Reuses the daily-evaluation domain shape: the roster is the union of
        aggregated ``class_session_records`` for the day and students sighted
        in the course that day — no separate enrollment table exists, so
        "enrolled/seen" follows exactly the same population
        ``evaluate_class_attendance`` writes. Students sighted but not yet
        evaluated export as status ``unknown``; evaluated rows keep their
        verdict (including 'excused'/'late'). Override state is derived from
        the ATT-038 notes prefix rather than a second persistence scheme.

        Emits an ADVISORY ``EXPORT`` governance event (log-and-continue): a
        ledger failure must never block delivery of a roster the requester is
        already authorized to see. The summary carries counts only.

        Memory: class sizes are hundreds, so a plain list is used instead of
        a server-side cursor; embeddings are never selected.
        """
        await self._require_course(course_id)

        window_start = datetime.combine(session_date, time.min, tzinfo=UTC)
        window_end = window_start + timedelta(days=1)

        existing_records = (
            await self.session.execute(
                select(ClassSessionRecord)
                .where(ClassSessionRecord.course_id == course_id)
                .where(ClassSessionRecord.session_date == session_date)
            )
        ).scalars().all()
        records_by_student = {record.student_id: record for record in existing_records}

        sighting_rows = (
            await self.session.execute(
                select(
                    Sighting.student_id,
                    func.avg(Sighting.confidence_score).label("avg_confidence"),
                    func.max(Sighting.timestamp).label("last_sighting_at"),
                )
                .where(Sighting.course_id == course_id)
                .where(Sighting.timestamp >= window_start)
                .where(Sighting.timestamp < window_end)
                .where(Sighting.student_id.is_not(None))
                .group_by(Sighting.student_id)
            )
        ).all()
        sightings_by_student = {
            row.student_id: row for row in sighting_rows if row.student_id is not None
        }

        student_ids = set(records_by_student) | set(sightings_by_student)
        export_rows: list[AttendanceExportRow] = []
        if not student_ids:
            return export_rows

        identity_rows = (
            await self.session.execute(
                select(Student.id, Student.student_number, User.full_name)
                .join(User, Student.user_id == User.id)
                .where(Student.id.in_(student_ids))
            )
        ).all()
        identities = {row.id: row for row in identity_rows}

        for student_id in sorted(
            student_ids,
            key=lambda sid: (identities[sid].student_number, str(sid)),
        ):
            record = records_by_student.get(student_id)
            sighting = sightings_by_student.get(student_id)
            notes = record.notes if record is not None else None
            override_applied = bool(notes) and notes.startswith(_OVERRIDE_NOTES_PREFIX)
            avg_confidence = getattr(sighting, "avg_confidence", None)
            last_sighting_at = getattr(sighting, "last_sighting_at", None)
            export_rows.append(
                AttendanceExportRow(
                    student_number=identities[student_id].student_number,
                    student_name=identities[student_id].full_name or "",
                    status=record.status.value if record is not None else "unknown",
                    confidence_score=(
                        f"{float(avg_confidence):.4f}" if avg_confidence is not None else ""
                    ),
                    last_sighting_at=self._serialize_datetime(last_sighting_at),
                    override_applied=override_applied,
                    override_reason=(
                        notes[len(_OVERRIDE_NOTES_PREFIX):].strip() if override_applied else ""
                    ),
                )
            )

        # Advisory emission (D1 revision for exports): strict=False so an
        # audit-storage failure degrades to a log line. Read-only workload,
        # so the rollback the wrapper performs cannot lose business writes.
        async with self.transaction():
            await emit(
                self.session,
                AuditEvent(
                    action=GovernanceAction.EXPORT,
                    entity_type="class_session_record",
                    entity_id=course_id,
                    actor_user_id=self._actor.id if self._actor is not None else None,
                    change_summary={
                        "session_date": session_date.isoformat(),
                        "format": "csv",
                        "rows": len(export_rows),
                    },
                ),
                strict=False,
            )

        return export_rows

    async def apply_manual_override(
        self,
        *,
        course_id: UUID,
        student_id: UUID,
        status: AttendanceStatus,
        reason: str,
    ) -> tuple[ClassSessionRecord, AttendanceStatus | None]:
        """Upsert TODAY's class-session record for one student (ATT-038).

        Manual overrides are last-write-wins and idempotent: re-submitting
        the same verdict updates ``status``/``notes``/``evaluated_at`` and
        emits a fresh OVERRIDE_APPLY event. Only PRESENT/ABSENT are accepted;
        'late'/'excused' stay evaluator-managed. The OVERRIDE_APPLY governance
        row is MANDATORY (D1) and joins this same transaction — an override
        can never land without its evidence. ``class_session_record_id``
        finally gets a writer (design §1.2), so auditors can trace exactly
        which roster row was overridden.

        Returns ``(record, previous_status)`` where ``previous_status`` is
        None when no aggregated row existed yet (the override creates one).
        """
        if status not in (AttendanceStatus.PRESENT, AttendanceStatus.ABSENT):
            raise AttendanceValidationError(
                "Manual overrides accept only 'present' or 'absent'."
            )
        if not reason.strip():
            raise AttendanceValidationError("An override reason is required.")

        await self._require_course(course_id)
        await self._require_student(student_id)

        session_date = datetime.now(tz=UTC).date()
        evaluated_at = datetime.now(tz=UTC)

        async with self.transaction():
            previous_status = await self.session.scalar(
                select(ClassSessionRecord.status)
                .where(ClassSessionRecord.student_id == student_id)
                .where(ClassSessionRecord.course_id == course_id)
                .where(ClassSessionRecord.session_date == session_date)
            )

            insert_stmt = pg_insert(ClassSessionRecord).values(
                student_id=student_id,
                course_id=course_id,
                session_date=session_date,
                status=status,
                sighting_count=0,
                required_sightings_threshold=_MANUAL_OVERRIDE_DEFAULT_THRESHOLD,
                evaluated_at=evaluated_at,
                notes=f"manual_override: {reason.strip()}",
            )
            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=[
                    ClassSessionRecord.student_id,
                    ClassSessionRecord.course_id,
                    ClassSessionRecord.session_date,
                ],
                set_={
                    "status": insert_stmt.excluded.status,
                    "notes": insert_stmt.excluded.notes,
                    "evaluated_at": insert_stmt.excluded.evaluated_at,
                },
            ).returning(ClassSessionRecord)

            record = (await self.session.execute(upsert_stmt)).scalar_one()

            await AuditService(self.session).emit(
                AuditEvent(
                    action=GovernanceAction.OVERRIDE_APPLY,
                    entity_type="class_session_record",
                    entity_id=record.id,
                    class_session_record_id=record.id,
                    actor_user_id=self._actor.id if self._actor is not None else None,
                    reason=reason.strip(),
                    change_summary={
                        "student_id": str(student_id),
                        "session_date": session_date.isoformat(),
                        "from": previous_status.value if previous_status else None,
                        "to": status.value,
                    },
                )
            )

        return record, previous_status

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
    "AttendanceExportRow",
    "AttendanceNotFoundError",
    "AttendanceService",
    "AttendanceServiceError",
    "AttendanceValidationError",
    "ClassSessionListResponse",
]