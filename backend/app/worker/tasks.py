"""Celery task orchestration for inference processing and heartbeat sighting logging."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session_factory
from app.domain.models import ClassSessionRecord, Course, Sighting
from app.domain.schemas import InferenceBatchRequest
from app.services.attendance_service import (
    AttendanceNotFoundError,
    AttendanceService,
    AttendanceValidationError,
)
from app.services.pipeline_service import process_inference_batch
from app.worker.celery_app import celery_app
from app.worker.triton_client import (
    TritonInferenceError,
    TritonModelUnavailableError,
    TritonServerUnavailableError,
    TritonTimeoutError,
)


LOGGER = logging.getLogger(__name__)


def _parse_uuid(value: Any) -> UUID | None:
    """Parse UUID values from task payloads without raising validation exceptions."""
    if value is None:
        return None

    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    """Parse ISO timestamps from payload values and normalize them to UTC."""
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_optional_float(value: Any) -> float | None:
    """Parse optional numeric values from untyped task payload dictionaries."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_int(value: Any, *, field_name: str, default: int) -> int:
    """Coerce optional task payload values to positive integers with strict validation."""
    if value is None:
        return default

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc

    if parsed < 1:
        raise ValueError(f"{field_name} must be greater than or equal to 1.")

    return parsed


async def _select_scheduled_active_course_ids(
    session: AsyncSession,
    *,
    target_date: date,
) -> list[UUID]:
    """Resolve active courses with attendance activity for the target date."""
    window_start = datetime.combine(target_date, time.min, tzinfo=UTC)
    window_end = window_start + timedelta(days=1)

    has_sightings_today = exists(
        select(Sighting.id)
        .where(Sighting.course_id == Course.id)
        .where(Sighting.timestamp >= window_start)
        .where(Sighting.timestamp < window_end)
    )
    has_records_today = exists(
        select(ClassSessionRecord.id)
        .where(ClassSessionRecord.course_id == Course.id)
        .where(ClassSessionRecord.session_date == target_date)
    )

    rows = await session.execute(
        select(Course.id)
        .where(Course.is_active.is_(True))
        .where(or_(has_sightings_today, has_records_today))
        .order_by(Course.id)
    )

    return list(rows.scalars().all())


async def _evaluate_daily_attendance(required_sightings_threshold: int) -> dict[str, Any]:
    """Evaluate daily attendance for active courses with current-date attendance activity."""
    current_date = datetime.now(tz=UTC).date()

    summary: dict[str, Any] = {
        "date": current_date.isoformat(),
        "required_sightings_threshold": required_sightings_threshold,
        "scheduled_active_courses": 0,
        "courses_evaluated": 0,
        "courses_failed": 0,
        "records_upserted": 0,
        "failed_course_ids": [],
    }

    session_factory = get_session_factory()
    async with session_factory() as session:
        course_ids = await _select_scheduled_active_course_ids(session, target_date=current_date)
        summary["scheduled_active_courses"] = len(course_ids)

        if not course_ids:
            return summary

        attendance_service = AttendanceService(session=session)
        failed_course_ids: list[str] = []
        courses_evaluated = 0
        records_upserted = 0

        for course_id in course_ids:
            try:
                updated_records = await attendance_service.evaluate_class_attendance(
                    course_id=course_id,
                    date=current_date,
                    required_sightings_threshold=required_sightings_threshold,
                )
                courses_evaluated += 1
                records_upserted += len(updated_records)
            except (AttendanceNotFoundError, AttendanceValidationError) as exc:
                await session.rollback()
                failed_course_ids.append(str(course_id))
                LOGGER.warning(
                    "Skipping attendance evaluation for course_id=%s date=%s due to validation error: %s",
                    course_id,
                    current_date.isoformat(),
                    exc,
                )
            except Exception:
                await session.rollback()
                failed_course_ids.append(str(course_id))
                LOGGER.exception(
                    "Unexpected failure while evaluating attendance for course_id=%s date=%s.",
                    course_id,
                    current_date.isoformat(),
                )

        summary["courses_evaluated"] = courses_evaluated
        summary["courses_failed"] = len(failed_course_ids)
        summary["records_upserted"] = records_upserted
        summary["failed_course_ids"] = failed_course_ids

    return summary


async def _log_pipeline_sightings(
    request: InferenceBatchRequest,
    pipeline_result: dict[str, Any],
) -> int:
    """Persist recognized heartbeat sightings derived from one inference pipeline run."""
    if request.course_id is None:
        return 0

    camera_id = request.camera_id.strip() if request.camera_id else "unknown_camera"
    generated_at = _parse_datetime(pipeline_result.get("generated_at")) or datetime.now(tz=UTC)

    session_factory = get_session_factory()
    logged_sightings = 0

    async with session_factory() as session:
        attendance_service = AttendanceService(session=session)

        for item in pipeline_result.get("results", []):
            if not isinstance(item, dict):
                continue

            student_id = _parse_uuid(item.get("student_id"))

            timestamp = _parse_datetime(item.get("captured_at")) or generated_at
            confidence_score = _parse_optional_float(item.get("detection_score"))
            embedding_reference = str(item["identity"]) if item.get("identity") is not None else None

            try:
                await attendance_service.log_sighting(
                    student_id=student_id,
                    camera_id=camera_id,
                    course_id=request.course_id,
                    timestamp=timestamp,
                    room_id=request.room_id,
                    confidence_score=confidence_score,
                    embedding_reference=embedding_reference,
                )
                logged_sightings += 1
            except (AttendanceNotFoundError, AttendanceValidationError):
                LOGGER.warning(
                    "Skipping invalid sighting for student_id=%s course_id=%s.",
                    student_id,
                    request.course_id,
                    exc_info=True,
                )
            except Exception:
                LOGGER.exception(
                    "Unexpected failure while logging sighting for student_id=%s course_id=%s.",
                    student_id,
                    request.course_id,
                )

    return logged_sightings


@celery_app.task(
    bind=True,
    name="app.worker.tasks.run_inference_pipeline",
    autoretry_for=(
        TritonTimeoutError,
        TritonServerUnavailableError,
        TritonModelUnavailableError,
        TritonInferenceError,
    ),
    retry_backoff=True,
    retry_backoff_max=30,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
)
def run_inference_pipeline(self, payload: dict[str, Any]) -> dict[str, Any]:
    """Celery task entrypoint for full attendance AI inference and sighting persistence."""
    try:
        request = InferenceBatchRequest.model_validate(payload)
    except ValidationError as exc:
        LOGGER.exception("Inference payload validation failed.")
        raise ValueError("Inference payload validation failed.") from exc

    try:
        pipeline_result = asyncio.run(process_inference_batch(request))
    except (
        TritonTimeoutError,
        TritonServerUnavailableError,
        TritonModelUnavailableError,
        TritonInferenceError,
    ):
        LOGGER.exception("Retryable Triton failure while processing task %s.", self.request.id)
        raise
    except Exception as exc:
        LOGGER.exception("Non-retryable pipeline failure while processing task %s.", self.request.id)
        raise RuntimeError("Inference pipeline execution failed.") from exc

    try:
        logged_sightings = asyncio.run(_log_pipeline_sightings(request, pipeline_result))
    except Exception:
        LOGGER.exception("Failed to persist heartbeat sightings for task %s.", self.request.id)
        logged_sightings = 0

    pipeline_result["sightings_logged"] = logged_sightings
    pipeline_result["task_id"] = self.request.id
    pipeline_result["task_state"] = "SUCCESS"
    return pipeline_result


@celery_app.task(
    bind=True,
    name="app.worker.tasks.task_evaluate_daily_attendance",
)
def task_evaluate_daily_attendance(
    self,
    required_sightings_threshold: int | None = None,
) -> dict[str, Any]:
    """Celery task entrypoint for periodic daily attendance aggregation."""
    threshold = _coerce_positive_int(
        required_sightings_threshold,
        field_name="required_sightings_threshold",
        default=3,
    )

    try:
        summary = asyncio.run(_evaluate_daily_attendance(required_sightings_threshold=threshold))
    except Exception as exc:
        LOGGER.exception(
            "Daily attendance evaluation failed for task %s.",
            self.request.id,
        )
        raise RuntimeError("Daily attendance evaluation execution failed.") from exc

    summary["task_id"] = self.request.id
    summary["task_state"] = "SUCCESS"
    return summary


__all__ = ["run_inference_pipeline", "task_evaluate_daily_attendance"]
