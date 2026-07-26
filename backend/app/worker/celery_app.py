"""Celery application configuration for asynchronous attendance inference workloads."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import timedelta
from functools import lru_cache

from celery import Celery
from celery.signals import task_postrun, worker_process_init
from kombu import Queue

LOGGER = logging.getLogger(__name__)


def _read_required_env(name: str) -> str:
    """Read a required environment variable and fail fast when missing."""
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Environment variable {name} must be set.")
    return value.strip()


def _read_int_env(name: str, default: int, *, min_value: int = 0) -> int:
    """Read an integer environment variable with explicit lower-bound validation."""
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer.") from exc

    if value < min_value:
        raise RuntimeError(
            f"Environment variable {name} must be greater than or equal to {min_value}."
        )

    return value


def _build_beat_schedule(
    *,
    attendance_eval_interval_seconds: int,
    attendance_required_sightings_threshold: int,
    demo_mode_enabled: bool,
    demo_sighting_interval_seconds: int,
) -> dict[str, object]:
    """Construct the beat_schedule dict, adding demo entries only when demo mode is active."""
    schedule: dict[str, object] = {
        "attendance-evaluation-hourly": {
            "task": "app.worker.tasks.task_evaluate_daily_attendance",
            "schedule": timedelta(seconds=attendance_eval_interval_seconds),
            "kwargs": {
                "required_sightings_threshold": attendance_required_sightings_threshold,
            },
            "options": {
                "queue": "attendance_aggregation",
                "routing_key": "attendance.aggregation",
            },
        },
    }

    if demo_mode_enabled:
        schedule["demo-synthetic-sighting"] = {
            "task": "app.worker.tasks.demo_emit_sighting",
            "schedule": timedelta(seconds=demo_sighting_interval_seconds),
            "options": {
                "queue": "attendance_aggregation",
                "routing_key": "attendance.aggregation",
            },
        }

    return schedule


@lru_cache(maxsize=1)
def get_celery_app() -> Celery:
    """Create and cache the Celery application used by inference workers."""
    redis_url = _read_required_env("ATTENDANCE_REDIS_URL")
    broker_url = os.getenv("ATTENDANCE_CELERY_BROKER_URL", redis_url).strip()
    result_backend = os.getenv("ATTENDANCE_CELERY_RESULT_BACKEND", redis_url).strip()

    soft_time_limit_seconds = _read_int_env(
        "ATTENDANCE_CELERY_TASK_SOFT_TIME_LIMIT_SECONDS",
        120,
        min_value=1,
    )
    hard_time_limit_seconds = _read_int_env(
        "ATTENDANCE_CELERY_TASK_TIME_LIMIT_SECONDS",
        180,
        min_value=soft_time_limit_seconds + 1,
    )
    attendance_eval_interval_seconds = _read_int_env(
        "ATTENDANCE_CELERY_ATTENDANCE_EVALUATION_INTERVAL_SECONDS",
        3_600,
        min_value=60,
    )
    attendance_required_sightings_threshold = _read_int_env(
        "ATTENDANCE_CELERY_ATTENDANCE_REQUIRED_SIGHTINGS_THRESHOLD",
        3,
        min_value=1,
    )

    demo_mode_enabled = (
        os.getenv("ATTENDANCE_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
    )
    demo_sighting_interval_seconds = _read_int_env(
        "ATTENDANCE_DEMO_SIGHTING_INTERVAL_SECONDS",
        5,
        min_value=1,
    )

    app = Celery(
        "attendance_v2_worker",
        broker=broker_url,
        backend=result_backend,
        include=["app.worker.tasks"],
    )

    app.conf.update(
        accept_content=["json"],
        task_serializer="json",
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_default_queue="inference",
        task_default_exchange="inference",
        task_default_routing_key="inference.default",
        task_queues=(
            Queue("inference", routing_key="inference.default"),
            Queue("inference_priority", routing_key="inference.priority"),
            Queue("attendance_aggregation", routing_key="attendance.aggregation"),
        ),
        task_routes={
            "app.worker.tasks.task_evaluate_daily_attendance": {
                "queue": "attendance_aggregation",
                "routing_key": "attendance.aggregation",
            },
        },
        beat_schedule=_build_beat_schedule(
            attendance_eval_interval_seconds=attendance_eval_interval_seconds,
            attendance_required_sightings_threshold=attendance_required_sightings_threshold,
            demo_mode_enabled=demo_mode_enabled,
            demo_sighting_interval_seconds=demo_sighting_interval_seconds,
        ),
        task_track_started=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=_read_int_env(
            "ATTENDANCE_CELERY_PREFETCH_MULTIPLIER",
            1,
            min_value=1,
        ),
        task_soft_time_limit=soft_time_limit_seconds,
        task_time_limit=hard_time_limit_seconds,
        result_expires=_read_int_env(
            "ATTENDANCE_CELERY_RESULT_EXPIRES_SECONDS",
            86_400,
            min_value=60,
        ),
        broker_connection_retry=True,
        broker_connection_retry_on_startup=True,
        broker_transport_options={
            "visibility_timeout": _read_int_env(
                "ATTENDANCE_CELERY_VISIBILITY_TIMEOUT_SECONDS",
                3_600,
                min_value=60,
            ),
            "socket_keepalive": True,
            "socket_timeout": _read_int_env(
                "ATTENDANCE_CELERY_SOCKET_TIMEOUT_SECONDS",
                5,
                min_value=1,
            ),
            "retry_on_timeout": True,
        },
        result_backend_transport_options={
            "retry_policy": {
                "timeout": _read_int_env(
                    "ATTENDANCE_CELERY_BACKEND_RETRY_TIMEOUT_SECONDS",
                    5,
                    min_value=1,
                ),
            },
        },
    )

    return app


celery_app = get_celery_app()


@worker_process_init.connect
def _prime_triton_readiness(**_: object) -> None:
    """Pre-verify Triton readiness for configured models once per worker process."""
    yolo_model = os.getenv("ATTENDANCE_TRITON_YOLO_MODEL_NAME", "yolov12").strip() or "yolov12"
    lvface_model = os.getenv("ATTENDANCE_TRITON_LVFACE_MODEL_NAME", "lvface").strip() or "lvface"

    try:
        from app.infrastructure.triton import get_triton_client

        get_triton_client().prime_readiness([yolo_model, lvface_model])
    except Exception:
        LOGGER.warning(
            "Failed to prime Triton readiness during worker init; readiness will be verified on the first inference call.",
            exc_info=True,
        )
    else:
        LOGGER.info(
            "Primed Triton readiness for models: %s, %s",
            yolo_model,
            lvface_model,
        )


@task_postrun.connect
def _dispose_engine_after_task(**_: object) -> None:
    """Dispose the asyncpg engine and clear its caches after every Celery task.

    Why: each Celery task invokes ``asyncio.run(...)`` which mints a fresh event
    loop. asyncpg's connection pool (and therefore the SQLAlchemy ``AsyncEngine``)
    binds to the loop that created it. Without disposal, ``get_session_factory()``
    returns the cached engine from task #1's loop on task #2's loop, producing
    ``RuntimeError: Future ... attached to a different loop`` on every subsequent
    task. Disposing and clearing the caches after each task guarantees each task
    builds a fresh engine on its own loop.
    """
    from app.core.database import dispose_engine

    try:
        asyncio.run(dispose_engine())
    except Exception:
        LOGGER.warning(
            "Failed to dispose asyncpg engine after task; next task will retry with a fresh engine.",
            exc_info=True,
        )


__all__ = ["celery_app", "get_celery_app"]
