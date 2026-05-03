"""Celery application configuration for asynchronous attendance inference workloads."""

from __future__ import annotations

import os
from datetime import timedelta
from functools import lru_cache

from celery import Celery
from kombu import Queue


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
            "app.worker.tasks.run_inference_pipeline": {
                "queue": "inference",
                "routing_key": "inference.default",
            },
            "app.worker.tasks.task_evaluate_daily_attendance": {
                "queue": "attendance_aggregation",
                "routing_key": "attendance.aggregation",
            },
        },
        beat_schedule={
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
        },
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


__all__ = ["celery_app", "get_celery_app"]
