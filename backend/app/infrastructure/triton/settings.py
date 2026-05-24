"""Triton client settings and environment variable parsing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _read_required_env(name: str) -> str:
    """Read a required environment variable and fail fast when not configured."""
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Environment variable {name} must be set.")
    return value.strip()


def _read_float_env(name: str, default: float, *, min_value: float) -> float:
    """Read a floating-point environment variable with lower-bound validation."""
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a floating-point number.") from exc

    if value < min_value:
        raise RuntimeError(
            f"Environment variable {name} must be greater than or equal to {min_value}."
        )

    return value


def _read_int_env(name: str, default: int, *, min_value: int) -> int:
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


def _read_bool_env(name: str, default: bool) -> bool:
    """Read a boolean environment variable using common truthy and falsy values."""
    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise RuntimeError(f"Environment variable {name} must be a boolean value.")


@dataclass(frozen=True, slots=True)
class TritonClientSettings:
    """Runtime settings controlling Triton connectivity and retry behavior."""

    url: str
    request_timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    ssl_enabled: bool
    total_retry_budget_seconds: float


@lru_cache(maxsize=1)
def get_triton_client_settings() -> TritonClientSettings:
    """Return cached Triton client settings sourced from environment variables."""
    return TritonClientSettings(
        url=_read_required_env("ATTENDANCE_TRITON_URL"),
        request_timeout_seconds=_read_float_env(
            "ATTENDANCE_TRITON_REQUEST_TIMEOUT_SECONDS",
            10.0,
            min_value=0.1,
        ),
        max_retries=_read_int_env("ATTENDANCE_TRITON_MAX_RETRIES", 3, min_value=0),
        retry_backoff_seconds=_read_float_env(
            "ATTENDANCE_TRITON_RETRY_BACKOFF_SECONDS",
            0.35,
            min_value=0.05,
        ),
        ssl_enabled=_read_bool_env("ATTENDANCE_TRITON_SSL_ENABLED", False),
        total_retry_budget_seconds=_read_float_env(
            "ATTENDANCE_TRITON_TOTAL_RETRY_BUDGET_SECONDS",
            60.0,
            min_value=1.0,
        ),
    )
