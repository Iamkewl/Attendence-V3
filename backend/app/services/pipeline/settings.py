"""Pipeline runtime configuration: PipelineSettings dataclass and env-var readers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


_VALID_BBOX_FORMATS = {"xyxy", "cxcywh", "auto"}


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """Runtime settings for the asynchronous AI inference pipeline."""

    yolo_model_name: str
    yolo_input_name: str
    yolo_output_name: str | None
    lvface_model_name: str
    lvface_input_name: str
    lvface_liveness_output_name: str | None
    lvface_embedding_output_name: str | None
    face_crop_size: int
    tracking_distance_threshold: float
    yolo_bbox_format: str
    yolo_bbox_normalized: bool


def _read_int_env(name: str, default: int, *, min_value: int) -> int:
    """Read integer environment configuration with explicit lower-bound validation."""
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


def _read_float_env(name: str, default: float, *, min_value: float) -> float:
    """Read floating-point environment configuration with lower-bound validation."""
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


def _read_str_env(name: str, default: str) -> str:
    """Read string environment configuration and return default when unset."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _read_bool_env(name: str, default: bool) -> bool:
    """Read boolean environment configuration; accepts 1/true/yes (case-insensitive)."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def _read_optional_str_env(name: str) -> str | None:
    """Read optional string environment configuration and normalize empty values to None."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


@lru_cache(maxsize=1)
def get_pipeline_settings() -> PipelineSettings:
    """Return cached pipeline runtime settings from environment variables."""
    yolo_bbox_format = _read_str_env("ATTENDANCE_TRITON_YOLO_BBOX_FORMAT", "xyxy")
    if yolo_bbox_format not in _VALID_BBOX_FORMATS:
        raise RuntimeError(
            f"ATTENDANCE_TRITON_YOLO_BBOX_FORMAT must be one of {sorted(_VALID_BBOX_FORMATS)}; "
            f"got {yolo_bbox_format!r}."
        )
    return PipelineSettings(
        yolo_model_name=_read_str_env("ATTENDANCE_TRITON_YOLO_MODEL_NAME", "yolov12"),
        yolo_input_name=_read_str_env("ATTENDANCE_TRITON_YOLO_INPUT_NAME", "INPUT__0"),
        yolo_output_name=_read_optional_str_env("ATTENDANCE_TRITON_YOLO_OUTPUT_NAME"),
        lvface_model_name=_read_str_env("ATTENDANCE_TRITON_LVFACE_MODEL_NAME", "lvface"),
        lvface_input_name=_read_str_env("ATTENDANCE_TRITON_LVFACE_INPUT_NAME", "INPUT__0"),
        lvface_liveness_output_name=_read_optional_str_env(
            "ATTENDANCE_TRITON_LVFACE_LIVENESS_OUTPUT_NAME"
        ),
        lvface_embedding_output_name=_read_optional_str_env(
            "ATTENDANCE_TRITON_LVFACE_EMBEDDING_OUTPUT_NAME"
        ),
        face_crop_size=_read_int_env("ATTENDANCE_PIPELINE_FACE_CROP_SIZE", 112, min_value=32),
        tracking_distance_threshold=_read_float_env(
            "ATTENDANCE_PIPELINE_TRACKING_DISTANCE_THRESHOLD",
            96.0,
            min_value=1.0,
        ),
        yolo_bbox_format=yolo_bbox_format,
        yolo_bbox_normalized=_read_bool_env("ATTENDANCE_TRITON_YOLO_BBOX_NORMALIZED", False),
    )
