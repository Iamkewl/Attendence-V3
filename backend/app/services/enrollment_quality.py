"""Enrollment quality gate policy (ATT-029), shared by API and CLI callers.

This module is the single source of truth for the enrollment minimum
quality threshold. It was relocated verbatim from
``app.api.v1.students`` when the bulk enrollment importer
(``scripts/import_enrollments.py``) needed the identical gate: services
must not import from ``app.api.*``, and duplicating a fail-closed policy
function invites drift. The API module re-exports
``_resolve_enrollment_min_quality`` so existing call sites and tests
(``from app.api.v1.students import _resolve_enrollment_min_quality``)
stay green untouched.

Semantics are unchanged from ATT-029:

- Read + validate ``ATTENDANCE_ENROLLMENT_MIN_QUALITY`` per call (never
  cached at import time) so operators/tests can change it at runtime.
- Default 0.5. Acceptable range [0.0, 1.0]. The gate comparison at every
  call site is strict: ``quality_score < min_quality`` refuses; equality
  passes.
- Malformed or out-of-range values FAIL CLOSED with RuntimeError — the
  caller surfaces it (API: HTTP 500; importer: run abort, exit 2)
  rather than silently accepting garbage embeddings under bad config.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np


_ENROLLMENT_MIN_QUALITY_DEFAULT = 0.5
_ENROLLMENT_MIN_QUALITY_ENV_NAME = "ATTENDANCE_ENROLLMENT_MIN_QUALITY"

# ---------------------------------------------------------------------------
# Live enrollment preview diagnostics (owner-requested phone-style UX).
#
# Cheap numpy-only heuristics evaluated per preview frame so the client gets
# actionable retake hints BEFORE hitting the real enroll endpoint. Thresholds
# are module constants (not env-tunable) on purpose: they describe universal
# image-capture ergonomics, not deployment policy.
# ---------------------------------------------------------------------------

PREVIEW_REASON_NO_FACE = "NO_FACE_DETECTED"
PREVIEW_REASON_POOR_LIGHTING = "POOR_LIGHTING"
PREVIEW_REASON_MOTION_BLUR = "MOTION_BLUR"
PREVIEW_REASON_FACE_TOO_SMALL = "FACE_TOO_SMALL"
PREVIEW_REASON_NOT_CENTERED = "NOT_CENTERED"
PREVIEW_REASON_MULTIPLE_FACES = "MULTIPLE_FACES"
PREVIEW_REASON_LOW_QUALITY = "LOW_QUALITY"

# Mean Rec.601 luma bounds: below 45 is a dark room, above 215 blows out the
# sensor and washes the face.
_MIN_MEAN_LUMA = 45.0
_MAX_MEAN_LUMA = 215.0

# Variance of the horizontal gradient: sharp frames have rich high-frequency
# content; motion blur flattens it toward zero.
_MOTION_BLUR_GRADIENT_VARIANCE_MIN = 100.0

# The largest face box must cover at least 4% of the frame area.
_MIN_FACE_AREA_FRACTION = 0.04

# The largest face center must sit inside the central 60% region.
_CENTER_REGION_HALF_SPAN = 0.2  # central band = [0.2, 0.8] per axis

_LUMA_WEIGHTS_RGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _resolve_enrollment_min_quality() -> float:
    """Read + validate the enrollment-quality minimum env var per call.

    Returns the configured minimum quality (default 0.5). Malformed values
    FAIL CLOSED — the strictest acceptable quality is 1.0, so any parser
    error or out-of-range value yields a RuntimeError, avoiding silent
    acceptance of a low-quality embedding under bad configuration.

    Acceptable range: [0.0, 1.0]. 0.0 disables the gate (matches pre-ATT-029
    behavior, kept as an escape hatch for testing or operator override);
    1.0 requires perfect quality (rarely reachable in practice).
    """
    raw = os.getenv(_ENROLLMENT_MIN_QUALITY_ENV_NAME)
    if raw is None or not raw.strip():
        return _ENROLLMENT_MIN_QUALITY_DEFAULT

    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable {_ENROLLMENT_MIN_QUALITY_ENV_NAME} must be a "
            f"float in [0.0, 1.0]; got {raw!r}."
        ) from exc

    if not (0.0 <= value <= 1.0):
        raise RuntimeError(
            f"Environment variable {_ENROLLMENT_MIN_QUALITY_ENV_NAME} must be a "
            f"float in [0.0, 1.0]; got {value!r}."
        )
    return value


def mean_luma(frame_rgb: np.ndarray) -> float:
    """Mean Rec.601 luma of an HWC RGB frame (values expected in 0..255)."""
    tensor = np.asarray(frame_rgb, dtype=np.float32)
    if tensor.ndim != 3 or tensor.shape[2] < 3:
        raise ValueError("Frame must be a rank-3 HWC RGB tensor.")
    luma = tensor[..., :3] @ _LUMA_WEIGHTS_RGB
    return float(luma.mean())


def horizontal_gradient_variance(frame_rgb: np.ndarray) -> float:
    """Variance of horizontal luminance gradients — a cheap blur proxy.

    Sharp frames carry high-frequency detail (large gradient variance);
    motion-blurred frames collapse toward zero.
    """
    tensor = np.asarray(frame_rgb, dtype=np.float32)
    if tensor.ndim != 3 or tensor.shape[2] < 3:
        raise ValueError("Frame must be a rank-3 HWC RGB tensor.")
    luma = tensor[..., :3] @ _LUMA_WEIGHTS_RGB
    gradient = np.diff(luma, axis=1)
    return float(gradient.var())


def preview_reasons(
    frame_rgb: np.ndarray,
    bboxes_px: Sequence[tuple[float, float, float, float]],
) -> list[str]:
    """Evaluate the preview diagnostics for one frame.

    ``bboxes_px`` are pixel-space ``(x, y, w, h)`` boxes from the detector.
    Returns the reason-code list in deterministic order; an empty list means
    the frame passes every diagnostic and is ready to capture. Zero boxes is
    reported by the caller as ``NO_FACE_DETECTED`` (the route owns that path).
    """
    if not bboxes_px:
        return [PREVIEW_REASON_NO_FACE]

    height, width = np.asarray(frame_rgb).shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Frame dimensions must be positive.")

    reasons: list[str] = []

    luma = mean_luma(frame_rgb)
    if luma < _MIN_MEAN_LUMA or luma > _MAX_MEAN_LUMA:
        reasons.append(PREVIEW_REASON_POOR_LIGHTING)

    if horizontal_gradient_variance(frame_rgb) < _MOTION_BLUR_GRADIENT_VARIANCE_MIN:
        reasons.append(PREVIEW_REASON_MOTION_BLUR)

    x, y, box_w, box_h = max(bboxes_px, key=lambda bbox: bbox[2] * bbox[3])
    frame_area = float(width * height)
    if (box_w * box_h) / frame_area < _MIN_FACE_AREA_FRACTION:
        reasons.append(PREVIEW_REASON_FACE_TOO_SMALL)

    center_x = (x + box_w / 2.0) / width
    center_y = (y + box_h / 2.0) / height
    span = _CENTER_REGION_HALF_SPAN
    if not (span <= center_x <= 1.0 - span and span <= center_y <= 1.0 - span):
        reasons.append(PREVIEW_REASON_NOT_CENTERED)

    if len(bboxes_px) > 1:
        reasons.append(PREVIEW_REASON_MULTIPLE_FACES)

    return reasons
