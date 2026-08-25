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


_ENROLLMENT_MIN_QUALITY_DEFAULT = 0.5
_ENROLLMENT_MIN_QUALITY_ENV_NAME = "ATTENDANCE_ENROLLMENT_MIN_QUALITY"


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
