"""ATT-029 regression: enrollment quality gate refuses to write a
low-quality face embedding as an active template.

Pre-fix: the /students/{id}/enroll route extracted an LVFace embedding via
`extract_enrollment_embedding` and stored it as the active template with
whatever `quality_score` LVFace returned — no server-side minimum. A 0.05-
quality embedding could be persisted as the active template with no server
pushback, and every subsequent recognition pass at the strict 0.85 cosine
threshold would then be confused by garbage.

Post-fix: enroll reads `ATTENDANCE_ENROLLMENT_MIN_QUALITY` (default 0.5)
per request. When `quality_score < threshold`, the API returns 422
"Image quality too low for enrollment" — no StudentEmbedding row, no
TemplateAuditLog row, no rotation of pre-existing templates to inactive.
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import patch

import numpy as np
import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


async def _provision_student_for_enrollment(_session_factory) -> tuple[uuid.UUID, uuid.UUID]:
    """Insert an instructor + active student; return (student_id, instructor_id)."""
    from app.domain.models import Student, User, UserRole
    from app.core.security import hash_password

    async with _session_factory() as session:
        instructor = User(
            id=uuid.uuid4(),
            email="instructor-att-029@test.example",
            full_name="Instructor ATT-029",
            password_hash=hash_password("TestPass1!"),
            role=UserRole.INSTRUCTOR,
            is_active=True,
        )
        session.add(instructor)
        await session.flush()

        student = Student(
            id=uuid.uuid4(),
            user_id=instructor.id,
            student_number="STU-ATT-029",
            program="Computer Science",
            enrollment_year=2024,
            is_active=True,
        )
        session.add(student)
        await session.commit()
        return student.id, instructor.id


def _enrollment_image_bytes() -> bytes:
    """A 16x16 white PNG that survives _decode_enrollment_image checks."""
    from io import BytesIO
    from PIL import Image
    img = Image.new("RGB", (16, 16), color=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_FAKE_ENROLL_IMAGE_FILE = ("probe.png", _enrollment_image_bytes(), "image/png")


def _instructor_cookies(instructor_id: uuid.UUID) -> dict:
    """Issue a session cookie for the instructor."""
    from app.domain.models import UserRole
    from app.core.security import create_access_token, get_security_settings
    token, _ = create_access_token(subject=instructor_id, role=UserRole.INSTRUCTOR)
    settings = get_security_settings()
    return {settings.access_cookie_name: token}


def _patch_quality(quality: float):
    """Patch extract_enrollment_embedding to return a fixed quality_score."""
    return patch(
        "app.api.v1.students.extract_enrollment_embedding",
        return_value=(np.zeros(512, dtype=np.float32), quality),
    )


class _EnvOverride:
    """Context manager for ATTENDANCE_ENROLLMENT_MIN_QUALITY manipulation."""

    def __init__(self, value: str | None) -> None:
        self._value = value
        self._backup: str | None = None
        self._had_backup: bool = False

    def __enter__(self) -> "_EnvOverride":
        self._had_backup = "ATTENDANCE_ENROLLMENT_MIN_QUALITY" in os.environ
        self._backup = os.environ.get("ATTENDANCE_ENROLLMENT_MIN_QUALITY")
        if self._value is None:
            os.environ.pop("ATTENDANCE_ENROLLMENT_MIN_QUALITY", None)
        else:
            os.environ["ATTENDANCE_ENROLLMENT_MIN_QUALITY"] = self._value
        return self

    def __exit__(self, *exc) -> None:
        if self._had_backup:
            os.environ["ATTENDANCE_ENROLLMENT_MIN_QUALITY"] = self._backup  # type: ignore[assignment]
        else:
            os.environ.pop("ATTENDANCE_ENROLLMENT_MIN_QUALITY", None)


# ---------------------------------------------------------------------------
# ACCEPT (literal): an upload with a low-quality synthetic face returns 422,
# NOT 201.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_029_low_quality_enrollment_returns_422(
    async_client: AsyncClient,
    _session_factory,
) -> None:
    """The issue's literal ACCEPT — low-quality upload returns 422, not 201.

    We monkeypatch the route's bound `extract_enrollment_embedding` to
    return quality=0.05 (well below the default 0.5 threshold). The test
    asserts HTTP 422 + a detail mentioning "quality".
    """
    student_id, instructor_id = await _provision_student_for_enrollment(_session_factory)
    cookies = _instructor_cookies(instructor_id)

    with _EnvOverride(None), _patch_quality(0.05):
        response = await async_client.post(
            f"/api/v1/students/{student_id}/enroll",
            files={"image_file": _FAKE_ENROLL_IMAGE_FILE},
            cookies=cookies,
        )

    assert response.status_code == 422, (
        f"Expected 422 for low-quality enrollment; got {response.status_code}. "
        f"Body: {response.text}"
    )
    detail = response.json().get("detail", "")
    assert "quality" in detail.lower(), f"422 detail should mention 'quality'; got {detail!r}"


# ---------------------------------------------------------------------------
# Boundaries: equality passes (gate is strict <); just below fails.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_029_quality_at_default_threshold_is_accepted(
    async_client: AsyncClient,
    _session_factory,
) -> None:
    """quality == 0.5 (exactly the default threshold) is accepted because
    the gate is `quality_score < min_quality` (strict less-than).
    """
    student_id, instructor_id = await _provision_student_for_enrollment(_session_factory)
    cookies = _instructor_cookies(instructor_id)

    with _EnvOverride(None), _patch_quality(0.5):
        response = await async_client.post(
            f"/api/v1/students/{student_id}/enroll",
            files={"image_file": _FAKE_ENROLL_IMAGE_FILE},
            cookies=cookies,
        )

    assert response.status_code == 201


@pytest.mark.asyncio
async def test_att_029_quality_just_below_default_threshold_returns_422(
    async_client: AsyncClient,
    _session_factory,
) -> None:
    """quality == 0.49 (just below default 0.5) is refused.

    Pins off-by-one — a maintainer flipping `<` to `<=` would silently
    refuse the boundary-acceptable case.
    """
    student_id, instructor_id = await _provision_student_for_enrollment(_session_factory)
    cookies = _instructor_cookies(instructor_id)

    with _EnvOverride(None), _patch_quality(0.49):
        response = await async_client.post(
            f"/api/v1/students/{student_id}/enroll",
            files={"image_file": _FAKE_ENROLL_IMAGE_FILE},
            cookies=cookies,
        )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Env var honored: raising the threshold refuses what default accepts;
# the 0.0 escape hatch accepts what default refuses.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_029_higher_env_threshold_refuses_what_default_accepts(
    async_client: AsyncClient,
    _session_factory,
) -> None:
    """ATTENDANCE_ENROLLMENT_MIN_QUALITY=0.9 refuses quality=0.6 (which
    the default 0.5 would accept).
    """
    student_id, instructor_id = await _provision_student_for_enrollment(_session_factory)
    cookies = _instructor_cookies(instructor_id)

    with _EnvOverride("0.9"), _patch_quality(0.6):
        response = await async_client.post(
            f"/api/v1/students/{student_id}/enroll",
            files={"image_file": _FAKE_ENROLL_IMAGE_FILE},
            cookies=cookies,
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_att_029_disabled_threshold_accepts_any_quality(
    async_client: AsyncClient,
    _session_factory,
) -> None:
    """ATTENDANCE_ENROLLMENT_MIN_QUALITY=0.0 (the escape hatch) accepts
    quality=0.05 — matches pre-fix behavior, kept as an operator override.
    """
    student_id, instructor_id = await _provision_student_for_enrollment(_session_factory)
    cookies = _instructor_cookies(instructor_id)

    with _EnvOverride("0.0"), _patch_quality(0.05):
        response = await async_client.post(
            f"/api/v1/students/{student_id}/enroll",
            files={"image_file": _FAKE_ENROLL_IMAGE_FILE},
            cookies=cookies,
        )

    assert response.status_code == 201


# ---------------------------------------------------------------------------
# No side effects on refusal: the gate must NOT create a StudentEmbedding
# row or rotate pre-existing templates to inactive.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_029_low_quality_refusal_writes_no_student_embedding_row(
    async_client: AsyncClient,
    _session_factory,
) -> None:
    """A 422 refusal must NOT create a StudentEmbedding row.

    Pre-fix accepted any quality and unconditionally created the embedding
    + audit log. Post-fix, refusal short-circuits before the DB flush +
    commit.
    """
    student_id, instructor_id = await _provision_student_for_enrollment(_session_factory)
    cookies = _instructor_cookies(instructor_id)

    with _EnvOverride(None), _patch_quality(0.05):
        response = await async_client.post(
            f"/api/v1/students/{student_id}/enroll",
            files={"image_file": _FAKE_ENROLL_IMAGE_FILE},
            cookies=cookies,
        )

    assert response.status_code == 422

    from sqlalchemy import select
    from app.domain.models import StudentEmbedding

    async with _session_factory() as session:
        rows = (
            await session.execute(
                select(StudentEmbedding).where(
                    StudentEmbedding.student_id == student_id
                )
            )
        ).scalars().all()
    assert rows == [], (
        f"Expected NO StudentEmbedding rows after refusal; got {len(rows)} "
        "— the 422 must short-circuit before any DB write."
    )


@pytest.mark.asyncio
async def test_att_029_low_quality_refusal_preserves_pre_existing_active_template(
    async_client: AsyncClient,
    _session_factory,
) -> None:
    """Pre-existing active templates must NOT be rotated to inactive on refusal.

    The pre-fix code archived the prior active template + created a new
    one unconditionally. Post-fix, a 422 refusal preserves the student's
    pre-existing enrollment state.
    """
    student_id, instructor_id = await _provision_student_for_enrollment(_session_factory)
    cookies = _instructor_cookies(instructor_id)

    # Insert one pre-existing active template at quality=0.99 (good).
    from sqlalchemy import select
    from app.domain.models import StudentEmbedding, TemplateAuditLog

    pre_existing_emb_id = uuid.uuid4()
    async with _session_factory() as session:
        session.add(
            StudentEmbedding(
                id=pre_existing_emb_id,
                student_id=student_id,
                embedding=np.zeros(512, dtype=np.float32).tolist(),
                pose_label="front",
                quality_score=0.99,
                is_active=True,
            )
        )
        await session.commit()

    # Attempt a NEW enrollment that fails the quality gate.
    with _EnvOverride(None), _patch_quality(0.05):
        response = await async_client.post(
            f"/api/v1/students/{student_id}/enroll",
            files={"image_file": _FAKE_ENROLL_IMAGE_FILE},
            cookies=cookies,
        )

    assert response.status_code == 422

    async with _session_factory() as session:
        rows = (
            await session.execute(
                select(StudentEmbedding).where(
                    StudentEmbedding.student_id == student_id
                )
            )
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].id == pre_existing_emb_id
        assert rows[0].is_active is True, (
            "Pre-existing active template must NOT be rotated to inactive "
            "on a 422 refusal — pre-existing enrollment state is preserved."
        )
        audit_rows = (
            await session.execute(
                select(TemplateAuditLog).where(
                    TemplateAuditLog.student_id == student_id
                )
            )
        ).scalars().all()
        assert audit_rows == [], (
            "Expected NO TemplateAuditLog rows after a 422 refusal — no "
            "rotation, no creation, no audit. "
            f"Got {len(audit_rows)} rows."
        )


# ---------------------------------------------------------------------------
# Fail-closed on bad configuration (security-adjacent — bad env surfaces
# as 500, not silent acceptance).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_029_malformed_env_threshold_surfaces_as_500(
    async_client: AsyncClient,
    _session_factory,
) -> None:
    """A malformed ATTENDANCE_ENROLLMENT_MIN_QUALITY raises at request
    time (the gate fails closed rather than silently accepting garbage
    under bad config).
    """
    student_id, instructor_id = await _provision_student_for_enrollment(_session_factory)
    cookies = _instructor_cookies(instructor_id)

    with _EnvOverride("not-a-number"), _patch_quality(0.95):
        response = await async_client.post(
            f"/api/v1/students/{student_id}/enroll",
            files={"image_file": _FAKE_ENROLL_IMAGE_FILE},
            cookies=cookies,
        )

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# Source-level / helper contracts (no HTTP transport needed).
# ---------------------------------------------------------------------------


def test_att_029_helper_returns_default_when_env_unset() -> None:
    from app.api.v1.students import _resolve_enrollment_min_quality
    with _EnvOverride(None):
        assert _resolve_enrollment_min_quality() == 0.5


def test_att_029_helper_env_override() -> None:
    from app.api.v1.students import _resolve_enrollment_min_quality
    with _EnvOverride("0.7"):
        assert _resolve_enrollment_min_quality() == 0.7


def test_att_029_helper_out_of_range_raises() -> None:
    from app.api.v1.students import _resolve_enrollment_min_quality
    for bad in ["1.5", "-0.1", "2.0"]:
        with _EnvOverride(bad):
            with pytest.raises(RuntimeError, match="in \\[0.0, 1.0\\]"):
                _resolve_enrollment_min_quality()


def test_att_029_helper_malformed_raises() -> None:
    from app.api.v1.students import _resolve_enrollment_min_quality
    for bad in ["not-a-number", "abc", "none"]:
        with _EnvOverride(bad):
            with pytest.raises(RuntimeError, match="must be a float"):
                _resolve_enrollment_min_quality()
