"""ATT-077 (#82): one source of truth for the identity-match threshold.

The 0.85 acceptance rule used to exist as a module constant consumed
independently by the ORM path and the raw-SQL batch path in
``matching.py`` — two call sites that could silently drift apart. The
constant is gone; both paths now read ``get_pipeline_settings().match_threshold``
(env override ``ATTENDANCE_MATCH_THRESHOLD``, validated 0 < value < 1).

Proven here three ways:
  1. settings default + env override + validation (unit);
  2. behavior: lowering the threshold via env flips a borderline probe on
     BOTH code paths, proving each really consumes the setting;
  3. structure: no hardcoded threshold survives in matching.py, and the old
     ``STRICT_SIMILARITY_THRESHOLD`` export is gone from the package surface.
"""

from __future__ import annotations

import inspect
import textwrap
import uuid

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from tests._fakes import _unit_norm_vector, vector_at_cosine

REFERENCE_SEED = 4321


def _session_factory_for(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


async def _seed_student_with_template(
    engine: AsyncEngine,
    reference: np.ndarray,
    *,
    label: str,
) -> uuid.UUID:
    """Insert User + Student + active StudentEmbedding. Returns student_id."""
    from app.core.security import hash_password
    from app.domain.models import Student, StudentEmbedding, User, UserRole

    async with _session_factory_for(engine)() as session:
        user = User(
            id=uuid.uuid4(),
            email=f"{label}-{uuid.uuid4().hex[:8]}@threshold.example",
            full_name=f"Threshold Fixture {label}",
            password_hash=hash_password("TestPass1!"),
            role=UserRole.AUDITOR,
            is_active=True,
        )
        session.add(user)
        await session.flush()

        student = Student(
            id=uuid.uuid4(),
            user_id=user.id,
            student_number=f"TH{uuid.uuid4().hex[:6].upper()}",
            program="Threshold Setting Fixture",
            enrollment_year=2024,
            is_active=True,
        )
        session.add(student)
        await session.flush()

        session.add(
            StudentEmbedding(
                id=uuid.uuid4(),
                student_id=student.id,
                embedding=reference.tolist(),
                pose_label="front",
                quality_score=1.0,
                is_active=True,
            )
        )
        await session.commit()
        return student.id


# ---------------------------------------------------------------------------
# 1. Settings: default, env override, fail-closed validation.
# ---------------------------------------------------------------------------


def test_att_077_default_threshold_is_0_85(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services.pipeline.settings import get_pipeline_settings

    monkeypatch.delenv("ATTENDANCE_MATCH_THRESHOLD", raising=False)
    get_pipeline_settings.cache_clear()
    try:
        assert get_pipeline_settings().match_threshold == pytest.approx(0.85)
    finally:
        get_pipeline_settings.cache_clear()


@pytest.mark.parametrize("raw", ["abc", "", "   "])
def test_att_077_malformed_or_empty_env_fails_or_defaults(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    from app.services.pipeline.settings import get_pipeline_settings

    if raw.strip():
        monkeypatch.setenv("ATTENDANCE_MATCH_THRESHOLD", raw)
        get_pipeline_settings.cache_clear()
        with pytest.raises(RuntimeError, match="floating-point"):
            get_pipeline_settings()
    else:
        # Empty/whitespace behaves like unset: fall back to the default
        # (same convention as _read_positive_int_env consumers).
        monkeypatch.setenv("ATTENDANCE_MATCH_THRESHOLD", raw)
        get_pipeline_settings.cache_clear()
        assert get_pipeline_settings().match_threshold == pytest.approx(0.85)
    get_pipeline_settings.cache_clear()


@pytest.mark.parametrize("raw", ["0", "0.0", "-0.5", "1", "1.0", "1.2"])
def test_att_077_out_of_range_env_fails_closed(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """The threshold must be strictly between 0 and 1 — never degrade."""
    from app.services.pipeline.settings import get_pipeline_settings

    monkeypatch.setenv("ATTENDANCE_MATCH_THRESHOLD", raw)
    get_pipeline_settings.cache_clear()
    with pytest.raises(RuntimeError, match="greater than 0 and less than 1"):
        get_pipeline_settings()
    get_pipeline_settings.cache_clear()


# ---------------------------------------------------------------------------
# 2. Behavior: an env-lowered threshold flips a borderline probe on BOTH paths.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_077_env_override_flows_to_both_matching_paths(
    test_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 0.60-similarity probe is rejected at 0.85 but accepted at 0.50.

    If either code path still read a hardcoded constant instead of the
    setting, its half of this assertion fails.
    """
    from app.services.pipeline.matching import (
        _resolve_nearest_embedding_match,
        _resolve_vector_matches,
    )
    from app.services.pipeline.settings import get_pipeline_settings

    monkeypatch.setenv("ATTENDANCE_MATCH_THRESHOLD", "0.5")
    get_pipeline_settings.cache_clear()
    try:
        assert get_pipeline_settings().match_threshold == pytest.approx(0.5)

        reference = _unit_norm_vector(seed=REFERENCE_SEED)
        student_id = await _seed_student_with_template(test_engine, reference, label="envflow")
        # 0.60 < default 0.85 (would reject) but >= override 0.50 (accepts).
        probe = vector_at_cosine(reference, 0.60)

        batch_match = (await _resolve_vector_matches([probe]))[0]
        assert batch_match.student_id == student_id, (
            "ATT-077: batch (raw-SQL) path did not honor ATTENDANCE_MATCH_THRESHOLD"
        )

        async with _session_factory_for(test_engine)() as session:
            orm_match = await _resolve_nearest_embedding_match(session, probe)
        assert orm_match.student_id == student_id, (
            "ATT-077: ORM path did not honor ATTENDANCE_MATCH_THRESHOLD"
        )
    finally:
        monkeypatch.delenv("ATTENDANCE_MATCH_THRESHOLD", raising=False)
        get_pipeline_settings.cache_clear()


@pytest.mark.asyncio
async def test_att_077_default_threshold_still_rejects_borderline_probe(
    test_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no env override the historical 0.85 rule holds on both paths."""
    from app.services.pipeline.matching import (
        _resolve_nearest_embedding_match,
        _resolve_vector_matches,
    )
    from app.services.pipeline.settings import get_pipeline_settings

    monkeypatch.delenv("ATTENDANCE_MATCH_THRESHOLD", raising=False)
    get_pipeline_settings.cache_clear()
    try:
        reference = _unit_norm_vector(seed=REFERENCE_SEED)
        await _seed_student_with_template(test_engine, reference, label="default")
        probe = vector_at_cosine(reference, 0.70)

        batch_match = (await _resolve_vector_matches([probe]))[0]
        assert batch_match.student_id is None
        assert batch_match.embedding_id is not None

        async with _session_factory_for(test_engine)() as session:
            orm_match = await _resolve_nearest_embedding_match(session, probe)
        assert orm_match.student_id is None
        assert orm_match.cosine_similarity == pytest.approx(0.70, abs=1e-3)
    finally:
        get_pipeline_settings.cache_clear()


# ---------------------------------------------------------------------------
# 3. Structure: no hardcoded threshold, no stale constant on package surface.
# ---------------------------------------------------------------------------


def test_att_077_no_stale_constant_on_pipeline_surface() -> None:
    """STRICT_SIMILARITY_THRESHOLD must be gone from matching + re-exports."""
    import importlib

    from app.services import pipeline_service
    matching = importlib.import_module("app.services.pipeline.matching")
    pkg = importlib.import_module("app.services.pipeline")
    facade = pipeline_service

    for module, name in ((matching, "matching"), (pkg, "pipeline.__init__"), (facade, "pipeline_service")):
        assert not hasattr(module, "STRICT_SIMILARITY_THRESHOLD"), (
            f"ATT-077 regression: {name} still exposes STRICT_SIMILARITY_THRESHOLD; "
            f"the threshold must come from PipelineSettings only."
        )
        if hasattr(module, "__all__") and module.__all__ is not None:
            assert "STRICT_SIMILARITY_THRESHOLD" not in module.__all__, name


def test_att_077_both_matching_paths_read_the_settings_attribute() -> None:
    """Source scan: each acceptance site reads match_threshold from settings."""
    from app.services.pipeline import matching as matching_module

    for func_name in ("_resolve_nearest_embedding_match", "_classify_match"):
        src = textwrap.dedent(inspect.getsource(getattr(matching_module, func_name)))
        assert "get_pipeline_settings().match_threshold" in src, (
            f"ATT-077 regression: {func_name} no longer reads "
            f"get_pipeline_settings().match_threshold — the single-source "
            f"rule has been bypassed."
        )
        assert "< 0.85" not in src and "0.85" not in src, (
            f"ATT-077 regression: {func_name} hardcodes a threshold literal."
        )


def test_att_077_orchestrator_reports_threshold_from_settings() -> None:
    """The serialized per-frame ``match_threshold`` must come from settings."""
    from app.services.pipeline import orchestrator as orchestrator_module

    src = textwrap.dedent(inspect.getsource(orchestrator_module.process_inference_batch))
    assert "settings.match_threshold" in src, (
        "ATT-077 regression: process_inference_batch no longer reports the "
        "threshold from PipelineSettings."
    )
