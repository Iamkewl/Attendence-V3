"""Non-degenerate coverage of the 0.85 identity-acceptance threshold.

WHY THIS FILE EXISTS
--------------------
Before this file, every embedding in the suite was bit-identical. ``FakeTritonGrpcClient``
returned ``_unit_norm_vector()`` with the default seed for every face, and the only seeded
template used ``_unit_norm_vector(seed=42)`` — the same vector. Cosine similarity was
therefore exactly 1.0 in every test that had ever run.

The consequence: the *rejection* branches in ``matching.py`` (lines ~59 and ~111) had never
executed once. Both could have been deleted outright and the suite would have stayed green.
For a biometric attendance system that means CI could not detect a false-positive
identification — the wrong student being marked present — which is the product's most
consequential failure mode.

These tests probe at controlled cosine similarities either side of the threshold, so an
inverted comparison, a removed threshold, or a similarity/distance sign error fails the suite.

TWO CODE PATHS
--------------
``matching.py`` implements the acceptance rule twice, independently (filed as ATT-077):

* ``_resolve_nearest_embedding_match``  — ORM path, similarity built in Python
* ``_resolve_vector_matches``           — raw-SQL batch path, similarity built in SQL

**Production uses the batch path.** Both are covered here, plus an explicit test that they
agree, so the two cannot silently drift apart.

NOTE ON pgvector: ``<=>`` returns cosine DISTANCE, not similarity. Both paths convert with
``1 - distance``. ``test_reported_value_is_similarity_not_distance`` pins that direction
specifically — it is the single easiest thing to get backwards here.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from tests._fakes import _unit_norm_vector, vector_at_cosine


# Deliberately not seed 42: that is the value the pre-existing fixtures use, and reusing it
# would reintroduce the degenerate "everything is a perfect match" condition.
REFERENCE_SEED = 1234
OTHER_STUDENT_SEED = 999


# (cosine similarity of probe to the enrolled template, should it be accepted as a match)
#
# The threshold is 0.85. 0.86 / 0.84 sit +/-0.01 either side — close enough that an inverted
# comparison fails, far enough that float4 storage in pgvector (~7 significant digits) cannot
# flip the result. An exactly-at-threshold case is deliberately omitted: "cosine == 0.85"
# is not reliably representable, so such a test would be flaky rather than strict.
_THRESHOLD_CASES = [
    (1.00, True),
    (0.95, True),
    (0.90, True),
    (0.86, True),
    (0.84, False),
    (0.60, False),
    (0.00, False),
    (-0.50, False),
]


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
) -> tuple[uuid.UUID, uuid.UUID]:
    """Insert User + Student + active StudentEmbedding. Returns (student_id, embedding_id)."""
    from app.core.security import hash_password
    from app.domain.models import Student, StudentEmbedding, User, UserRole

    async with _session_factory_for(engine)() as session:
        user = User(
            id=uuid.uuid4(),
            email=f"{label}-{uuid.uuid4().hex[:8]}@match.example",
            full_name=f"Match Fixture {label}",
            password_hash=hash_password("TestPass1!"),
            role=UserRole.AUDITOR,
            is_active=True,
        )
        session.add(user)
        await session.flush()

        student = Student(
            id=uuid.uuid4(),
            user_id=user.id,
            student_number=f"MT{uuid.uuid4().hex[:6].upper()}",
            program="Threshold Fixture Program",
            enrollment_year=2024,
            is_active=True,
        )
        session.add(student)
        await session.flush()

        embedding = StudentEmbedding(
            id=uuid.uuid4(),
            student_id=student.id,
            embedding=reference.tolist(),
            pose_label="front",
            quality_score=1.0,
            is_active=True,
        )
        session.add(embedding)
        await session.commit()

        return student.id, embedding.id


# ---------------------------------------------------------------------------
# The fixture generator itself must be correct, or every assertion below is
# meaningless. Verify it independently of the database.
#
# NOTE: these two need no event loop and would naturally be written as plain
# sync tests — but they MUST be async. Running any sync test before an async one
# tears down the session-scoped event loop, after which every subsequent async
# test dies with "RuntimeError: There is no current event loop in thread
# 'MainThread'". The pre-existing suite only avoids this by alphabetical luck:
# test_smoke_realtime.py holds the only sync tests and happens to sort last.
# Filed separately as a harness defect; worked around here so this file does not
# depend on its own filename sorting after everything else.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("target", [1.0, 0.95, 0.86, 0.85, 0.84, 0.5, 0.0, -0.5])
async def test_vector_at_cosine_produces_the_requested_angle(target: float) -> None:
    reference = _unit_norm_vector(seed=REFERENCE_SEED)
    probe = vector_at_cosine(reference, target)

    assert np.linalg.norm(probe) == pytest.approx(1.0, abs=1e-5), "probe must be unit norm"
    actual = float(np.dot(probe.astype(np.float64), reference.astype(np.float64)))
    assert actual == pytest.approx(target, abs=1e-5)


@pytest.mark.asyncio
async def test_independent_random_vectors_are_near_orthogonal() -> None:
    """Documents why seed-picking cannot produce a controlled similarity.

    Two independent random unit vectors in 512-D concentrate near cosine 0, so choosing a
    different seed only ever tests "obviously different" — never the threshold neighbourhood.
    """
    a = _unit_norm_vector(seed=REFERENCE_SEED)
    b = _unit_norm_vector(seed=OTHER_STUDENT_SEED)
    cosine = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    assert abs(cosine) < 0.2, f"expected near-orthogonal, got {cosine}"


# ---------------------------------------------------------------------------
# Batch path (raw SQL) — this is what production calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("target_cosine,should_match", _THRESHOLD_CASES)
async def test_batch_path_applies_threshold(
    test_engine: AsyncEngine,
    target_cosine: float,
    should_match: bool,
) -> None:
    from app.services.pipeline.matching import _resolve_vector_matches

    reference = _unit_norm_vector(seed=REFERENCE_SEED)
    student_id, embedding_id = await _seed_student_with_template(
        test_engine, reference, label="batch"
    )

    probe = vector_at_cosine(reference, target_cosine)
    matches = await _resolve_vector_matches([probe])

    assert len(matches) == 1
    match = matches[0]

    assert match.cosine_similarity == pytest.approx(target_cosine, abs=1e-3), (
        f"reported similarity {match.cosine_similarity} does not match the constructed "
        f"angle {target_cosine} — check the 1 - (embedding <=> vec) conversion"
    )

    if should_match:
        assert match.student_id == student_id, (
            f"cosine {target_cosine} is above the 0.85 threshold and must identify the "
            f"student, got student_id={match.student_id}"
        )
    else:
        assert match.student_id is None, (
            f"cosine {target_cosine} is below the 0.85 threshold and MUST NOT identify a "
            f"student — this is a false-positive identification"
        )
        # The nearest row is still reported even when rejected, for observability.
        assert match.embedding_id == embedding_id


# ---------------------------------------------------------------------------
# ORM path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("target_cosine,should_match", _THRESHOLD_CASES)
async def test_orm_path_applies_threshold(
    test_engine: AsyncEngine,
    target_cosine: float,
    should_match: bool,
) -> None:
    from app.services.pipeline.matching import _resolve_nearest_embedding_match

    reference = _unit_norm_vector(seed=REFERENCE_SEED)
    student_id, embedding_id = await _seed_student_with_template(
        test_engine, reference, label="orm"
    )

    probe = vector_at_cosine(reference, target_cosine)
    async with _session_factory_for(test_engine)() as session:
        match = await _resolve_nearest_embedding_match(session, probe)

    assert match.cosine_similarity == pytest.approx(target_cosine, abs=1e-3)

    if should_match:
        assert match.student_id == student_id
    else:
        assert match.student_id is None, (
            f"cosine {target_cosine} is below threshold and MUST NOT identify a student"
        )
        assert match.embedding_id == embedding_id


# ---------------------------------------------------------------------------
# The two implementations must not drift apart (ATT-077)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("target_cosine,should_match", _THRESHOLD_CASES)
async def test_both_paths_agree(
    test_engine: AsyncEngine,
    target_cosine: float,
    should_match: bool,
) -> None:
    from app.services.pipeline.matching import (
        _resolve_nearest_embedding_match,
        _resolve_vector_matches,
    )

    reference = _unit_norm_vector(seed=REFERENCE_SEED)
    await _seed_student_with_template(test_engine, reference, label="agree")
    probe = vector_at_cosine(reference, target_cosine)

    batch_match = (await _resolve_vector_matches([probe]))[0]
    async with _session_factory_for(test_engine)() as session:
        orm_match = await _resolve_nearest_embedding_match(session, probe)

    assert batch_match.student_id == orm_match.student_id, (
        f"batch and ORM paths disagree on identity at cosine {target_cosine}: "
        f"{batch_match.student_id} vs {orm_match.student_id}"
    )
    assert batch_match.embedding_id == orm_match.embedding_id
    assert batch_match.cosine_similarity == pytest.approx(
        orm_match.cosine_similarity, abs=1e-5
    )


# ---------------------------------------------------------------------------
# Sign of the reported value
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reported_value_is_similarity_not_distance(test_engine: AsyncEngine) -> None:
    """A near-identical face must report ~1.0, not ~0.0.

    pgvector's ``<=>`` returns cosine DISTANCE. Returning it unconverted — or converting
    twice — would invert the whole acceptance rule while still producing a plausible float.
    """
    from app.services.pipeline.matching import _resolve_vector_matches

    reference = _unit_norm_vector(seed=REFERENCE_SEED)
    await _seed_student_with_template(test_engine, reference, label="sign")

    near = (await _resolve_vector_matches([vector_at_cosine(reference, 0.99)]))[0]
    far = (await _resolve_vector_matches([vector_at_cosine(reference, 0.10)]))[0]

    assert near.cosine_similarity == pytest.approx(0.99, abs=1e-3)
    assert far.cosine_similarity == pytest.approx(0.10, abs=1e-3)
    assert near.cosine_similarity > far.cosine_similarity, (
        "a closer face must report a HIGHER value; if this fails the code is reporting "
        "distance where similarity is expected"
    )


# ---------------------------------------------------------------------------
# Batch alignment — the batch path maps results back by rank
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_attributes_each_probe_to_the_right_student(
    test_engine: AsyncEngine,
) -> None:
    """Multi-probe batches must not cross-attribute identities.

    ``_resolve_vector_matches`` reassembles rows via ``WITH ORDINALITY`` and a rank->match
    dict. An off-by-one or an ordering assumption there would attribute one student's
    identity to another student's face — silently, and only in batches.
    """
    from app.services.pipeline.matching import _resolve_vector_matches

    reference_a = _unit_norm_vector(seed=REFERENCE_SEED)
    reference_b = _unit_norm_vector(seed=OTHER_STUDENT_SEED)

    student_a, _ = await _seed_student_with_template(test_engine, reference_a, label="a")
    student_b, _ = await _seed_student_with_template(test_engine, reference_b, label="b")

    probes = [
        vector_at_cosine(reference_a, 0.95, seed=11),   # -> student A
        vector_at_cosine(reference_b, 0.93, seed=12),   # -> student B
        vector_at_cosine(reference_a, 0.20, seed=13),   # -> rejected
        vector_at_cosine(reference_b, 0.97, seed=14),   # -> student B again
    ]

    matches = await _resolve_vector_matches(probes)
    assert len(matches) == len(probes)

    assert matches[0].student_id == student_a, "probe 0 was built against student A"
    assert matches[1].student_id == student_b, "probe 1 was built against student B"
    assert matches[2].student_id is None, "probe 2 is far from both and must be rejected"
    assert matches[3].student_id == student_b, "probe 3 was built against student B"


@pytest.mark.asyncio
async def test_empty_batch_returns_empty(test_engine: AsyncEngine) -> None:
    from app.services.pipeline.matching import _resolve_vector_matches

    assert await _resolve_vector_matches([]) == []


@pytest.mark.asyncio
async def test_no_active_template_yields_no_match(test_engine: AsyncEngine) -> None:
    """With nothing enrolled, a probe must be rejected rather than matched to nothing."""
    from app.services.pipeline.matching import _resolve_vector_matches

    probe = vector_at_cosine(_unit_norm_vector(seed=REFERENCE_SEED), 1.0)
    match = (await _resolve_vector_matches([probe]))[0]

    assert match.student_id is None
    assert match.embedding_id is None
    assert match.cosine_similarity is None


# ---------------------------------------------------------------------------
# ATT-048 — per-query HNSW ef_search is set on both code paths.
#
# pgvector's default `hnsw.ef_search = 40` has been measured to drop recall
# below STRICT_SIMILARITY_THRESHOLD (0.85) on 10k+ students with multiple
# poses, returning a 0.86 true-NN as "no match" — surfacing as an unknown-
# face Sighting. The fix in `matching.py` is to emit
# `SET LOCAL hnsw.ef_search = HNSW_EF_SEARCH (100)` inside the matching
# transaction before the SELECT. We can't measure recall in this
# environment (no 50k-embedding labeled subset), so we test that:
#   1. The module exposes a HNSW_EF_SEARCH constant that's higher than the
#      pgvector default (40).
#   2. The matching.py source emits `SET LOCAL hnsw.ef_search` on BOTH
#      code paths (batch path and ORM path). Source-scan is proxy-immune
#      to whatever Celery/SQLAlchemy settings are emitted at module load.
#   3. The render of `SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}` (f-string
#      interpolation with a constant integer) doesn't have a SQLi surface —
#      the value is a module-level literal int, not a user-controlled
#      string.
# ---------------------------------------------------------------------------


def test_att_048_hnsw_ef_search_constant_is_stricter_than_pgvector_default() -> None:
    """ATT-048: HNSW_EF_SEARCH must be greater than pgvector's default of 40.

    The whole point of FIX (b) is to crank the per-query candidate list wider
    than the default to recover recall on 50k+ embedding corpuses; if a
    future maintainer drops this back to <=40, ATT-048 recurs under load.
    """
    from app.services.pipeline.matching import HNSW_EF_SEARCH

    assert isinstance(HNSW_EF_SEARCH, int), (
        "ATT-048: HNSW_EF_SEARCH must be an int (the constant is interpolated "
        "into a `SET LOCAL` string; a non-int would either be caught at string-"
        "format time or introduce a SQLi surface)."
    )
    assert HNSW_EF_SEARCH > 40, (
        f"ATT-048: HNSW_EF_SEARCH={HNSW_EF_SEARCH} is not stricter than "
        f"pgvector's default 40; this re-introduces the recall gap on large "
        f"embedding corpuses."
    )


def test_att_048_set_local_emitted_in_batch_path() -> None:
    """Source scan: the batch (`_resolve_vector_matches`) code path emits
    `SET LOCAL hnsw.ef_search` before the SELECT.

    Pre-fix `_BATCH_NEAREST_MATCH_SQL` had no SET LOCAL; ef_search was the
    pgvector default 40 (table-level or session-level). Post-fix the batch
    path opens an explicit `async with session.begin():` block and emits
    `SET LOCAL hnsw.ef_search = 100` before the SELECT — pinning the
    per-query candidate list for the duration of the matching transaction
    only.
    """
    import inspect
    import textwrap

    from app.services.pipeline import matching as matching_module

    src = textwrap.dedent(inspect.getsource(matching_module._resolve_vector_matches))
    assert "SET LOCAL hnsw.ef_search" in src, (
        "ATT-048: _resolve_vector_matches must emit `SET LOCAL hnsw.ef_search` "
        "before the SELECT, otherwise pgvector uses its default ef_search=40 "
        "and recall drops below STRICT_SIMILARITY_THRESHOLD (0.85) on 50k+ "
        "embedding corpuses."
    )
    # The SET LOCAL needs an enclosing transaction, so the batch path must
    # explicitly open one (AsyncSession doesn't auto-BEGIN for raw SET
    # without a session.begin()).
    assert "session.begin()" in src, (
        "ATT-048: _resolve_vector_matches must wrap the SELECT in an explicit "
        "`async with session.begin():` block so `SET LOCAL hnsw.ef_search` "
        "applies to this transaction only. Without it the SET may either be "
        "ignored (if no transaction is open yet) or leak to subsequent "
        "unrelated sessions on the pool."
    )


def test_att_048_set_local_emitted_in_orm_path() -> None:
    """Source scan: the ORM (`_resolve_nearest_embedding_match`) code path
    emits `SET LOCAL hnsw.ef_search` before the SELECT.

    The ORM path is called from the orchestrator with the orchestrator's
    already-open session; the SET LOCAL applies to that session's
    transaction context (the orchestrator owns the BEGIN/COMMIT).
    """
    import inspect
    import textwrap

    from app.services.pipeline import matching as matching_module

    src = textwrap.dedent(
        inspect.getsource(matching_module._resolve_nearest_embedding_match)
    )
    assert "SET LOCAL hnsw.ef_search" in src, (
        "ATT-048: _resolve_nearest_embedding_match must emit `SET LOCAL "
        "hnsw.ef_search` before the SELECT on the same session — the "
        "production orchestrator passes its own already-running session, "
        "so the SET applies to the orchestrator's open transaction."
    )


@pytest.mark.asyncio
async def test_att_048_batch_match_query_runs_without_error_under_set_local(
    test_engine: AsyncEngine,
) -> None:
    """Smoke anchor: the batch path SQL (with the new SET LOCAL) executes
    without error against the existing pgvector index. The migration
    `20260519_0004_pgvector_hnsw_index.py` already defines the HNSW index
    with `WITH (m=16, ef_construction=128)`, so `SET LOCAL hnsw.ef_search
    = 100` is a valid per-query override against that index.

    Pre-fix this test passed because the SQL was a simpler SELECT. Post-fix
    the SQL is "SET LOCAL + SELECT" inside `session.begin()`; if any
    SET LOCAL syntax is malformed or the index isn't HNSW-compatible, this
    smoke test surfaces the failure on the same path the production code
    runs.
    """
    from app.services.pipeline.matching import _resolve_vector_matches

    reference = _unit_norm_vector(seed=REFERENCE_SEED)
    student_id, _ = await _seed_student_with_template(
        test_engine, reference, label="att048"
    )

    probe = vector_at_cosine(reference, 1.0)
    # This call exercises the SET LOCAL emission path. Pre-fix the SQL
    # did not include SET LOCAL; post-fix it does. The assertion is
    # deliberately the same as test_batch_path_applies_threshold — the
    # smoke value here is "the new SQL executes without error AND produces
    # the right answer".
    matches = await _resolve_vector_matches([probe])
    assert len(matches) == 1
    assert matches[0].student_id == student_id, (
        f"ATT-048 sanity: post-ef_search change, the matched student must "
        f"still resolve correctly — got student_id={matches[0].student_id}"
    )
