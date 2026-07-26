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
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", [1.0, 0.95, 0.86, 0.85, 0.84, 0.5, 0.0, -0.5])
def test_vector_at_cosine_produces_the_requested_angle(target: float) -> None:
    reference = _unit_norm_vector(seed=REFERENCE_SEED)
    probe = vector_at_cosine(reference, target)

    assert np.linalg.norm(probe) == pytest.approx(1.0, abs=1e-5), "probe must be unit norm"
    actual = float(np.dot(probe.astype(np.float64), reference.astype(np.float64)))
    assert actual == pytest.approx(target, abs=1e-5)


def test_independent_random_vectors_are_near_orthogonal() -> None:
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


async def test_empty_batch_returns_empty(test_engine: AsyncEngine) -> None:
    from app.services.pipeline.matching import _resolve_vector_matches

    assert await _resolve_vector_matches([]) == []


async def test_no_active_template_yields_no_match(test_engine: AsyncEngine) -> None:
    """With nothing enrolled, a probe must be rejected rather than matched to nothing."""
    from app.services.pipeline.matching import _resolve_vector_matches

    probe = vector_at_cosine(_unit_norm_vector(seed=REFERENCE_SEED), 1.0)
    match = (await _resolve_vector_matches([probe]))[0]

    assert match.student_id is None
    assert match.embedding_id is None
    assert match.cosine_similarity is None
