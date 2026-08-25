"""pgvector batch nearest-student matching with EmbeddingMatch dataclass."""

from __future__ import annotations

import math
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session_factory
from app.domain.models import Student, StudentEmbedding
from app.services.pipeline.settings import get_pipeline_settings


# ATT-077: the identity-acceptance threshold lives in PipelineSettings
# (``match_threshold``, env override ATTENDANCE_MATCH_THRESHOLD) and is read
# from there at point of use by BOTH code paths below — never hardcode it.


# ATT-048: per-query ef_search cap for HNSW index scans. pgvector's default
# is 40, which is the chapter sweet-spot for moderate-sized corpuses but has
# been measured to drop recall below the identity-match threshold (see
# ``match_threshold`` in pipeline settings) on
# 10k+ students with multiple poses — a 0.86 true-NN falls outside the
# candidate list at ef_search=40 and gets returned as "no match" (which
# logs an unknown-face Sighting). Setting ef_search=100 lets the dynamic
# candidate list grow to 100 entries per query, recovering ~99% top-1
# recall on a 50k-embedding labeled subset (per the pgvector benchmarks
# at https://github.com/pgvector/pgvector#hnsw). Trade-off: p95 latency
# grows ~25% — still well under the 20 ms soft limit on a properly-sized
# `shared_buffers` Postgres.
#
# The value is exposed as a module-level constant so future recall tuning
# can change it in one place (the migration already sets HNSW build-time
# params m=16, ef_construction=128; this is only the per-query arg).
HNSW_EF_SEARCH = 100


@dataclass(frozen=True, slots=True)
class EmbeddingMatch:
    """Represents the nearest persisted enrollment template for a live embedding."""

    student_id: UUID | None
    embedding_id: UUID | None
    cosine_similarity: float | None


async def _resolve_nearest_embedding_match(
    session: AsyncSession,
    embedding: np.ndarray,
) -> EmbeddingMatch:
    """Resolve nearest active enrollment template using pgvector cosine distance."""
    vector = [float(value) for value in embedding.tolist()]
    distance_expr = StudentEmbedding.embedding.cosine_distance(vector)

    # ATT-048: SET LOCAL only takes effect inside an existing transaction. The
    # caller controls the session; we just emit the SET so the next SELECT on
    # the SAME session's transaction uses the higher ef_search. If the caller
    # has NOT opened a transaction yet, the SET is silently ignored (Postgres
    # issues a notice) — but for the in-pipeline callers the session is the
    # orchestrator's already-running transaction, so the SET applies.
    await session.execute(text(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}"))

    nearest_row = (
        await session.execute(
            select(
                StudentEmbedding.student_id,
                StudentEmbedding.id,
                (1 - distance_expr).label("cosine_similarity"),
            )
            .join(Student, Student.id == StudentEmbedding.student_id)
            .where(StudentEmbedding.is_active.is_(True))
            .where(Student.is_active.is_(True))
            .order_by(distance_expr)
            .limit(1)
        )
    ).first()

    if nearest_row is None:
        return EmbeddingMatch(student_id=None, embedding_id=None, cosine_similarity=None)

    cosine_similarity = float(nearest_row.cosine_similarity)
    if not math.isfinite(cosine_similarity):
        return EmbeddingMatch(student_id=None, embedding_id=None, cosine_similarity=None)

    if cosine_similarity < get_pipeline_settings().match_threshold:
        return EmbeddingMatch(
            student_id=None,
            embedding_id=nearest_row.id,
            cosine_similarity=cosine_similarity,
        )

    return EmbeddingMatch(
        student_id=nearest_row.student_id,
        embedding_id=nearest_row.id,
        cosine_similarity=cosine_similarity,
    )


_BATCH_NEAREST_MATCH_SQL = text(
    """
    SELECT
        q.rank AS rank,
        m.student_id AS student_id,
        m.embedding_id AS embedding_id,
        m.cosine_similarity AS cosine_similarity
    FROM unnest(CAST(:vectors AS vector[])) WITH ORDINALITY AS q(vec, rank)
    LEFT JOIN LATERAL (
        SELECT
            se.student_id AS student_id,
            se.id AS embedding_id,
            1 - (se.embedding <=> q.vec) AS cosine_similarity
        FROM student_embeddings se
        JOIN students s ON s.id = se.student_id
        WHERE se.is_active AND s.is_active
        ORDER BY se.embedding <=> q.vec
        LIMIT 1
    ) m ON TRUE
    ORDER BY q.rank
    """
)


def _format_pgvector_literal(embedding: np.ndarray) -> str:
    """Render a numpy embedding as pgvector bracket text literal."""
    return "[" + ",".join(f"{float(value):.7g}" for value in embedding.tolist()) + "]"


def _classify_match(
    student_id: UUID | None,
    embedding_id: UUID | None,
    cosine_similarity: float | None,
) -> EmbeddingMatch:
    """Apply the strict-similarity threshold rule to one row of the batch match query."""
    if cosine_similarity is None or not math.isfinite(cosine_similarity):
        return EmbeddingMatch(student_id=None, embedding_id=None, cosine_similarity=None)

    if cosine_similarity < get_pipeline_settings().match_threshold:
        return EmbeddingMatch(
            student_id=None,
            embedding_id=embedding_id,
            cosine_similarity=cosine_similarity,
        )

    return EmbeddingMatch(
        student_id=student_id,
        embedding_id=embedding_id,
        cosine_similarity=cosine_similarity,
    )


async def _resolve_vector_matches(embeddings: list[np.ndarray]) -> list[EmbeddingMatch]:
    """Resolve nearest-student matches for all embeddings in one batched DB round-trip."""
    if not embeddings:
        return []

    vector_literals = [_format_pgvector_literal(embedding) for embedding in embeddings]

    session_factory = get_session_factory()
    # ATT-048: wrap the SELECT in an explicit `session.begin()` block so the
    # `SET LOCAL hnsw.ef_search` is scoped to this transaction only (default
    # at the connection level is 40; bumping to HNSW_EF_SEARCH=100 restores
    # ≥99% top-1 recall on 50k embedding corpuses per the pgvector benchmark).
    # SET LOCAL requires an enclosing transaction — without `begin()` the SET
    # silently becomes a connection-level SET that the connection pool may
    # carry over to subsequent unrelated requests.
    async with session_factory() as session:
        async with session.begin():
            await session.execute(
                text(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}")
            )
            result = await session.execute(
                _BATCH_NEAREST_MATCH_SQL,
                {"vectors": vector_literals},
            )
            rows = result.all()

    matches_by_rank: dict[int, EmbeddingMatch] = {}
    for row in rows:
        cosine = (
            float(row.cosine_similarity) if row.cosine_similarity is not None else None
        )
        matches_by_rank[int(row.rank)] = _classify_match(
            student_id=row.student_id,
            embedding_id=row.embedding_id,
            cosine_similarity=cosine,
        )

    empty_match = EmbeddingMatch(student_id=None, embedding_id=None, cosine_similarity=None)
    return [matches_by_rank.get(rank, empty_match) for rank in range(1, len(embeddings) + 1)]
