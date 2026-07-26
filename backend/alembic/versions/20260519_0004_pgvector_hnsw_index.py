"""Replace pgvector ivfflat index on student_embeddings with HNSW.

Revision ID: 20260519_0004
Revises: 20260503_0003
Create Date: 2026-05-19 10:00:00

HNSW has no minimum-rows requirement (ivfflat needs a warm corpus to produce
useful clusters), tolerates green-field deployments, and supports incremental
build as enrollments grow. Cosine ops match the existing query operator.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "20260519_0004"
down_revision: str | None = "20260503_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Swap the ivfflat embedding index for an HNSW cosine-ops index."""
    op.execute("DROP INDEX IF EXISTS ix_student_embeddings_embedding_cosine;")
    op.execute(
        "CREATE INDEX ix_student_embeddings_embedding_cosine "
        "ON student_embeddings USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 128);"
    )


def downgrade() -> None:
    """Restore the original ivfflat embedding index."""
    op.execute("DROP INDEX IF EXISTS ix_student_embeddings_embedding_cosine;")
    op.execute(
        "CREATE INDEX ix_student_embeddings_embedding_cosine "
        "ON student_embeddings USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100);"
    )
    op.drop_index("ix_does_not_exist", table_name="student_embeddings")
