"""Add nullable template_version to student_embeddings.

Revision ID: 20260824_0005
Revises: 20260519_0004
Create Date: 2026-08-24 10:00:00

Decision D13 (DECISIONS-2026-08-24): store, per face template, the version
of the embedding pipeline that produced it so a future model change can
re-embed populations selectively instead of wholesale. Nullable because
rows written before this column existed (API/kiosk enrollments) carry no
version provenance and stay NULL until re-enrolled.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260824_0005"
down_revision: str | None = "20260519_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable integer template_version column."""
    op.add_column(
        "student_embeddings",
        sa.Column("template_version", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    """Drop template_version (round-trip safe per CI migration job)."""
    op.drop_column("student_embeddings", "template_version")
