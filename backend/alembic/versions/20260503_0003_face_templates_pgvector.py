"""Add multi-pose student embeddings and template audit logs with pgvector search support.

Revision ID: 20260503_0003
Revises: 20260501_0002
Create Date: 2026-05-03 10:40:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover - compatibility fallback for older pgvector builds.
    from pgvector.sqlalchemy import VECTOR as Vector


# revision identifiers, used by Alembic.
revision: str = "20260503_0003"
down_revision: str | None = "20260501_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create pgvector extension, enrollment template tables, and unknown-sighting support."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.create_table(
        "student_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("embedding", Vector(512), nullable=False),
        sa.Column("pose_label", sa.String(length=32), nullable=False, server_default=sa.text("'front'")),
        sa.Column("quality_score", sa.Float(), nullable=False, server_default=sa.text("1.0")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "char_length(btrim(pose_label)) > 0",
            name="student_embeddings_pose_label_not_blank",
        ),
        sa.CheckConstraint(
            "quality_score >= 0 AND quality_score <= 1",
            name="student_embeddings_quality_score_range",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            ["students.id"],
            name="fk_student_embeddings_student_id_students",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_student_embeddings"),
    )
    op.create_index(
        "ix_student_embeddings_student_pose_active",
        "student_embeddings",
        ["student_id", "pose_label", "is_active"],
        unique=False,
    )
    op.execute(
        "CREATE INDEX ix_student_embeddings_embedding_cosine "
        "ON student_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
    )

    op.create_table(
        "template_audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_embedding_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("pose_label", sa.String(length=32), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column(
            "event_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "char_length(btrim(action)) > 0",
            name="template_audit_logs_action_not_blank",
        ),
        sa.ForeignKeyConstraint(
            ["student_embedding_id"],
            ["student_embeddings.id"],
            name="fk_template_audit_logs_student_embedding_id_student_embeddings",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            ["students.id"],
            name="fk_template_audit_logs_student_id_students",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_template_audit_logs"),
    )
    op.create_index(
        "ix_template_audit_logs_student_created_at",
        "template_audit_logs",
        ["student_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_template_audit_logs_embedding_created_at",
        "template_audit_logs",
        ["student_embedding_id", "created_at"],
        unique=False,
    )

    op.drop_constraint("fk_sightings_student_id_students", "sightings", type_="foreignkey")
    op.alter_column(
        "sightings",
        "student_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )
    op.create_foreign_key(
        "fk_sightings_student_id_students",
        "sightings",
        "students",
        ["student_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Drop pgvector template tables and restore strict sighting student requirement."""
    op.drop_constraint("fk_sightings_student_id_students", "sightings", type_="foreignkey")
    op.execute("DELETE FROM sightings WHERE student_id IS NULL;")
    op.alter_column(
        "sightings",
        "student_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
    op.create_foreign_key(
        "fk_sightings_student_id_students",
        "sightings",
        "students",
        ["student_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.drop_index("ix_template_audit_logs_embedding_created_at", table_name="template_audit_logs")
    op.drop_index("ix_template_audit_logs_student_created_at", table_name="template_audit_logs")
    op.drop_table("template_audit_logs")

    op.execute("DROP INDEX IF EXISTS ix_student_embeddings_embedding_cosine;")
    op.drop_index("ix_student_embeddings_student_pose_active", table_name="student_embeddings")
    op.drop_table("student_embeddings")
