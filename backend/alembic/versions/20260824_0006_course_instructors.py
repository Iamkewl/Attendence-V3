"""Add course_instructors association table for course-scoped authz.

Revision ID: 20260824_0006
Revises: 20260824_0005
Create Date: 2026-08-24 10:00:00

ATT-016 / decision D8 (Option A): M:N link between courses and users with a
CHECK-constrained ``role_in_course`` ('owner','ta') instead of extending the
native ``user_role`` enum — ALTER TYPE ... ADD VALUE cannot run inside the
transactional migration and breaks the CI downgrade round-trip. Constraint
and index names are explicit so ORM metadata and DB agree and future
``alembic revision --autogenerate`` runs emit empty diffs.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "20260824_0006"
down_revision: str | None = "20260824_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create course_instructors with CASCADE FKs and role CHECK."""
    op.create_table(
        "course_instructors",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("course_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "role_in_course",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'owner'"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_course_instructors"),
        sa.UniqueConstraint("course_id", "user_id", name="uq_course_instructors_course_user"),
        sa.CheckConstraint(
            "role_in_course IN ('owner', 'ta')",
            name="course_instructor_role_valid",
        ),
        sa.ForeignKeyConstraint(
            ["course_id"],
            ["courses.id"],
            ondelete="CASCADE",
            name="fk_course_instructors_course_id_courses",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE",
            name="fk_course_instructors_user_id_users",
        ),
    )
    op.create_index("ix_course_instructors_user_id", "course_instructors", ["user_id"])


def downgrade() -> None:
    """Drop the association table; constraints and indexes go with it."""
    op.drop_index("ix_course_instructors_user_id", table_name="course_instructors")
    op.drop_table("course_instructors")
