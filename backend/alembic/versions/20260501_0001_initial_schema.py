"""Initial attendance database schema.

Revision ID: 20260501_0001
Revises:
Create Date: 2026-05-01 19:40:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "20260501_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


user_role_enum = postgresql.ENUM(
    "admin",
    "instructor",
    "auditor",
    "operator",
    name="user_role",
    create_type=False,
)
attendance_status_enum = postgresql.ENUM(
    "present",
    "absent",
    "late",
    "excused",
    name="attendance_status",
    create_type=False,
)
attendance_source_enum = postgresql.ENUM(
    "manual",
    "qr_code",
    "biometric",
    "imported",
    name="attendance_source",
    create_type=False,
)


def upgrade() -> None:
    """Create all domain tables, constraints, indexes, and enum types."""
    bind = op.get_bind()
    user_role_enum.create(bind, checkfirst=True)
    attendance_status_enum.create(bind, checkfirst=True)
    attendance_source_enum.create(bind, checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("full_name", sa.String(length=120), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column(
            "role",
            user_role_enum,
            nullable=False,
            server_default=sa.text("'instructor'::user_role"),
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint("char_length(btrim(full_name)) >= 2", name="users_full_name_min_length"),
        sa.CheckConstraint(
            "char_length(password_hash) >= 60",
            name="users_password_hash_min_length",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_role_active", "users", ["role", "is_active"], unique=False)

    op.create_table(
        "students",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_number", sa.String(length=32), nullable=False),
        sa.Column("program", sa.String(length=120), nullable=False),
        sa.Column("enrollment_year", sa.SmallInteger(), nullable=False),
        sa.Column("graduation_year", sa.SmallInteger(), nullable=True),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
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
        sa.CheckConstraint(
            "enrollment_year BETWEEN 2000 AND 2100",
            name="students_enrollment_year_range",
        ),
        sa.CheckConstraint(
            "graduation_year IS NULL OR graduation_year >= enrollment_year",
            name="students_graduation_year_valid",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_students_user_id_users", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_students"),
        sa.UniqueConstraint("student_number", name="uq_students_student_number"),
        sa.UniqueConstraint("user_id", name="uq_students_user_id"),
    )
    op.create_index(
        "ix_students_enrollment_active",
        "students",
        ["enrollment_year", "is_active"],
        unique=False,
    )

    op.create_table(
        "courses",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("code", sa.String(length=24), nullable=False),
        sa.Column("title", sa.String(length=160), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("credits", sa.SmallInteger(), nullable=False, server_default=sa.text("3")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
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
        sa.CheckConstraint("credits BETWEEN 1 AND 12", name="courses_credits_range"),
        sa.PrimaryKeyConstraint("id", name="pk_courses"),
        sa.UniqueConstraint("code", name="uq_courses_code"),
    )
    op.create_index("ix_courses_active", "courses", ["is_active"], unique=False)

    op.create_table(
        "rooms",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("code", sa.String(length=24), nullable=False),
        sa.Column("building", sa.String(length=120), nullable=False),
        sa.Column("floor", sa.SmallInteger(), nullable=True),
        sa.Column("capacity", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
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
        sa.CheckConstraint("capacity BETWEEN 1 AND 2000", name="rooms_capacity_range"),
        sa.PrimaryKeyConstraint("id", name="pk_rooms"),
        sa.UniqueConstraint("code", name="uq_rooms_code"),
    )
    op.create_index("ix_rooms_building_active", "rooms", ["building", "is_active"], unique=False)

    op.create_table(
        "attendance_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("course_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("room_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("recorded_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "status",
            attendance_status_enum,
            nullable=False,
            server_default=sa.text("'present'::attendance_status"),
        ),
        sa.Column(
            "source",
            attendance_source_enum,
            nullable=False,
            server_default=sa.text("'manual'::attendance_source"),
        ),
        sa.Column("session_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("session_ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("notes", sa.Text(), nullable=True),
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
        sa.CheckConstraint(
            "session_ended_at IS NULL OR session_ended_at >= session_started_at",
            name="attendance_session_time_order",
        ),
        sa.ForeignKeyConstraint(
            ["course_id"],
            ["courses.id"],
            name="fk_attendance_records_course_id_courses",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["recorded_by_user_id"],
            ["users.id"],
            name="fk_attendance_records_recorded_by_user_id_users",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["room_id"],
            ["rooms.id"],
            name="fk_attendance_records_room_id_rooms",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            ["students.id"],
            name="fk_attendance_records_student_id_students",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_attendance_records"),
        sa.UniqueConstraint(
            "student_id",
            "course_id",
            "session_started_at",
            name="uq_attendance_student_course_started",
        ),
    )
    op.create_index(
        "ix_attendance_course_started_at",
        "attendance_records",
        ["course_id", "session_started_at"],
        unique=False,
    )
    op.create_index(
        "ix_attendance_status_started_at",
        "attendance_records",
        ["status", "session_started_at"],
        unique=False,
    )
    op.create_index(
        "ix_attendance_student_started_at",
        "attendance_records",
        ["student_id", "session_started_at"],
        unique=False,
    )

    op.create_table(
        "governance_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("attendance_record_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("change_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("ip_address", postgresql.INET(), nullable=True),
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
        sa.CheckConstraint(
            "char_length(btrim(action)) > 0",
            name="governance_action_not_blank",
        ),
        sa.CheckConstraint(
            "char_length(btrim(entity_type)) > 0",
            name="governance_entity_type_not_blank",
        ),
        sa.ForeignKeyConstraint(
            ["actor_user_id"],
            ["users.id"],
            name="fk_governance_logs_actor_user_id_users",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["attendance_record_id"],
            ["attendance_records.id"],
            name="fk_governance_logs_attendance_record_id_attendance_records",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_governance_logs"),
    )
    op.create_index(
        "ix_governance_actor_created_at",
        "governance_logs",
        ["actor_user_id", "created_at"],
        unique=False,
    )
    op.create_index("ix_governance_attendance_record_id", "governance_logs", ["attendance_record_id"], unique=False)
    op.create_index("ix_governance_created_at", "governance_logs", ["created_at"], unique=False)
    op.create_index("ix_governance_entity_lookup", "governance_logs", ["entity_type", "entity_id"], unique=False)
    op.create_index("ix_governance_request_id", "governance_logs", ["request_id"], unique=False)


def downgrade() -> None:
    """Drop all domain tables, indexes, constraints, and enum types."""
    op.drop_index("ix_governance_request_id", table_name="governance_logs")
    op.drop_index("ix_governance_entity_lookup", table_name="governance_logs")
    op.drop_index("ix_governance_created_at", table_name="governance_logs")
    op.drop_index("ix_governance_attendance_record_id", table_name="governance_logs")
    op.drop_index("ix_governance_actor_created_at", table_name="governance_logs")
    op.drop_table("governance_logs")

    op.drop_index("ix_attendance_student_started_at", table_name="attendance_records")
    op.drop_index("ix_attendance_status_started_at", table_name="attendance_records")
    op.drop_index("ix_attendance_course_started_at", table_name="attendance_records")
    op.drop_table("attendance_records")

    op.drop_index("ix_rooms_building_active", table_name="rooms")
    op.drop_table("rooms")

    op.drop_index("ix_courses_active", table_name="courses")
    op.drop_table("courses")

    op.drop_index("ix_students_enrollment_active", table_name="students")
    op.drop_table("students")

    op.drop_index("ix_users_role_active", table_name="users")
    op.drop_table("users")

    bind = op.get_bind()
    attendance_source_enum.drop(bind, checkfirst=True)
    attendance_status_enum.drop(bind, checkfirst=True)
    user_role_enum.drop(bind, checkfirst=True)
