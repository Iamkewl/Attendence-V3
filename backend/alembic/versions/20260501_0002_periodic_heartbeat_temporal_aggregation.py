"""Refactor attendance persistence for periodic heartbeat and temporal aggregation.

Revision ID: 20260501_0002
Revises: 20260501_0001
Create Date: 2026-05-01 23:05:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "20260501_0002"
down_revision: str | None = "20260501_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


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
    """Migrate turnstile attendance events to class sessions plus raw heartbeat sightings."""
    op.rename_table("attendance_records", "class_session_records")

    op.drop_constraint(
        "fk_governance_logs_attendance_record_id_attendance_records",
        "governance_logs",
        type_="foreignkey",
    )
    op.drop_index("ix_governance_attendance_record_id", table_name="governance_logs")
    op.alter_column(
        "governance_logs",
        "attendance_record_id",
        new_column_name="class_session_record_id",
        existing_type=postgresql.UUID(as_uuid=True),
    )
    op.create_foreign_key(
        "fk_governance_logs_class_session_record_id_class_session_records",
        "governance_logs",
        "class_session_records",
        ["class_session_record_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_governance_class_session_record_id",
        "governance_logs",
        ["class_session_record_id"],
        unique=False,
    )

    op.drop_index("ix_attendance_course_started_at", table_name="class_session_records")
    op.drop_index("ix_attendance_status_started_at", table_name="class_session_records")
    op.drop_index("ix_attendance_student_started_at", table_name="class_session_records")

    op.drop_constraint(
        "uq_attendance_student_course_started",
        "class_session_records",
        type_="unique",
    )
    op.drop_constraint(
        "attendance_session_time_order",
        "class_session_records",
        type_="check",
    )

    op.add_column(
        "class_session_records",
        sa.Column("session_date", sa.Date(), nullable=True),
    )
    op.add_column(
        "class_session_records",
        sa.Column("sighting_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "class_session_records",
        sa.Column(
            "required_sightings_threshold",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )
    op.add_column(
        "class_session_records",
        sa.Column(
            "evaluated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )

    op.execute(
        """
        UPDATE class_session_records
        SET session_date = (session_started_at AT TIME ZONE 'UTC')::date
        WHERE session_started_at IS NOT NULL
        """
    )
    op.alter_column("class_session_records", "session_date", nullable=False)

    op.drop_constraint(
        "fk_attendance_records_recorded_by_user_id_users",
        "class_session_records",
        type_="foreignkey",
    )
    op.drop_column("class_session_records", "recorded_by_user_id")
    op.drop_column("class_session_records", "source")
    op.drop_column("class_session_records", "session_started_at")
    op.drop_column("class_session_records", "session_ended_at")
    op.drop_column("class_session_records", "recorded_at")

    op.alter_column(
        "class_session_records",
        "status",
        existing_type=attendance_status_enum,
        server_default=sa.text("'absent'::attendance_status"),
    )

    op.create_check_constraint(
        "class_session_sighting_count_non_negative",
        "class_session_records",
        "sighting_count >= 0",
    )
    op.create_check_constraint(
        "class_session_threshold_positive",
        "class_session_records",
        "required_sightings_threshold >= 1",
    )

    op.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY student_id, course_id, session_date
                    ORDER BY updated_at DESC, created_at DESC, id DESC
                ) AS row_rank
            FROM class_session_records
        )
        DELETE FROM class_session_records target
        USING ranked
        WHERE target.id = ranked.id
          AND ranked.row_rank > 1
        """
    )

    op.create_unique_constraint(
        "uq_class_session_student_course_date",
        "class_session_records",
        ["student_id", "course_id", "session_date"],
    )

    op.create_index(
        "ix_class_session_course_date",
        "class_session_records",
        ["course_id", "session_date"],
        unique=False,
    )
    op.create_index(
        "ix_class_session_student_date",
        "class_session_records",
        ["student_id", "session_date"],
        unique=False,
    )
    op.create_index(
        "ix_class_session_status_date",
        "class_session_records",
        ["status", "session_date"],
        unique=False,
    )

    op.create_table(
        "sightings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("course_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("room_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("camera_id", sa.String(length=64), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("embedding_reference", sa.String(length=255), nullable=True),
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
            "char_length(btrim(camera_id)) > 0",
            name="sightings_camera_id_not_blank",
        ),
        sa.CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)",
            name="sightings_confidence_range",
        ),
        sa.ForeignKeyConstraint(
            ["course_id"],
            ["courses.id"],
            name="fk_sightings_course_id_courses",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["room_id"],
            ["rooms.id"],
            name="fk_sightings_room_id_rooms",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            ["students.id"],
            name="fk_sightings_student_id_students",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_sightings"),
    )
    op.create_index(
        "ix_sightings_course_timestamp",
        "sightings",
        ["course_id", "timestamp"],
        unique=False,
    )
    op.create_index(
        "ix_sightings_student_timestamp",
        "sightings",
        ["student_id", "timestamp"],
        unique=False,
    )
    op.create_index(
        "ix_sightings_camera_timestamp",
        "sightings",
        ["camera_id", "timestamp"],
        unique=False,
    )


def downgrade() -> None:
    """Revert heartbeat and class-session tables back to turnstile attendance records."""
    op.drop_index("ix_sightings_camera_timestamp", table_name="sightings")
    op.drop_index("ix_sightings_student_timestamp", table_name="sightings")
    op.drop_index("ix_sightings_course_timestamp", table_name="sightings")
    op.drop_table("sightings")

    op.drop_index("ix_class_session_status_date", table_name="class_session_records")
    op.drop_index("ix_class_session_student_date", table_name="class_session_records")
    op.drop_index("ix_class_session_course_date", table_name="class_session_records")

    op.drop_constraint(
        "uq_class_session_student_course_date",
        "class_session_records",
        type_="unique",
    )
    op.drop_constraint(
        "class_session_threshold_positive",
        "class_session_records",
        type_="check",
    )
    op.drop_constraint(
        "class_session_sighting_count_non_negative",
        "class_session_records",
        type_="check",
    )

    op.drop_constraint(
        "fk_governance_logs_class_session_record_id_class_session_records",
        "governance_logs",
        type_="foreignkey",
    )
    op.drop_index("ix_governance_class_session_record_id", table_name="governance_logs")
    op.alter_column(
        "governance_logs",
        "class_session_record_id",
        new_column_name="attendance_record_id",
        existing_type=postgresql.UUID(as_uuid=True),
    )

    op.add_column(
        "class_session_records",
        sa.Column("recorded_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "class_session_records",
        sa.Column(
            "source",
            attendance_source_enum,
            nullable=False,
            server_default=sa.text("'manual'::attendance_source"),
        ),
    )
    op.add_column(
        "class_session_records",
        sa.Column("session_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "class_session_records",
        sa.Column("session_ended_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "class_session_records",
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )

    op.execute(
        """
        UPDATE class_session_records
        SET session_started_at = session_date::timestamp AT TIME ZONE 'UTC'
        WHERE session_date IS NOT NULL
        """
    )
    op.alter_column("class_session_records", "session_started_at", nullable=False)

    op.drop_column("class_session_records", "evaluated_at")
    op.drop_column("class_session_records", "required_sightings_threshold")
    op.drop_column("class_session_records", "sighting_count")
    op.drop_column("class_session_records", "session_date")

    op.alter_column(
        "class_session_records",
        "status",
        existing_type=attendance_status_enum,
        server_default=sa.text("'present'::attendance_status"),
    )

    op.create_check_constraint(
        "attendance_session_time_order",
        "class_session_records",
        "session_ended_at IS NULL OR session_ended_at >= session_started_at",
    )
    op.create_unique_constraint(
        "uq_attendance_student_course_started",
        "class_session_records",
        ["student_id", "course_id", "session_started_at"],
    )
    op.create_index(
        "ix_attendance_course_started_at",
        "class_session_records",
        ["course_id", "session_started_at"],
        unique=False,
    )
    op.create_index(
        "ix_attendance_status_started_at",
        "class_session_records",
        ["status", "session_started_at"],
        unique=False,
    )
    op.create_index(
        "ix_attendance_student_started_at",
        "class_session_records",
        ["student_id", "session_started_at"],
        unique=False,
    )

    op.rename_table("class_session_records", "attendance_records")

    op.create_foreign_key(
        "fk_attendance_records_recorded_by_user_id_users",
        "attendance_records",
        "users",
        ["recorded_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_foreign_key(
        "fk_governance_logs_attendance_record_id_attendance_records",
        "governance_logs",
        "attendance_records",
        ["attendance_record_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_governance_attendance_record_id",
        "governance_logs",
        ["attendance_record_id"],
        unique=False,
    )
