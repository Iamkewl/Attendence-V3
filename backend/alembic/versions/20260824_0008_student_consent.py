"""Student biometric consent columns + governance vocabulary extension.

Revision ID: 20260824_0008
Revises: 20260824_0007
Create Date: 2026-08-24 14:00:00

ATT-044 (biometric consent MVP) with ATT-038/ATT-045 vocabulary enablement:

1. ``students.biometric_consent_status`` — String CHECK-constrained domain
   ('pending','granted','denied','withdrawn'), DEFAULT 'pending' NOT NULL so
   pre-existing rows backfill as pending (no consent captured yet = no
   biometric processing allowed once enforcement is enabled).
2. ``students.biometric_consent_at`` — nullable timestamptz recording WHEN the
   latest decision was captured.
3. ``governance_action_domain`` CHECK extension (design Q8 friction rule):
   this is the feature migration that ships writers for the previously
   reserved actions CONSENT_GRANT / CONSENT_WITHDRAW / CONSENT_DENIED /
   OVERRIDE_APPLY / EMBED_HARD_DELETE, so they must move into the enforced
   DB vocabulary. EXPORT stays reserved (still no writer).

Round-trip safe for CI's upgrade head -> downgrade base -> re-upgrade: the
downgrade drops the added columns and restores the 0007-era CHECK before any
table work; both statements are metadata-only on empty-or-populated tables.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260824_0008"
down_revision: str | None = "20260824_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Vocabulary as of 20260824_0007 (restored verbatim by downgrade).
_CHECK_V0007 = (
    "action IN ("
    "'USER_CREATE','USER_UPDATE','USER_DELETE',"
    "'STUDENT_CREATE','STUDENT_UPDATE','STUDENT_DELETE',"
    "'TEMPLATE_ENROLL','ATTENDANCE_EVALUATE','REFRESH_REUSED',"
    "'LOGIN_SUCCEEDED','LOGOUT',"
    "'INFERENCE_ENQUEUED','TASK_READ','RECOGNITION_RUN',"
    "'GOVERNANCE_PURGE')"
)

# This revision's writers: everything above PLUS the newly wired consent,
# override, and embedding-hard-delete actions. EXPORT remains reserved.
_CHECK_V0008 = (
    "action IN ("
    "'USER_CREATE','USER_UPDATE','USER_DELETE',"
    "'STUDENT_CREATE','STUDENT_UPDATE','STUDENT_DELETE',"
    "'TEMPLATE_ENROLL','ATTENDANCE_EVALUATE','REFRESH_REUSED',"
    "'LOGIN_SUCCEEDED','LOGOUT',"
    "'INFERENCE_ENQUEUED','TASK_READ','RECOGNITION_RUN',"
    "'GOVERNANCE_PURGE',"
    "'CONSENT_GRANT','CONSENT_WITHDRAW','CONSENT_DENIED',"
    "'OVERRIDE_APPLY','EMBED_HARD_DELETE')"
)


def upgrade() -> None:
    """Add consent columns to students and extend the governance vocabulary."""
    op.add_column(
        "students",
        sa.Column(
            "biometric_consent_status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
    )
    op.create_check_constraint(
        "students_biometric_consent_status_domain",
        "students",
        sa.text(
            "biometric_consent_status IN "
            "('pending','granted','denied','withdrawn')"
        ),
    )
    op.add_column(
        "students",
        sa.Column(
            "biometric_consent_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    # Deliberate friction (design Q8): wiring writers for reserved actions is
    # a migration. Drop-and-recreate keeps one canonical constraint name.
    op.drop_constraint("governance_action_domain", "governance_logs", type_="check")
    op.create_check_constraint(
        "governance_action_domain",
        "governance_logs",
        sa.text(_CHECK_V0008),
    )


def downgrade() -> None:
    """Restore the 0007-era vocabulary and drop the consent columns."""
    op.drop_constraint("governance_action_domain", "governance_logs", type_="check")
    op.create_check_constraint(
        "governance_action_domain",
        "governance_logs",
        sa.text(_CHECK_V0007),
    )
    op.drop_constraint(
        "students_biometric_consent_status_domain",
        "students",
        type_="check",
    )
    op.drop_column("students", "biometric_consent_at")
    op.drop_column("students", "biometric_consent_status")
