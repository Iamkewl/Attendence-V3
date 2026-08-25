"""Governance vocabulary extension for ATT-039 attendance export events.

Revision ID: 20260824_0009
Revises: 20260824_0008
Create Date: 2026-08-25 09:00:00

ATT-039 (CSV attendance roster export) ships the first writer for the
previously reserved ``EXPORT`` action, so — per design Q8's deliberate
friction rule — this migration moves it into the enforced
``governance_action_domain`` CHECK. No columns or tables change; both
statements are metadata-only on empty-or-populated tables.

Round-trip safe for CI's upgrade head -> downgrade base -> re-upgrade:
downgrade restores the 0008-era CHECK verbatim. As with every vocabulary
shrink, a downgrade on a database that still holds EXPORT rows would fail
on constraint validation; CI round-trips against an empty schema.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260824_0009"
down_revision: str | None = "20260824_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Vocabulary as of 20260824_0008 (restored verbatim by downgrade).
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

# This revision's writer: everything above PLUS 'EXPORT' (ATT-039 CSV
# attendance roster export).
_CHECK_V0009 = (
    "action IN ("
    "'USER_CREATE','USER_UPDATE','USER_DELETE',"
    "'STUDENT_CREATE','STUDENT_UPDATE','STUDENT_DELETE',"
    "'TEMPLATE_ENROLL','ATTENDANCE_EVALUATE','REFRESH_REUSED',"
    "'LOGIN_SUCCEEDED','LOGOUT',"
    "'INFERENCE_ENQUEUED','TASK_READ','RECOGNITION_RUN',"
    "'GOVERNANCE_PURGE',"
    "'CONSENT_GRANT','CONSENT_WITHDRAW','CONSENT_DENIED',"
    "'OVERRIDE_APPLY','EMBED_HARD_DELETE',"
    "'EXPORT')"
)


def upgrade() -> None:
    """Admit EXPORT into the enforced governance action vocabulary."""
    # Deliberate friction (design Q8): wiring a writer for a reserved action
    # is a migration. Drop-and-recreate keeps one canonical constraint name.
    op.drop_constraint("governance_action_domain", "governance_logs", type_="check")
    op.create_check_constraint(
        "governance_action_domain",
        "governance_logs",
        sa.text(_CHECK_V0009),
    )


def downgrade() -> None:
    """Restore the 0008-era vocabulary."""
    op.drop_constraint("governance_action_domain", "governance_logs", type_="check")
    op.create_check_constraint(
        "governance_action_domain",
        "governance_logs",
        sa.text(_CHECK_V0008),
    )
