"""Governance hardening: action-domain CHECK, listing index, append-only trigger, purge function.

Revision ID: 20260824_0007
Revises: 20260824_0006
Create Date: 2026-08-24 12:00:00

ATT-006 foundation (design ``audit-service-design.md`` §5-§7, decisions D2/D3/D8):

1. ``governance_action_domain`` CHECK pins the implemented event vocabulary.
   Adding a value is a deliberate migration (design Q8). Reserved future
   actions (CONSENT_*, OVERRIDE_APPLY, EMBED_HARD_DELETE, EXPORT) are absent
   until their feature migrations arrive.
2. ``ix_governance_action_created_at (action, created_at DESC)`` backs the
   filtered listing on GET /api/v1/governance/events.
3. Append-only guard (Option B): BEFORE UPDATE OR DELETE raises, with one
   narrow carve-out — the referential-action UPDATE fired by the FKs'
   ``ON DELETE SET NULL`` (actor deleted / session record deleted), which is
   the documented survival contract for governance rows. Deliberately NO ON
   TRUNCATE component — the test fixtures TRUNCATE governance_logs between
   tests (backend/tests/conftest.py _DOMAIN_TABLES) and a TRUNCATE guard
   would break every test run.
4. ``purge_governance_before(cutoff)`` — ops-only SECURITY DEFINER retention
   purge (decision D3). It logs its own GOVERNANCE_PURGE invocation row
   BEFORE deleting, then temporarily disarms the append-only trigger via a
   transaction-local custom GUC instead of session_replication_role: the
   latter is SUSET-only and would require superuser, which an ops role must
   not hold. Scheduling (Celery beat wiring against
   ATTENDANCE_GOVERNANCE_RETENTION_DAYS, default 2555 days per decision D2)
   is deferred to ATT-045.

Round-trip safe for CI's upgrade head -> downgrade base -> re-upgrade: the
downgrade drops trigger/function/index/constraint before any table work and
governance_logs itself predates this revision untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260824_0007"
down_revision: str | None = "20260824_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_APPEND_ONLY_FUNCTION = """
CREATE FUNCTION public.forbid_governance_mutation() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        -- Single permitted mutation: the referential-action UPDATE fired by
        -- ``ON DELETE SET NULL`` on actor_user_id / class_session_record_id
        -- (the documented survival contract: governance rows outlive their
        -- actor or session record). Allowed ONLY when every other column -
        -- including id and timestamps - is byte-identical and the FK column
        -- went NOT NULL -> NULL. Anything else (tampering, anonymized
        -- re-attribution, edited payloads) is refused. NOTE: any future
        -- migration adding a column to governance_logs must extend this
        -- comparison, or ordinary FK anonymization will start failing.
        IF NEW.id IS NOT DISTINCT FROM OLD.id
           AND NEW.action IS NOT DISTINCT FROM OLD.action
           AND NEW.entity_type IS NOT DISTINCT FROM OLD.entity_type
           AND NEW.entity_id IS NOT DISTINCT FROM OLD.entity_id
           AND NEW.reason IS NOT DISTINCT FROM OLD.reason
           AND NEW.change_summary IS NOT DISTINCT FROM OLD.change_summary
           AND NEW.request_id IS NOT DISTINCT FROM OLD.request_id
           AND NEW.ip_address IS NOT DISTINCT FROM OLD.ip_address
           AND NEW.created_at IS NOT DISTINCT FROM OLD.created_at
           AND NEW.updated_at IS NOT DISTINCT FROM OLD.updated_at
           AND (
               NEW.actor_user_id IS NOT DISTINCT FROM OLD.actor_user_id
               OR (OLD.actor_user_id IS NOT NULL AND NEW.actor_user_id IS NULL)
           )
           AND (
               NEW.class_session_record_id IS NOT DISTINCT FROM OLD.class_session_record_id
               OR (OLD.class_session_record_id IS NOT NULL AND NEW.class_session_record_id IS NULL)
           )
        THEN
            RETURN NEW;
        END IF;
    END IF;

    -- Break-glass path for the sanctioned retention purge ONLY:
    -- public.purge_governance_before() sets app.governance_purge_active = 'on'
    -- as a transaction-local GUC before its DELETE. Any other UPDATE/DELETE
    -- (including from the application DB role or via SQL injection in app
    -- context) is refused.
    IF COALESCE(current_setting('app.governance_purge_active', true), 'off') = 'on' THEN
        RETURN COALESCE(NEW, OLD);
    END IF;
    RAISE EXCEPTION 'governance_logs is append-only'
        USING ERRCODE = 'restrict_violation';
END $$ LANGUAGE plpgsql;
"""

_APPEND_ONLY_TRIGGER = """
CREATE TRIGGER trg_governance_append_only
BEFORE UPDATE OR DELETE ON public.governance_logs
FOR EACH ROW EXECUTE FUNCTION public.forbid_governance_mutation();
"""

_PURGE_FUNCTION = """
CREATE FUNCTION public.purge_governance_before(cutoff timestamptz)
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    candidate_count integer;
    purged_count integer;
BEGIN
    IF cutoff IS NULL THEN
        RAISE EXCEPTION 'cutoff must not be null';
    END IF;

    SELECT count(*) INTO candidate_count
    FROM public.governance_logs
    WHERE created_at < cutoff;

    -- Log the invocation FIRST (decision D3): the row records who/when/how
    -- many even if the delete that follows fails. The freshly inserted row
    -- has created_at = now() > cutoff, so it survives its own delete below.
    INSERT INTO public.governance_logs (
        id, actor_user_id, class_session_record_id, action, entity_type,
        entity_id, reason, change_summary, request_id, ip_address,
        created_at, updated_at
    ) VALUES (
        gen_random_uuid(), NULL, NULL, 'GOVERNANCE_PURGE', 'governance_log',
        NULL, 'retention_purge (ATTENDANCE_GOVERNANCE_RETENTION_DAYS)',
        jsonb_build_object(
            'cutoff', cutoff,
            'candidate_rows', candidate_count,
            'source', 'ops_security_definer'
        ),
        NULL, NULL, now(), now()
    );

    -- Transaction-local disarm of the append-only trigger; restored when the
    -- transaction ends. Custom GUCs are settable without superuser.
    PERFORM set_config('app.governance_purge_active', 'on', true);

    DELETE FROM public.governance_logs WHERE created_at < cutoff;
    GET DIAGNOSTICS purged_count = ROW_COUNT;

    RETURN purged_count;
END $$;
"""


def upgrade() -> None:
    """Pin vocabulary, add listing index, enforce append-only, ship purge path."""
    # 1. Vocabulary pinning over IMPLEMENTED actions only; extend via future
    #    migrations when reserved actions ship — deliberate friction.
    op.create_check_constraint(
        "governance_action_domain",
        "governance_logs",
        sa.text(
            "action IN ("
            "'USER_CREATE','USER_UPDATE','USER_DELETE',"
            "'STUDENT_CREATE','STUDENT_UPDATE','STUDENT_DELETE',"
            "'TEMPLATE_ENROLL','ATTENDANCE_EVALUATE','REFRESH_REUSED',"
            "'LOGIN_SUCCEEDED','LOGOUT',"
            "'INFERENCE_ENQUEUED','TASK_READ','RECOGNITION_RUN',"
            "'GOVERNANCE_PURGE')"
        ),
    )
    # 2. Action-filtered, newest-first listing support.
    op.create_index(
        "ix_governance_action_created_at",
        "governance_logs",
        ["action", sa.text("created_at DESC")],
    )
    # 3. Append-only guard (design §5 Option B). NOTE: intentionally no ON
    #    TRUNCATE component — test fixtures TRUNCATE this table.
    op.execute(_APPEND_ONLY_FUNCTION)
    op.execute(_APPEND_ONLY_TRIGGER)
    # 4. Ops-only retention purge path (scheduling deferred to ATT-045).
    op.execute(_PURGE_FUNCTION)


def downgrade() -> None:
    """Drop the purge path, append-only guard, index, and vocabulary CHECK."""
    op.execute("DROP FUNCTION IF EXISTS public.purge_governance_before(timestamptz)")
    op.execute("DROP TRIGGER IF EXISTS trg_governance_append_only ON public.governance_logs")
    op.execute("DROP FUNCTION IF EXISTS public.forbid_governance_mutation()")
    op.drop_index("ix_governance_action_created_at", table_name="governance_logs")
    op.drop_constraint("governance_action_domain", "governance_logs", type_="check")
