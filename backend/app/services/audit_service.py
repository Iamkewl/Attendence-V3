"""Governance audit facade — the single write path for ``governance_logs`` rows.

ATT-006 foundation. Every audited action in the system is recorded through
this module; callers never construct :class:`GovernanceLog` ORM rows ad hoc
(mirrors the pipeline-facade rule).

Load-bearing choices (design: ``audit-service-design.md`` §2, decisions D1–D7):

- **flush(), never commit().** Emitted rows join the CALLER's transaction;
  commit stays where it is today. This gives all-or-nothing semantics: a
  rolled-back business operation can never leave a phantom governance row,
  and a successful mutation can never lack its record.
- **Mandatory vs advisory.** Actions in ``MANDATORY_ACTIONS`` fail the
  business operation when their audit write fails (strict). Advisory actions
  (auth/inference side-effects) log-and-continue (non-strict), mirroring the
  best-effort Redis-publish precedent in ``attendance_service``. Note that a
  non-strict emission may ROLL BACK the surrounding transaction to keep the
  session usable after a failed flush — callers must invoke it before any
  business writes they intend to keep.
- **Privacy hard rules.** ``change_summary`` carries field names and
  non-secret scalars only. Embedding vectors, embedding references/pseudonyms,
  passwords, password hashes, tokens, and secrets are stripped defensively by
  :func:`_sanitize_change_summary` and asserted absent by unit test.
  IP addresses are stored (nullable INET) **only** for consent/auth events
  per decision D4 and are never logged elsewhere.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import GovernanceLog


if TYPE_CHECKING:
    from datetime import datetime

    from fastapi import Request


LOGGER = logging.getLogger(__name__)


class GovernanceAction(str, Enum):
    """CHECK-constrained event vocabulary (DB constraint ``governance_action_domain``).

    Adding a value requires BOTH a new enum member here AND a migration
    extending the CHECK — deliberate friction (design Q8). Values below the
    ``reserved`` marker ship in the enum so ATT-044 (consent), ATT-038
    (overrides), and ATT-045 (retention/export) land without redesigning the
    vocabulary, but they have no writers yet and are intentionally excluded
    from the database CHECK until their feature migrations arrive.
    """

    USER_CREATE = "USER_CREATE"
    USER_UPDATE = "USER_UPDATE"
    USER_DELETE = "USER_DELETE"
    STUDENT_CREATE = "STUDENT_CREATE"
    STUDENT_UPDATE = "STUDENT_UPDATE"
    STUDENT_DELETE = "STUDENT_DELETE"
    TEMPLATE_ENROLL = "TEMPLATE_ENROLL"
    ATTENDANCE_EVALUATE = "ATTENDANCE_EVALUATE"
    REFRESH_REUSED = "REFRESH_REUSED"
    # advisory (decisions D1/D4/D6/D7):
    LOGIN_SUCCEEDED = "LOGIN_SUCCEEDED"
    LOGOUT = "LOGOUT"
    INFERENCE_ENQUEUED = "INFERENCE_ENQUEUED"
    TASK_READ = "TASK_READ"
    RECOGNITION_RUN = "RECOGNITION_RUN"
    GOVERNANCE_PURGE = "GOVERNANCE_PURGE"  # written only by purge_governance_before() SQL
    # --- reserved for parked efforts (no writer yet; not in the DB CHECK) ---
    CONSENT_GRANT = "CONSENT_GRANT"
    CONSENT_WITHDRAW = "CONSENT_WITHDRAW"
    OVERRIDE_APPLY = "OVERRIDE_APPLY"
    EMBED_HARD_DELETE = "EMBED_HARD_DELETE"
    EXPORT = "EXPORT"


# Implemented vocabulary mirrored by the ``governance_action_domain`` CHECK
# constraint (migration 20260824_0007). Reserved enum members stay out until
# their feature migrations extend the constraint.
IMPLEMENTED_ACTIONS: frozenset[str] = frozenset(
    {
        GovernanceAction.USER_CREATE.value,
        GovernanceAction.USER_UPDATE.value,
        GovernanceAction.USER_DELETE.value,
        GovernanceAction.STUDENT_CREATE.value,
        GovernanceAction.STUDENT_UPDATE.value,
        GovernanceAction.STUDENT_DELETE.value,
        GovernanceAction.TEMPLATE_ENROLL.value,
        GovernanceAction.ATTENDANCE_EVALUATE.value,
        GovernanceAction.REFRESH_REUSED.value,
        GovernanceAction.LOGIN_SUCCEEDED.value,
        GovernanceAction.LOGOUT.value,
        GovernanceAction.INFERENCE_ENQUEUED.value,
        GovernanceAction.TASK_READ.value,
        GovernanceAction.RECOGNITION_RUN.value,
        GovernanceAction.GOVERNANCE_PURGE.value,
    }
)

ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "user",
        "student",
        "student_embedding",
        "class_session_record",
        "inference_task",
        "auth_session",
        "governance_log",
    }
)

# Decision D2: governance retention must exceed the biometric horizon
# (ATT-045). Consumed by the future scheduled purge caller (scheduling is
# deferred); enforced operationally via purge_governance_before().
GOVERNANCE_RETENTION_DEFAULT_DAYS = 2555

# Decision D1: lifecycle events fail-the-op; auth/inference side-effects and
# TASK_READ are log-and-continue. Reserved future actions (consent, override,
# export, embed hard-delete) are mandatory by policy ahead of their wiring.
MANDATORY_ACTIONS: frozenset[GovernanceAction] = frozenset(
    {
        GovernanceAction.USER_CREATE,
        GovernanceAction.USER_UPDATE,
        GovernanceAction.USER_DELETE,
        GovernanceAction.STUDENT_CREATE,
        GovernanceAction.STUDENT_UPDATE,
        GovernanceAction.STUDENT_DELETE,
        GovernanceAction.TEMPLATE_ENROLL,
        GovernanceAction.ATTENDANCE_EVALUATE,
        GovernanceAction.REFRESH_REUSED,
        GovernanceAction.CONSENT_GRANT,
        GovernanceAction.CONSENT_WITHDRAW,
        GovernanceAction.OVERRIDE_APPLY,
        GovernanceAction.EMBED_HARD_DELETE,
        GovernanceAction.EXPORT,
    }
)

# Decision D4: IP capture is purpose-bound to consent + auth events. Any other
# action silently drops the supplied address (routine domain events store none).
IP_ALLOWED_ACTIONS: frozenset[GovernanceAction] = frozenset(
    {
        GovernanceAction.LOGIN_SUCCEEDED,
        GovernanceAction.LOGOUT,
        GovernanceAction.REFRESH_REUSED,
        GovernanceAction.CONSENT_GRANT,
        GovernanceAction.CONSENT_WITHDRAW,
    }
)

# Privacy allowlist inversion (Q9): keys that must never survive into a
# persisted ``change_summary``. ``_sanitize_change_summary`` strips them
# defensively; a unit test asserts the stripping so regressions are loud.
_FORBIDDEN_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "embedding",
        "embeddings",
        "embedding_vector",
        "embedding_reference",
        "matched_embedding_id",
        "identity",
        "password",
        "password_hash",
        "token",
        "access_token",
        "refresh_token",
        "secret",
        "api_key",
        "authorization",
        "cookie",
        "ip_address",
    }
)


@dataclass(frozen=True)
class AuditEvent:
    """One governance ledger entry awaiting emission."""

    action: GovernanceAction
    entity_type: str
    entity_id: UUID | None = None
    class_session_record_id: UUID | None = None
    actor_user_id: UUID | None = None  # None => system actor (e.g. Celery aggregation)
    reason: str | None = None
    change_summary: Mapping[str, object] = field(default_factory=dict)
    request_id: UUID | None = None
    ip_address: str | None = None  # INET-castable string; honored only for IP_ALLOWED_ACTIONS


def _sanitize_change_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Return a JSON-safe copy of ``summary`` with forbidden secret keys removed.

    Hard privacy rule (mission invariant + design Q9): embeddings, credential
    material, and tokens never enter the governance ledger even when a caller
    mistakenly passes them.
    """
    sanitized: dict[str, object] = {}
    for key, value in summary.items():
        if str(key).lower() in _FORBIDDEN_SUMMARY_KEYS:
            continue
        sanitized[key] = value
    return sanitized


def resolve_request_context(request: Request) -> tuple[UUID | None, str | None]:
    """Extract ``(request_id, client_ip)`` from a FastAPI request, tolerating absence.

    ``request.state.request_id`` is stored as a string by RequestIDMiddleware;
    the column is UUID, so parse defensively. The IP is ``request.client.host``
    only — X-Forwarded-For is deliberately NOT trusted (design Q10).
    """
    rid: UUID | None = None
    raw_request_id = getattr(request.state, "request_id", None)
    if raw_request_id:
        try:
            rid = UUID(str(raw_request_id))
        except (ValueError, TypeError):
            rid = None
    ip = request.client.host if request.client else None
    return rid, ip


class AuditService:
    """Read/write access to the append-only governance ledger."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def emit(self, event: AuditEvent) -> GovernanceLog:
        """Insert one governance row via flush() — joins the CALLER's transaction.

        Never commits: the row becomes durable exactly when the surrounding
        business transaction commits, and vanishes with its rollback.
        """
        ip_address = (
            event.ip_address
            if event.action in IP_ALLOWED_ACTIONS and event.ip_address
            else None
        )
        row = GovernanceLog(
            action=event.action.value,
            entity_type=event.entity_type,
            entity_id=event.entity_id,
            class_session_record_id=event.class_session_record_id,
            actor_user_id=event.actor_user_id,
            reason=event.reason,
            change_summary=_sanitize_change_summary(event.change_summary),
            request_id=event.request_id,
            ip_address=ip_address,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def list_events(
        self,
        *,
        actor_user_id: UUID | None = None,
        entity_type: str | None = None,
        entity_id: UUID | None = None,
        action: str | None = None,
        class_session_record_id: UUID | None = None,
        request_id: UUID | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[GovernanceLog]:
        """List ledger entries newest-first; each filter maps onto an existing index."""
        stmt = select(GovernanceLog)
        if actor_user_id is not None:
            stmt = stmt.where(GovernanceLog.actor_user_id == actor_user_id)
        if entity_type is not None:
            stmt = stmt.where(GovernanceLog.entity_type == entity_type)
        if entity_id is not None:
            stmt = stmt.where(GovernanceLog.entity_id == entity_id)
        if action is not None:
            stmt = stmt.where(GovernanceLog.action == action)
        if class_session_record_id is not None:
            stmt = stmt.where(GovernanceLog.class_session_record_id == class_session_record_id)
        if request_id is not None:
            stmt = stmt.where(GovernanceLog.request_id == request_id)
        if since is not None:
            stmt = stmt.where(GovernanceLog.created_at >= since)
        if until is not None:
            stmt = stmt.where(GovernanceLog.created_at <= until)

        stmt = stmt.order_by(GovernanceLog.created_at.desc()).offset(offset).limit(limit)
        return list((await self._execute_read(stmt)).scalars().all())

    async def _execute_read(self, statement: object):
        """Execute a read query, rolling back the session on failure."""
        try:
            return await self.session.execute(statement)  # type: ignore[arg-type]
        except Exception:
            await self.session.rollback()
            raise


async def emit(
    session: AsyncSession, event: AuditEvent, *, strict: bool = True
) -> GovernanceLog | None:
    """Module-level convenience wrapper implementing the D1 failure policy.

    ``strict=True`` (mandatory actions): re-raise any emission failure — the
    caller's exception handling rolls back the whole transaction, so the
    business operation cannot complete without its audit record.

    ``strict=False`` (advisory actions): swallow and log. If the flush poisoned
    the transaction, roll it back so the caller's subsequent business writes
    still succeed; therefore call this BEFORE the business writes it must not
    destroy.
    """
    try:
        return await AuditService(session).emit(event)
    except Exception:
        if strict:
            raise
        LOGGER.exception(
            "Advisory governance event %s/%s could not be persisted; continuing.",
            event.action.value,
            event.entity_type,
        )
        try:
            await session.rollback()
        except Exception:
            LOGGER.exception("Session rollback after advisory audit failure also failed.")
        return None


__all__ = [
    "AuditEvent",
    "AuditService",
    "ENTITY_TYPES",
    "GOVERNANCE_RETENTION_DEFAULT_DAYS",
    "GovernanceAction",
    "IMPLEMENTED_ACTIONS",
    "IP_ALLOWED_ACTIONS",
    "MANDATORY_ACTIONS",
    "emit",
    "resolve_request_context",
]
