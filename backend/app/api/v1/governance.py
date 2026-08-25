"""Governance ledger read endpoints (ATT-006).

Read-ONLY by design: no write endpoints exist and none ever will — purging is
an ops-only SECURITY DEFINER SQL function (decision D3), never an API
principal. Access is restricted to ADMIN and AUDITOR (decision D5): AUDITOR
sees the whole ledger but holds write power nowhere else; INSTRUCTOR and
OPERATOR are denied 403 (fail closed).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentGovernanceReader
from app.core.database import get_async_session
from app.domain.schemas import GovernanceLogRead
from app.services.audit_service import AuditService, ENTITY_TYPES, GovernanceAction


router = APIRouter(prefix="/governance", tags=["Governance"])

_EntityTypeFilter = Annotated[
    str | None,
    Query(description="Filter by entity type (validated against the event vocabulary)."),
]


@router.get(
    "/events",
    response_model=list[GovernanceLogRead],
    summary="List Governance Events",
    description=(
        "Read the append-only governance ledger, newest first. Filters map 1:1 "
        "onto existing indexes: actor_user_id, entity_type/entity_id, action "
        "(+ created_at window), class_session_record_id, request_id."
    ),
)
async def list_governance_events(
    reader: CurrentGovernanceReader,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    actor_user_id: UUID | None = Query(None),
    entity_type: _EntityTypeFilter = None,
    entity_id: UUID | None = Query(None),
    action: GovernanceAction | None = Query(None),
    class_session_record_id: UUID | None = Query(None),
    request_id: UUID | None = Query(None),
    since: Annotated[datetime | None, Query(description="Inclusive lower created_at bound.")] = None,
    until: Annotated[datetime | None, Query(description="Inclusive upper created_at bound.")] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> list[GovernanceLogRead]:
    """List governance events with vocabulary-validated filters, newest first."""
    if entity_type is not None and entity_type not in ENTITY_TYPES:
        # Fail closed on unknown vocabulary instead of silently returning [].
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Unknown entity_type {entity_type!r}; expected one of: "
                f"{', '.join(sorted(ENTITY_TYPES))}."
            ),
        )

    rows = await AuditService(session).list_events(
        actor_user_id=actor_user_id,
        entity_type=entity_type,
        entity_id=entity_id,
        action=action.value if action is not None else None,
        class_session_record_id=class_session_record_id,
        request_id=request_id,
        since=since,
        until=until,
        offset=offset,
        limit=limit,
    )
    return [GovernanceLogRead.model_validate(row) for row in rows]


__all__ = ["router"]
