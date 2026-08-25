"""GovernanceLog ORM model."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import text

from ._base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from .user import User
    from .session import ClassSessionRecord


class GovernanceLog(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Captures auditable governance actions across attendance domain entities.

    Append-only by database trigger (migration 20260824_0007): UPDATE and
    DELETE raise; the only sanctioned deletion path is the ops-only SECURITY
    DEFINER ``purge_governance_before(cutoff)``. The vocabulary CHECK below
    mirrors that migration's ``governance_action_domain`` constraint so ORM
    metadata and DB stay in sync (autogenerate stays empty).
    """

    __tablename__ = "governance_logs"
    __table_args__ = (
        CheckConstraint("char_length(btrim(action)) > 0", name="governance_action_not_blank"),
        CheckConstraint(
            "char_length(btrim(entity_type)) > 0",
            name="governance_entity_type_not_blank",
        ),
        CheckConstraint(
            "action IN ("
            "'USER_CREATE','USER_UPDATE','USER_DELETE',"
            "'STUDENT_CREATE','STUDENT_UPDATE','STUDENT_DELETE',"
            "'TEMPLATE_ENROLL','ATTENDANCE_EVALUATE','REFRESH_REUSED',"
            "'LOGIN_SUCCEEDED','LOGOUT',"
            "'INFERENCE_ENQUEUED','TASK_READ','RECOGNITION_RUN',"
            "'GOVERNANCE_PURGE',"
            "'CONSENT_GRANT','CONSENT_WITHDRAW','CONSENT_DENIED',"
            "'OVERRIDE_APPLY','EMBED_HARD_DELETE')",
            name="governance_action_domain",
        ),
        Index("ix_governance_actor_created_at", "actor_user_id", "created_at"),
        Index("ix_governance_entity_lookup", "entity_type", "entity_id"),
        Index("ix_governance_class_session_record_id", "class_session_record_id"),
        Index("ix_governance_request_id", "request_id"),
        Index("ix_governance_created_at", "created_at"),
        Index(
            "ix_governance_action_created_at",
            "action",
            text("created_at DESC"),
        ),
    )

    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    class_session_record_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("class_session_records.id", ondelete="SET NULL"),
        nullable=True,
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    change_summary: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    request_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(INET, nullable=True)

    actor_user: Mapped[User | None] = relationship(
        back_populates="governance_logs",
        lazy="select",
        foreign_keys=[actor_user_id],
    )
    class_session_record: Mapped[ClassSessionRecord | None] = relationship(
        back_populates="governance_logs",
        lazy="select",
        foreign_keys=[class_session_record_id],
    )
