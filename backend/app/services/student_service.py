"""Student domain service encapsulating enrollment and profile persistence rules."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Student, User
from app.domain.schemas import StudentCreate, StudentUpdate
from app.services.audit_service import AuditEvent, GovernanceAction
from app.services.base import AsyncCRUDService


class StudentServiceError(Exception):
    """Base error type for student service validation failures."""


class StudentLinkValidationError(StudentServiceError):
    """Raised when a student profile cannot be linked to a valid user account."""


class StudentService(AsyncCRUDService[Student, StudentCreate, StudentUpdate]):
    """Application service for transactional student CRUD operations."""

    def __init__(self, session: AsyncSession, *, actor: User | None = None) -> None:
        super().__init__(session=session, model=Student)
        self._actor = actor

    def _actor_id(self) -> UUID | None:
        return self._actor.id if self._actor is not None else None

    async def get_by_student_number(self, student_number: str) -> Student | None:
        """Load one student profile by unique student number."""
        stmt = select(Student).where(Student.student_number == student_number.strip())
        return (await self._execute_read(stmt)).scalar_one_or_none()

    async def list_students(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        is_active: bool | None = None,
        enrollment_year: int | None = None,
    ) -> list[Student]:
        """Return paginated students with optional status and cohort filters."""
        filters = []
        if is_active is not None:
            filters.append(Student.is_active.is_(is_active))
        if enrollment_year is not None:
            filters.append(Student.enrollment_year == enrollment_year)

        return await self.list(
            offset=offset,
            limit=limit,
            filters=filters or None,
            order_by=Student.created_at.desc(),
        )

    async def create_student(self, payload: StudentCreate) -> Student:
        """Create a student profile after validating linked user presence and state."""
        await self._validate_linked_user(payload.user_id)
        # STUDENT_CREATE is mandatory (D1): the queued event flushes inside
        # create()'s transaction, so a failed audit write fails the create.
        # The factory runs after the INSERT to capture the generated id.
        self.queue_audit_event(
            lambda created: AuditEvent(
                action=GovernanceAction.STUDENT_CREATE,
                entity_type="student",
                entity_id=created.id,
                actor_user_id=self._actor_id(),
                change_summary={
                    "student_number": payload.student_number,
                    "enrollment_year": payload.enrollment_year,
                },
            )
        )
        return await self.create(payload)

    async def update_student(self, student_id: UUID, payload: StudentUpdate) -> Student | None:
        """Patch mutable student fields and return the updated profile when found."""
        values = payload.model_dump(exclude_unset=True)
        self.queue_audit_event(
            lambda _result: AuditEvent(
                action=GovernanceAction.STUDENT_UPDATE,
                entity_type="student",
                entity_id=student_id,
                actor_user_id=self._actor_id(),
                change_summary={"fields_changed": sorted(values)},
            )
        )
        return await self.update(student_id, values)

    async def delete_student(self, student_id: UUID) -> bool:
        """Delete a student profile, recording its pre-delete activation state."""
        student = await self.get(student_id)
        if student is None:
            return False

        self.queue_audit_event(
            lambda _result: AuditEvent(
                action=GovernanceAction.STUDENT_DELETE,
                entity_type="student",
                entity_id=student.id,
                actor_user_id=self._actor_id(),
                change_summary={"was_active": student.is_active},
            )
        )
        return await self.delete(student_id)

    async def _validate_linked_user(self, user_id: UUID) -> None:
        """Ensure each student profile maps to an existing, active user identity."""
        stmt = select(User).where(User.id == user_id)
        user = (await self._execute_read(stmt)).scalar_one_or_none()

        if user is None:
            raise StudentLinkValidationError("Linked user does not exist.")

        if not user.is_active:
            raise StudentLinkValidationError("Linked user is inactive.")


__all__ = ["StudentLinkValidationError", "StudentService", "StudentServiceError"]
