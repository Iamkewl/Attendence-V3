"""Student domain service encapsulating enrollment and profile persistence rules."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Student, User
from app.domain.schemas import StudentCreate, StudentUpdate
from app.services.base import AsyncCRUDService


class StudentServiceError(Exception):
    """Base error type for student service validation failures."""


class StudentLinkValidationError(StudentServiceError):
    """Raised when a student profile cannot be linked to a valid user account."""


class StudentService(AsyncCRUDService[Student, StudentCreate, StudentUpdate]):
    """Application service for transactional student CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session=session, model=Student)

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
        return await self.create(payload)

    async def update_student(self, student_id: UUID, payload: StudentUpdate) -> Student | None:
        """Patch mutable student fields and return the updated profile when found."""
        return await self.update(student_id, payload.model_dump(exclude_unset=True))

    async def _validate_linked_user(self, user_id: UUID) -> None:
        """Ensure each student profile maps to an existing, active user identity."""
        stmt = select(User).where(User.id == user_id)
        user = (await self._execute_read(stmt)).scalar_one_or_none()

        if user is None:
            raise StudentLinkValidationError("Linked user does not exist.")

        if not user.is_active:
            raise StudentLinkValidationError("Linked user is inactive.")


__all__ = ["StudentLinkValidationError", "StudentService", "StudentServiceError"]
