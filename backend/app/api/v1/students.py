"""Student profile endpoints protected with instructor/admin RBAC policies."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Annotated
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentAdminUser, CurrentInstructorUser
from app.core.database import get_async_session
from app.domain.models import Student, StudentEmbedding, TemplateAuditLog
from app.domain.schemas import StudentCreate, StudentEnrollmentRead, StudentRead, StudentUpdate
from app.services.enrollment_quality import _resolve_enrollment_min_quality
from app.services.audit_service import AuditEvent, AuditService, GovernanceAction
from app.services.pipeline_service import extract_enrollment_embedding
from app.services.student_service import StudentLinkValidationError, StudentService
from app.infrastructure.triton import (
    TritonInferenceError,
    TritonModelUnavailableError,
    TritonServerUnavailableError,
    TritonTimeoutError,
)


router = APIRouter(prefix="/students", tags=["Students"])
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ATT-029: enrollment quality gate.
#
# The threshold resolver now lives in ``app.services.enrollment_quality``
# (single source of truth, shared with the bulk enrollment importer —
# services must not import from the api layer, and duplicating a
# fail-closed policy invites drift). It is imported at the top of this
# module so existing call sites and tests importing
# ``_resolve_enrollment_min_quality`` from here keep working unchanged.
#
# Behavior is unchanged: the threshold is read PER-REQUEST (not cached at
# import), default 0.5, range [0.0, 1.0]. When `quality_score` falls
# strictly below it, the API refuses the upload with 422 "Image quality
# too low for enrollment"; no StudentEmbedding row is written, no
# TemplateAuditLog row is written, and no existing template is rotated to
# inactive. Malformed env values fail closed (RuntimeError → 500).
# ---------------------------------------------------------------------------


def _normalize_pose_label(raw_pose_label: str | None) -> str:
    """Normalize pose labels for deterministic multi-template enrollment."""
    pose_label = (raw_pose_label or "front").strip().lower()
    if not pose_label:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="pose_label must not be blank.",
        )
    if len(pose_label) > 32:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="pose_label must be at most 32 characters long.",
        )
    return pose_label


async def _decode_enrollment_image(image_file: UploadFile) -> np.ndarray:
    """Decode uploaded image bytes to a float32 RGB tensor consumable by Triton helpers."""
    if image_file.content_type and not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image.",
        )

    image_bytes = await image_file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
            tensor = np.asarray(rgb_image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid decodable image.",
        ) from exc

    if tensor.ndim != 3 or tensor.shape[0] < 8 or tensor.shape[1] < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image dimensions are too small for enrollment.",
        )

    return np.ascontiguousarray(tensor, dtype=np.float32)


@router.post(
    "",
    response_model=StudentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create Student",
    description="Create a student profile linked to an existing active user.",
)
async def create_student(
    payload: StudentCreate,
    current_user: CurrentInstructorUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> StudentRead:
    """Create one student profile with linked-user validation."""
    service = StudentService(session, actor=current_user)

    try:
        return await service.create_student(payload)
    except StudentLinkValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The student_number or user mapping already exists.",
        ) from exc


@router.get(
    "",
    response_model=list[StudentRead],
    summary="List Students",
    description="List students with optional cohort and active-state filters.",
)
async def list_students(
    _: CurrentInstructorUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    is_active: Annotated[bool | None, Query()] = None,
    enrollment_year: Annotated[int | None, Query(ge=2000, le=2100)] = None,
) -> list[StudentRead]:
    """Return a paginated collection of student records."""
    service = StudentService(session)
    return await service.list_students(
        offset=offset,
        limit=limit,
        is_active=is_active,
        enrollment_year=enrollment_year,
    )


@router.get(
    "/{student_id}",
    response_model=StudentRead,
    summary="Get Student",
    description="Fetch one student profile by immutable identifier.",
)
async def get_student(
    student_id: UUID,
    _: CurrentInstructorUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> StudentRead:
    """Load one student profile by UUID."""
    service = StudentService(session)
    student = await service.get(student_id)
    if student is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")
    return student


@router.post(
    "/{student_id}/enroll",
    response_model=StudentEnrollmentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll Student Face Template",
    description=(
        "Extract one 512D face embedding from an uploaded image and persist it "
        "as an active enrollment template for the specified student and pose."
    ),
)
async def enroll_student_template(
    student_id: UUID,
    current_user: CurrentInstructorUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    image_file: Annotated[UploadFile, File(description="Image file used for enrollment.")],
    pose_label: Annotated[str | None, Form(max_length=32)] = None,
) -> StudentEnrollmentRead:
    """Create or rotate one active pose template for a student using Triton embeddings."""
    student = await session.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")

    if not student.is_active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot enroll templates for an inactive student.",
        )

    normalized_pose_label = _normalize_pose_label(pose_label)
    image_tensor = await _decode_enrollment_image(image_file)

    try:
        embedding, quality_score = await extract_enrollment_embedding(image_tensor)
    except (
        TritonTimeoutError,
        TritonServerUnavailableError,
        TritonModelUnavailableError,
        TritonInferenceError,
    ) as exc:
        LOGGER.exception(
            "Enrollment embedding extraction failed for student_id=%s pose=%s.",
            student_id,
            normalized_pose_label,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding extraction service is unavailable. Please retry shortly.",
        ) from exc
    except Exception as exc:
        LOGGER.exception(
            "Unexpected enrollment failure for student_id=%s pose=%s.",
            student_id,
            normalized_pose_label,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process enrollment image.",
        ) from exc

    # ATT-029: server-side enrollment quality gate. The pre-fix API stored
    # whatever `quality_score` LVFace returned with no threshold — a 0.05
    # embedding could become the active template and poison subsequent
    # recognition passes. The gate reads `ATTENDANCE_ENROLLMENT_MIN_QUALITY`
    # per-request (default 0.5) and refuses the upload with 422 when the
    # quality falls below it. NO StudentEmbedding row, TemplateAuditLog
    # row, or pose-rotation occurs when the gate triggers — the student's
    # pre-existing enrollment state is preserved exactly as it was.
    #
    # The 422 is chosen (not 400 / 409) per the issue's literal FIX:
    # "reject ... with 422 'Image quality too low for enrollment, please
    # retake'". 422 UNPROCESSABLE ENTITY is the right HTTP code for a
    # client-provided image that the server can read but cannot process
    # to a usable embedding per the server's own quality policy.
    #
    # Fail-closed: a bad ATTENDANCE_ENROLLMENT_MIN_QUALITY configuration
    # raises a RuntimeError out of _resolve_enrollment_min_quality(); we
    # surface that as a 500 (with a LOGGER.exception line for the diagnosis)
    # rather than silently accepting garbage embeddings under bad config.
    try:
        min_quality = _resolve_enrollment_min_quality()
    except RuntimeError as exc:
        LOGGER.exception(
            "Enrollment quality gate refused to start for student_id=%s pose=%s.",
            student_id,
            normalized_pose_label,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Enrollment quality gate is misconfigured "
                "(ATTENDANCE_ENROLLMENT_MIN_QUALITY error). "
                "Please contact the administrator."
            ),
        ) from exc

    if quality_score < min_quality:
        LOGGER.info(
            "Enrollment refused for student_id=%s pose=%s: quality_score=%.4f "
            "below ATTENDANCE_ENROLLMENT_MIN_QUALITY=%.4f.",
            student_id,
            normalized_pose_label,
            quality_score,
            min_quality,
        )
        raise HTTPException(
            # ATT-029 note: use HTTP_422_UNPROCESSABLE_CONTENT (the new
            # starlette 1.x name) — the older HTTP_422_UNPROCESSABLE_ENTITY
            # fires StarletteDeprecationWarning, which pyproject.toml
            # promotes to an error via filterwarnings=["error"]. The other
            # call sites in students.py (lines 101, 106) still use the old
            # name because no test exercises those routes in a way that
            # hits the constant; touching them now would broaden the diff
            # beyond B26's owned scope.
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Image quality too low for enrollment "
                f"(quality={quality_score:.4f}, minimum={min_quality:.4f}). "
                f"Please retake the image and retry."
            ),
        )

    created_embedding: StudentEmbedding | None = None
    try:
        active_pose_embeddings = list(
            (
                await session.execute(
                    select(StudentEmbedding)
                    .where(StudentEmbedding.student_id == student_id)
                    .where(StudentEmbedding.pose_label == normalized_pose_label)
                    .where(StudentEmbedding.is_active.is_(True))
                    .with_for_update()
                )
            ).scalars().all()
        )

        for active_embedding in active_pose_embeddings:
            active_embedding.is_active = False
            session.add(
                TemplateAuditLog(
                    student_id=student_id,
                    student_embedding_id=active_embedding.id,
                    action="archived",
                    pose_label=active_embedding.pose_label,
                    quality_score=active_embedding.quality_score,
                    event_metadata={"reason": "pose_reenrollment"},
                )
            )

        created_embedding = StudentEmbedding(
            student_id=student_id,
            embedding=[float(value) for value in embedding.tolist()],
            pose_label=normalized_pose_label,
            quality_score=float(quality_score),
            is_active=True,
        )
        session.add(created_embedding)
        await session.flush()

        session.add(
            TemplateAuditLog(
                student_id=student_id,
                student_embedding_id=created_embedding.id,
                action="created",
                pose_label=normalized_pose_label,
                quality_score=float(quality_score),
                event_metadata={"source": "students_enroll_api_v1"},
            )
        )
        # ATT-006: TEMPLATE_ENROLL is a mandatory governance event (D1) and
        # joins this same transaction — flush-only, committed by the single
        # commit below (the TemplateAuditLog same-tx pattern with a second
        # row type). Summary carries the quality score and pose label, NEVER
        # any vector material.
        await AuditService(session).emit(
            AuditEvent(
                action=GovernanceAction.TEMPLATE_ENROLL,
                entity_type="student_embedding",
                entity_id=created_embedding.id,
                actor_user_id=current_user.id,
                change_summary={
                    "pose_label": normalized_pose_label,
                    "quality_score": float(quality_score),
                    "replaced_count": len(active_pose_embeddings),
                },
            )
        )
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Enrollment write conflicted with an existing record.",
        ) from exc
    except Exception:
        await session.rollback()
        raise

    if created_embedding is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Enrollment failed to persist an embedding.",
        )

    return created_embedding


@router.patch(
    "/{student_id}",
    response_model=StudentRead,
    summary="Update Student",
    description="Patch mutable student fields by identifier.",
)
async def update_student(
    student_id: UUID,
    payload: StudentUpdate,
    current_user: CurrentInstructorUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> StudentRead:
    """Apply partial updates to an existing student profile."""
    service = StudentService(session, actor=current_user)

    try:
        updated = await service.update_student(student_id, payload)
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The requested student update conflicts with an existing record.",
        ) from exc

    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")

    return updated


@router.delete(
    "/{student_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Student",
    description="Delete one student profile by identifier.",
)
async def delete_student(
    student_id: UUID,
    current_user: CurrentAdminUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> Response:
    """Delete a student profile with administrator privileges."""
    service = StudentService(session, actor=current_user)
    deleted = await service.delete_student(student_id)

    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")

    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
