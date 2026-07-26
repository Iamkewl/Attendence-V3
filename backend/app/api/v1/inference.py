"""Inference API endpoints for enqueueing and tracking asynchronous AI pipeline tasks."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

import numpy as np
from celery import states
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image
from pydantic import ValidationError
from sqlalchemy import select

from app.api.deps import CurrentInstructorUser, CurrentUser
from app.core.database import get_session_factory
from app.domain.models import Student, User
from app.domain.schemas import (
    ImageTensorPayload,
    InferenceBatchRequest,
    InferenceTaskAccepted,
    InferenceTaskStatus,
    RecognitionDetection,
    RecognitionMatch,
    RecognitionPhotoResponse,
)
from app.infrastructure.triton import (
    TritonClientError,
)
from app.services.pipeline_service import process_inference_batch
from app.worker.celery_app import celery_app
from app.worker.tasks import run_inference_pipeline

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["Inference"])


def _read_positive_int_env(name: str, default: int) -> int:
    """Read an int env var; reject zero/negative/non-int values (default when unset)."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer.") from exc
    if value < 1:
        raise RuntimeError(f"Environment variable {name} must be a positive integer.")
    return value


_DEFAULT_MAX_FRAME_BYTES = 4 * 1024 * 1024  # 4 MiB raw tensor bytes; the issue's upper-bound example
_MAX_FRAME_BYTES = _read_positive_int_env("ATTENDANCE_MAX_FRAME_BYTES", _DEFAULT_MAX_FRAME_BYTES)


def _enqueue_inference_batch(
    payload: InferenceBatchRequest,
    *,
    use_priority_queue: bool,
) -> InferenceTaskAccepted:
    """Submit inference payload to Celery and return task metadata."""
    queue_name = "inference_priority" if use_priority_queue else "inference"
    routing_key = "inference.priority" if use_priority_queue else "inference.default"

    try:
        async_result = run_inference_pipeline.apply_async(
            args=[payload.model_dump(mode="json")],
            queue=queue_name,
            routing_key=routing_key,
        )
    except Exception as exc:
        LOGGER.exception("Failed to enqueue inference task.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference queue is unavailable. Please retry shortly.",
        ) from exc

    return InferenceTaskAccepted(
        task_id=async_result.id,
        state=states.PENDING,
        queued_at=datetime.now(tz=UTC),
        frame_count=len(payload.frames),
    )


@router.post(
    "/stream",
    response_model=InferenceTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue Stream Inference",
    description=(
        "Enqueue one raw frame stream payload as an asynchronous inference task. "
        "The uploaded file must contain raw uint8/float32 tensor bytes matching the declared shape."
    ),
)
async def enqueue_stream_inference(
    _: CurrentUser,
    frame_file: Annotated[UploadFile, File(description="Raw frame tensor bytes.")],
    frame_id: Annotated[str, Form(min_length=1, max_length=128)],
    width: Annotated[int, Form(ge=1, le=4096)],
    height: Annotated[int, Form(ge=1, le=4096)],
    channels: Annotated[int, Form(ge=1, le=4)] = 3,
    dtype: Annotated[str, Form(pattern="^(uint8|float32)$")] = "uint8",
    normalize: Annotated[bool, Form()] = True,
    request_id: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
    session_id: Annotated[UUID | None, Form()] = None,
    course_id: Annotated[UUID | None, Form()] = None,
    room_id: Annotated[UUID | None, Form()] = None,
    camera_id: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
    confidence_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.25,
    liveness_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
    include_embeddings: Annotated[bool, Form()] = False,
    priority: Annotated[bool, Form()] = False,
) -> InferenceTaskAccepted:
    """Accept one streamed frame and enqueue the asynchronous inference pipeline."""
    frame_bytes = await frame_file.read()
    if not frame_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded frame_file is empty.",
        )

    if len(frame_bytes) > _MAX_FRAME_BYTES:
        # An oversized declared shape (e.g. 4096x4096x4 float32 ~= 256 MiB)
        # would otherwise pass schema validation and force a worker OOM. Cap
        # at intake so the worker never allocates the array.
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Uploaded frame_file exceeds the maximum of {_MAX_FRAME_BYTES} bytes; "
                f"reduce the declared tensor shape or raise ATTENDANCE_MAX_FRAME_BYTES."
            ),
        )

    try:
        frame_payload = ImageTensorPayload(
            frame_id=frame_id,
            data_base64=base64.b64encode(frame_bytes).decode("ascii"),
            width=width,
            height=height,
            channels=channels,
            dtype=dtype,
            normalize=normalize,
            captured_at=datetime.now(tz=UTC),
        )
        batch_payload = InferenceBatchRequest(
            request_id=request_id,
            session_id=session_id,
            course_id=course_id,
            room_id=room_id,
            camera_id=camera_id,
            frames=[frame_payload],
            confidence_threshold=confidence_threshold,
            liveness_threshold=liveness_threshold,
            include_embeddings=include_embeddings,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid frame tensor payload.",
        ) from exc

    return _enqueue_inference_batch(batch_payload, use_priority_queue=priority)


@router.post(
    "/batch",
    response_model=InferenceTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue Batch Inference",
    description="Enqueue a batch of validated image tensor frames for asynchronous inference processing.",
)
async def enqueue_batch_inference(
    _: CurrentUser,
    payload: InferenceBatchRequest,
    priority: bool = False,
) -> InferenceTaskAccepted:
    """Accept a validated multi-frame payload and enqueue the asynchronous inference pipeline."""
    return _enqueue_inference_batch(payload, use_priority_queue=priority)


@router.get(
    "/tasks/{task_id}",
    response_model=InferenceTaskStatus,
    summary="Get Inference Task Status",
    description="Retrieve Celery task state and optional result payload for a previously enqueued inference task.",
)
async def get_inference_task_status(_: CurrentUser, task_id: str) -> InferenceTaskStatus:
    """Return execution state for a task ID from the Celery result backend."""
    try:
        task_result = celery_app.AsyncResult(task_id)
    except Exception as exc:
        LOGGER.exception("Failed to query Celery task status.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to query task status from result backend.",
        ) from exc

    state = task_result.state

    if state == states.SUCCESS:
        if isinstance(task_result.result, dict):
            sanitized = _strip_embeddings_from_task_result(task_result.result)
            return InferenceTaskStatus(task_id=task_id, state=state, result=sanitized)
        return InferenceTaskStatus(
            task_id=task_id,
            state=state,
            result={"value": task_result.result},
        )

    if state in {states.FAILURE, states.REVOKED}:
        return InferenceTaskStatus(task_id=task_id, state=state, error=str(task_result.result))

    return InferenceTaskStatus(task_id=task_id, state=state)


def _strip_embeddings_from_task_result(result: dict[str, object]) -> dict[str, object]:
    """Defense-in-depth: scrub face embeddings before returning task results without ownership checks."""
    sanitized: dict[str, object] = dict(result)
    items = sanitized.get("results")
    if isinstance(items, list):
        sanitized["results"] = [
            {**item, "embedding": None} if isinstance(item, dict) and "embedding" in item else item
            for item in items
        ]
    return sanitized


_MAX_PHOTO_BYTES = 10 * 1024 * 1024  # 10 MB


def _decode_photo_to_tensor(raw_bytes: bytes) -> Image.Image:
    """Decode JPEG/PNG bytes to RGB PIL and to a uint8 HxWx3 numpy array (CPU-bound).

    This is CPU-bound work (typically 200–800 ms for a 10 MB JPEG) and must
    NOT run on the FastAPI event loop — keeping it there stalls every other
    request, including /healthz, the WebSocket keepalive, and the realtime
    broadcast fan-out. The caller is expected to run this via asyncio.to_thread.
    """
    pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return pil_image


@router.post(
    "/photo",
    response_model=RecognitionPhotoResponse,
    summary="Synchronous Recognition Test",
    description=(
        "Decode an image, run full detection + recognition pipeline in-process, "
        "return matched faces. Does not persist sighting rows."
    ),
)
async def recognize_photo(
    _: CurrentInstructorUser,
    file: Annotated[UploadFile, File(description="JPEG or PNG image.")],
    course_id: Annotated[UUID | None, Form()] = None,
    confidence_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.25,
    liveness_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
) -> RecognitionPhotoResponse:
    """Run synchronous detection and recognition on a single uploaded photo."""
    raw_bytes = await file.read()

    if len(raw_bytes) > _MAX_PHOTO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image file exceeds maximum allowed size of 10 MB.",
        )

    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        pil_image = await asyncio.to_thread(_decode_photo_to_tensor, raw_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image decode failed: file is not a valid JPEG or PNG.",
        ) from exc

    width, height = pil_image.size
    frame_bytes = await asyncio.to_thread(lambda: np.asarray(pil_image, dtype=np.uint8).tobytes())

    now = datetime.now(tz=UTC)
    frame_payload = ImageTensorPayload(
        frame_id="photo_test",
        data_base64=base64.b64encode(frame_bytes).decode("ascii"),
        width=width,
        height=height,
        channels=3,
        dtype="uint8",
        normalize=True,
        captured_at=now,
    )
    batch_request = InferenceBatchRequest(
        course_id=course_id,
        frames=[frame_payload],
        confidence_threshold=confidence_threshold,
        liveness_threshold=liveness_threshold,
        include_embeddings=False,
    )

    try:
        pipeline_result = await process_inference_batch(batch_request)
    except TritonClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Triton unavailable: {exc}",
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected error in synchronous inference pipeline.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal pipeline error.",
        ) from exc

    raw_results: list[dict] = pipeline_result.get("results", [])

    # Batch-resolve student names and numbers for all matched student IDs.
    matched_student_ids = [
        UUID(item["student_id"])
        for item in raw_results
        if item.get("student_id") is not None
    ]

    student_info: dict[UUID, tuple[str, str]] = {}
    if matched_student_ids:
        session_factory = get_session_factory()
        async with session_factory() as session:
            rows = (
                await session.execute(
                    select(
                        Student.id,
                        Student.student_number,
                        Student.user_id,
                    ).where(Student.id.in_(matched_student_ids))
                )
            ).all()

            user_ids = [row.user_id for row in rows]
            user_rows = (
                await session.execute(
                    select(User.id, User.full_name).where(
                        User.id.in_(user_ids)
                    )
                )
            ).all()

            user_name_map: dict[UUID, str] = {row.id: row.full_name for row in user_rows}
            for row in rows:
                student_info[row.id] = (
                    user_name_map.get(row.user_id, ""),
                    row.student_number,
                )

    detections: list[RecognitionDetection] = []
    for item in raw_results:
        raw_student_id = item.get("student_id")
        match: RecognitionMatch | None = None
        if raw_student_id is not None:
            sid = UUID(raw_student_id)
            full_name, student_number = student_info.get(sid, ("", ""))
            match = RecognitionMatch(
                student_id=sid,
                student_full_name=full_name,
                student_number=student_number,
                cosine_similarity=float(item.get("cosine_similarity") or 0.0),
            )

        detections.append(
            RecognitionDetection(
                track_id=int(item["track_id"]),
                bbox=[float(v) for v in item["bbox"]],
                confidence=float(item["detection_score"]),
                liveness_score=float(item["liveness_score"]),
                is_live=bool(item["is_live"]),
                match=match,
            )
        )

    match_count = sum(1 for d in detections if d.match is not None)

    return RecognitionPhotoResponse(
        image_width=width,
        image_height=height,
        detection_count=len(detections),
        match_count=match_count,
        processed_at=now,
        detections=detections,
    )


__all__ = ["router"]
