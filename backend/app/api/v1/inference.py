"""Inference API endpoints for enqueueing and tracking asynchronous AI pipeline tasks."""

from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from celery import states
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.api.deps import CurrentUser
from app.domain.schemas import (
    ImageTensorPayload,
    InferenceBatchRequest,
    InferenceTaskAccepted,
    InferenceTaskStatus,
)
from app.worker.celery_app import celery_app
from app.worker.tasks import run_inference_pipeline


LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["Inference"])


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
            return InferenceTaskStatus(task_id=task_id, state=state, result=task_result.result)
        return InferenceTaskStatus(
            task_id=task_id,
            state=state,
            result={"value": task_result.result},
        )

    if state in {states.FAILURE, states.REVOKED}:
        return InferenceTaskStatus(task_id=task_id, state=state, error=str(task_result.result))

    return InferenceTaskStatus(task_id=task_id, state=state)


__all__ = ["router"]
