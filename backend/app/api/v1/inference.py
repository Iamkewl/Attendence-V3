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
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from PIL import Image
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentIngestUser, CurrentUser
from app.core.database import get_async_session, get_session_factory
from app.core.security import get_redis_client
from app.domain.models import Student, User, UserRole
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
from app.services.audit_service import AuditEvent, GovernanceAction, emit
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

# Aggregate cap for /batch payloads. Without it the JSON endpoint admits up to
# max_frames x 256 MiB of declared tensor data — the same worker-OOM class
# ATT-013 describes for /stream, reachable with plain HTTP instead of an
# upload. Enforced twice: on the declared Content-Length before parsing (so a
# huge body is refused without buffering) and on the sum of per-frame declared
# tensor sizes after schema validation.
_DEFAULT_MAX_BATCH_BYTES = 64 * 1024 * 1024  # 64 MiB aggregate
_MAX_BATCH_BYTES = _read_positive_int_env("ATTENDANCE_MAX_BATCH_BYTES", _DEFAULT_MAX_BATCH_BYTES)

# Internal error messages are kept generic by default; operators can opt in
# to surfacing str(exception) on FAILURE/REVOKED task states for debugging.
_REVEAL_INTERNAL_ERRORS = os.getenv("ATTENDANCE_REVEAL_INTERNAL_ERRORS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

# Task ownership is stored in Redis at enqueue time and consulted by the
# status route. A 24h TTL covers the Celery result_expires window without
# tailing the result backend forever. The UUID-string task_id is the key
# suffix; collisions are bounded by the UUID v4 space.
_TASK_OWNER_TTL_SECONDS = _read_positive_int_env("ATTENDANCE_TASK_OWNER_TTL_SECONDS", 86_400)
_TASK_OWNER_KEY_PREFIX = "inference_task:owner"


def _task_read_auditing_enabled() -> bool:
    """Return whether TASK_READ governance events should be written (D7).

    Read PER CALL, never cached at import: TASK_READ auditing is OFF by
    default because the frontend polls this endpoint and the volume is
    unpredictable; operators opt in via ATTENDANCE_TASK_READ_AUDIT=true when
    a data-access-report requirement materializes.
    """
    return os.getenv("ATTENDANCE_TASK_READ_AUDIT", "").strip().lower() in {"1", "true", "yes"}


async def _set_task_owner(task_id: str, owner_id: UUID) -> None:
    """Record the user who enqueued a task so the status route can authorize reads.

    Best-effort: ownership cannot block enqueue — if Redis is unavailable the
    task is still queued; the status route will then return 404 to all callers
    (deny existence) rather than leak state. Logged, not raised.
    """
    try:
        client = await get_redis_client()
        await client.set(
            f"{_TASK_OWNER_KEY_PREFIX}:{task_id}",
            str(owner_id),
            ex=_TASK_OWNER_TTL_SECONDS,
        )
    except Exception:
        LOGGER.exception(
            "Failed to record inference task owner for task_id=%s; status route will deny reads.",
            task_id,
        )


async def _is_task_owner(task_id: str, user_id: UUID) -> bool:
    """Return True iff the caller is the recorded owner of ``task_id``.

    A missing owner key (TTL expired, Redis was unavailable at enqueue, or the
    task predates the ownership mechanism) is treated as *not owner* — the
    route then returns 404 to deny existence. Fail closed.
    """
    try:
        client = await get_redis_client()
        value = await client.get(f"{_TASK_OWNER_KEY_PREFIX}:{task_id}")
    except Exception:
        LOGGER.exception("Failed to query inference task owner for task_id=%s.", task_id)
        return False

    return bool(value) and value == str(user_id)


def _enqueue_inference_batch(
    payload: InferenceBatchRequest,
    *,
    use_priority_queue: bool,
) -> InferenceTaskAccepted:
    """Submit inference payload to Celery and return task metadata (sync part)."""
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
    current_user: CurrentIngestUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
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
    # Read in bounded chunks and abort as soon as the cap is exceeded so an
    # oversized chunked upload is rejected mid-stream instead of being fully
    # buffered in the API process first.
    chunks: list[bytes] = []
    received = 0
    while True:
        chunk = await frame_file.read(256 * 1024)
        if not chunk:
            break
        received += len(chunk)
        if received > _MAX_FRAME_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"Uploaded frame_file exceeds the maximum of {_MAX_FRAME_BYTES} bytes; "
                    f"reduce the declared tensor shape or raise ATTENDANCE_MAX_FRAME_BYTES."
                ),
            )
        chunks.append(chunk)
    frame_bytes = b"".join(chunks)
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
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
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
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid frame tensor payload.",
        ) from exc

    accepted = _enqueue_inference_batch(batch_payload, use_priority_queue=priority)
    await _set_task_owner(accepted.task_id, current_user.id)
    # Advisory INFERENCE_ENQUEUED (D1): log-and-continue. No IP stored —
    # routine domain events capture none (D4).
    await emit(
        session,
        AuditEvent(
            action=GovernanceAction.INFERENCE_ENQUEUED,
            entity_type="inference_task",
            actor_user_id=current_user.id,
            change_summary={"frame_count": len(batch_payload.frames), "source": "stream"},
        ),
        strict=False,
    )
    await session.commit()
    return accepted


@router.post(
    "/batch",
    response_model=InferenceTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue Batch Inference",
    description="Enqueue a batch of validated image tensor frames for asynchronous inference processing.",
)
async def enqueue_batch_inference(
    current_user: CurrentIngestUser,
    request: Request,
    payload: InferenceBatchRequest,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    priority: bool = False,
) -> InferenceTaskAccepted:
    """Accept a validated multi-frame payload and enqueue the asynchronous inference pipeline."""
    # Pre-parse guard: refuse oversized bodies before FastAPI buffers the JSON.
    declared_length = request.headers.get("content-length", "")
    if declared_length.isdigit() and int(declared_length) > _MAX_BATCH_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"Batch payload exceeds the maximum of {_MAX_BATCH_BYTES} bytes; "
                f"split the batch or raise ATTENDANCE_MAX_BATCH_BYTES."
            ),
        )

    # Post-parse accounting: per-frame and aggregate caps computed from the
    # declared tensor shapes (cheap arithmetic, no base64 decoding needed).
    total_declared = 0
    for frame in payload.frames:
        bytes_per_value = 4 if frame.dtype == "float32" else 1
        declared = frame.width * frame.height * frame.channels * bytes_per_value
        if declared > _MAX_FRAME_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"Frame '{frame.frame_id}' declares {declared} bytes which exceeds the "
                    f"per-frame maximum of {_MAX_FRAME_BYTES} bytes "
                    f"(ATTENDANCE_MAX_FRAME_BYTES)."
                ),
            )
        total_declared += declared
    if total_declared > _MAX_BATCH_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"Batch declares {total_declared} bytes across frames which exceeds the "
                f"aggregate maximum of {_MAX_BATCH_BYTES} bytes (ATTENDANCE_MAX_BATCH_BYTES)."
            ),
        )

    accepted = _enqueue_inference_batch(payload, use_priority_queue=priority)
    await _set_task_owner(accepted.task_id, current_user.id)
    # Advisory INFERENCE_ENQUEUED (D1): log-and-continue. No IP stored —
    # routine domain events capture none (D4).
    await emit(
        session,
        AuditEvent(
            action=GovernanceAction.INFERENCE_ENQUEUED,
            entity_type="inference_task",
            actor_user_id=current_user.id,
            change_summary={"frame_count": len(payload.frames), "source": "batch"},
        ),
        strict=False,
    )
    await session.commit()
    return accepted


@router.get(
    "/tasks/{task_id}",
    response_model=InferenceTaskStatus,
    summary="Get Inference Task Status",
    description="Retrieve Celery task state and optional result payload for a previously enqueued inference task.",
)
async def get_inference_task_status(
    current_user: CurrentUser,
    task_id: str,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> InferenceTaskStatus:
    """Return execution state for a task ID from the Celery result backend.

    Authorization: only the user who enqueued the task may read it (a per-task
    owner:{user_id} key is written in Redis at enqueue time). Administrators
    may read any task. A non-owner caller gets 404 (deny existence, not 403 —
    a 403 would leak the existence of an unknown task ID). If the owner record
    is missing (TTL expired, Redis was unavailable at enqueue), non-admin
    callers are denied; admins are still allowed to read.
    """
    try:
        task_result = celery_app.AsyncResult(task_id)
    except Exception as exc:
        LOGGER.exception("Failed to query Celery task status.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to query task status from result backend.",
        ) from exc

    state = task_result.state

    is_admin = current_user.role == UserRole.ADMIN
    if not is_admin and not await _is_task_owner(task_id, current_user.id):
        # Deny existence: a 404 cannot be distinguished from "task never
        # existed" by the caller.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No inference task found for that task_id.",
        )

    # TASK_READ (D7): defined in the vocabulary but OFF by default — frontend
    # polling makes volume unpredictable. Emissions are advisory and only for
    # AUTHORIZED reads; denials are not data-access events.
    if _task_read_auditing_enabled():
        await emit(
            session,
            AuditEvent(
                action=GovernanceAction.TASK_READ,
                entity_type="inference_task",
                actor_user_id=current_user.id,
                change_summary={"task_id": task_id},
            ),
            strict=False,
        )
        await session.commit()

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
        error = (
            str(task_result.result)
            if _REVEAL_INTERNAL_ERRORS
            else "Inference task failed; see server logs for details."
        )
        return InferenceTaskStatus(task_id=task_id, state=state, error=error)

    return InferenceTaskStatus(task_id=task_id, state=state)


_EMBEDDING_DERIVED_KEYS: tuple[str, ...] = ("embedding", "identity", "matched_embedding_id")


def _strip_embeddings_from_task_result(result: dict[str, object]) -> dict[str, object]:
    """Defense-in-depth: scrub face embedding and embedding-derived pseudonyms.

    Strips, per detection in ``results[]``:
      - ``embedding``: the raw 512-D vector (the original concern).
      - ``identity``: a 16-hex sha256-truncated digest of the embedding. This is
        a stable pseudonymous identifier of an enrolled student — FERPA/BIPA
        treats it as PII.
      - ``matched_embedding_id``: the UUID of the matched StudentEmbedding row,
        a stable pseudonym of an enrolled student's template.

    All three are removed (set to ``None`` only if present in the source dict,
    so as not to *add* keys the worker never wrote) before any task result is
    returned to the API client. The realtime broadcast
    (_publish_live_sighting_event in attendance_service.py) still ships
    ``embedding_reference`` to subscribers — fixing that is out of this issue's
    file scope; flagged in the B02 PR body.
    """
    sanitized: dict[str, object] = dict(result)
    items = sanitized.get("results")
    if isinstance(items, list):
        def _scrub(item: object) -> object:
            if not isinstance(item, dict):
                return item
            stripped = dict(item)
            for key in _EMBEDDING_DERIVED_KEYS:
                if key in stripped:
                    stripped[key] = None
            return stripped

        sanitized["results"] = [_scrub(item) for item in items]
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
    current_user: CurrentIngestUser,
    file: Annotated[UploadFile, File(description="JPEG or PNG image.")],
    session: Annotated[AsyncSession, Depends(get_async_session)],
    course_id: Annotated[UUID | None, Form()] = None,
    confidence_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.25,
    liveness_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
) -> RecognitionPhotoResponse:
    """Run synchronous detection and recognition on a single uploaded photo."""
    raw_bytes = await file.read()

    if len(raw_bytes) > _MAX_PHOTO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
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

    # Advisory RECOGNITION_RUN (D1): one row per synchronous recognition run
    # with the match count only — no embedding material, no IP (D4).
    await emit(
        session,
        AuditEvent(
            action=GovernanceAction.RECOGNITION_RUN,
            entity_type="inference_task",
            actor_user_id=current_user.id,
            change_summary={
                "match_count": match_count,
                "detection_count": len(detections),
                "course_id": str(course_id) if course_id else None,
            },
        ),
        strict=False,
    )
    await session.commit()

    return RecognitionPhotoResponse(
        image_width=width,
        image_height=height,
        detection_count=len(detections),
        match_count=match_count,
        processed_at=now,
        detections=detections,
    )


__all__ = ["router"]
