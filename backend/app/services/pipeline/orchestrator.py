"""Async pipeline orchestrator: process_inference_batch and extract_enrollment_embedding."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np

from app.domain.schemas import InferenceBatchRequest
from app.infrastructure.triton import TritonGrpcClient, get_triton_client

from .detection import Detection, _parse_detections
from .embedding import _decode_embeddings, _identity_from_embedding
from .frame import _crop_face, _decode_frame_tensor, _frame_to_model_input, _prepare_face_batch, _resize_nearest
from .liveness import _decode_liveness_scores
from .matching import _resolve_vector_matches
from .settings import PipelineSettings, get_pipeline_settings
from .tracking import TrackedDetection, _track_detections


LOGGER = logging.getLogger(__name__)


class NoFaceDetectedError(ValueError):
    """Raised when enrollment extraction is asked to require a detection.

    The bulk importer passes ``require_detection=True`` so an image with
    zero YOLO detections becomes an explicit NO_FACE reject instead of the
    legacy whole-frame-resize fallback, which would happily embed a
    face-less photo as a garbage template. Default behavior (fallback,
    no raise) is unchanged for the API route.
    """


def _select_output_tensor(
    outputs: dict[str, np.ndarray],
    *,
    preferred_name: str | None,
    use_last_if_multiple: bool,
) -> np.ndarray:
    """Select an output tensor from Triton response using deterministic fallback rules."""
    if preferred_name is not None and preferred_name in outputs:
        return outputs[preferred_name]

    if use_last_if_multiple and len(outputs) > 1:
        return list(outputs.values())[-1]

    return next(iter(outputs.values()))


def _build_lvface_output_names(settings: PipelineSettings) -> list[str] | None:
    """Build the lvface output_names list for a single batched infer call.

    When both liveness and embedding output names are configured, request both
    explicitly so a single Triton RPC returns the merged dict. Otherwise return
    None so Triton returns all model outputs and the order-based fallback in
    `_select_output_tensor` continues to apply.
    """
    if settings.lvface_liveness_output_name and settings.lvface_embedding_output_name:
        return [
            settings.lvface_liveness_output_name,
            settings.lvface_embedding_output_name,
        ]
    return None


async def extract_enrollment_embedding(
    face_tensor: np.ndarray,
    *,
    triton_client: TritonGrpcClient | None = None,
    require_detection: bool = False,
) -> tuple[np.ndarray, float]:
    """Extract one 512D normalized embedding plus quality proxy from a face image tensor.

    ``require_detection=True`` (used by the bulk enrollment importer)
    raises :class:`NoFaceDetectedError` when the detector finds zero faces
    instead of falling back to resizing the whole frame; the default
    ``False`` preserves the historical API-route behavior exactly.
    """
    settings = get_pipeline_settings()
    client = triton_client or get_triton_client()

    if face_tensor.ndim != 3:
        raise ValueError("Enrollment image tensor must be a rank-3 HWC tensor.")

    normalized_tensor = np.ascontiguousarray(face_tensor.astype(np.float32, copy=False))
    detector_input = _frame_to_model_input(normalized_tensor)
    detection_outputs = await client.infer_fp32_async(
        model_name=settings.yolo_model_name,
        tensors={settings.yolo_input_name: detector_input},
        output_names=[settings.yolo_output_name] if settings.yolo_output_name else None,
    )
    detections = _parse_detections(
        frame_index=0,
        frame_id="enroll_frame",
        outputs=detection_outputs,
        confidence_threshold=0.25,
        frame_height=normalized_tensor.shape[0],
        frame_width=normalized_tensor.shape[1],
        preferred_output_name=settings.yolo_output_name,
        bbox_format=settings.yolo_bbox_format,
        bbox_normalized=settings.yolo_bbox_normalized,
    )

    if detections:
        best_detection = max(detections, key=lambda candidate: candidate.score)
        try:
            aligned_face = _crop_face(
                normalized_tensor,
                best_detection.bbox,
                crop_size=settings.face_crop_size,
            )
        except ValueError:
            aligned_face = _resize_nearest(
                normalized_tensor,
                settings.face_crop_size,
                settings.face_crop_size,
            )
    else:
        if require_detection:
            raise NoFaceDetectedError("No face detected in enrollment image.")
        aligned_face = _resize_nearest(
            normalized_tensor,
            settings.face_crop_size,
            settings.face_crop_size,
        )

    face_batch = _prepare_face_batch([aligned_face])

    lvface_outputs = await client.infer_fp32_async(
        model_name=settings.lvface_model_name,
        tensors={settings.lvface_input_name: face_batch},
        output_names=_build_lvface_output_names(settings),
    )
    liveness_tensor = _select_output_tensor(
        lvface_outputs,
        preferred_name=settings.lvface_liveness_output_name,
        use_last_if_multiple=False,
    )
    quality_score = _decode_liveness_scores(liveness_tensor, expected_count=1)[0]

    embedding_tensor = _select_output_tensor(
        lvface_outputs,
        preferred_name=settings.lvface_embedding_output_name,
        use_last_if_multiple=True,
    )
    embedding = _decode_embeddings(embedding_tensor, expected_count=1)[0]

    return embedding, float(np.clip(quality_score, 0.0, 1.0))


async def analyze_enrollment_frame(
    face_tensor: np.ndarray,
    *,
    triton_client: TritonGrpcClient | None = None,
) -> tuple[list[Detection], float]:
    """Run detection + liveness-quality scoring WITHOUT computing an embedding.

    Powers the live enrollment preview endpoint (``POST /students/enroll/preview``):
    the caller needs the raw detections (count, largest bbox) and the same
    quality proxy the enroll flow uses, but must not materialize biometric
    embeddings on a 600ms-cadence polling path.

    ``require_detection`` semantics mirror :func:`extract_enrollment_embedding`:
    zero YOLO detections raises :class:`NoFaceDetectedError` instead of the
    whole-frame-resize fallback. The preview ROUTE catches that and reports it
    in-band as JSON (``detected=false``) rather than surfacing an HTTP error.
    """
    settings = get_pipeline_settings()
    client = triton_client or get_triton_client()

    if face_tensor.ndim != 3:
        raise ValueError("Enrollment image tensor must be a rank-3 HWC tensor.")

    normalized_tensor = np.ascontiguousarray(face_tensor.astype(np.float32, copy=False))
    detector_input = _frame_to_model_input(normalized_tensor)
    detection_outputs = await client.infer_fp32_async(
        model_name=settings.yolo_model_name,
        tensors={settings.yolo_input_name: detector_input},
        output_names=[settings.yolo_output_name] if settings.yolo_output_name else None,
    )
    detections = _parse_detections(
        frame_index=0,
        frame_id="enroll_preview",
        outputs=detection_outputs,
        confidence_threshold=0.25,
        frame_height=normalized_tensor.shape[0],
        frame_width=normalized_tensor.shape[1],
        preferred_output_name=settings.yolo_output_name,
        bbox_format=settings.yolo_bbox_format,
        bbox_normalized=settings.yolo_bbox_normalized,
    )

    if not detections:
        raise NoFaceDetectedError("No face detected in enrollment preview frame.")

    best_detection = max(detections, key=lambda candidate: candidate.score)
    try:
        aligned_face = _crop_face(
            normalized_tensor,
            best_detection.bbox,
            crop_size=settings.face_crop_size,
        )
    except ValueError:
        aligned_face = _resize_nearest(
            normalized_tensor,
            settings.face_crop_size,
            settings.face_crop_size,
        )

    face_batch = _prepare_face_batch([aligned_face])
    lvface_outputs = await client.infer_fp32_async(
        model_name=settings.lvface_model_name,
        tensors={settings.lvface_input_name: face_batch},
        output_names=_build_lvface_output_names(settings),
    )
    liveness_tensor = _select_output_tensor(
        lvface_outputs,
        preferred_name=settings.lvface_liveness_output_name,
        use_last_if_multiple=False,
    )
    quality_score = _decode_liveness_scores(liveness_tensor, expected_count=1)[0]

    return detections, float(np.clip(quality_score, 0.0, 1.0))


async def process_inference_batch(
    request: InferenceBatchRequest,
    *,
    triton_client: TritonGrpcClient | None = None,
) -> dict[str, Any]:
    """Execute full frame pipeline and resolve strict-threshold identity via pgvector search."""
    settings = get_pipeline_settings()
    client = triton_client or get_triton_client()

    frames = [_decode_frame_tensor(frame) for frame in request.frames]

    detections_by_frame: list[list[Detection]] = []
    for frame_index, (frame_payload, frame_tensor) in enumerate(zip(request.frames, frames, strict=True)):
        detector_input = _frame_to_model_input(frame_tensor)
        detection_outputs = await client.infer_fp32_async(
            model_name=settings.yolo_model_name,
            tensors={settings.yolo_input_name: detector_input},
            output_names=[settings.yolo_output_name] if settings.yolo_output_name else None,
            request_id=request.request_id,
        )

        parsed_detections = _parse_detections(
            frame_index=frame_index,
            frame_id=frame_payload.frame_id,
            outputs=detection_outputs,
            confidence_threshold=request.confidence_threshold,
            frame_height=frame_tensor.shape[0],
            frame_width=frame_tensor.shape[1],
            preferred_output_name=settings.yolo_output_name,
            bbox_format=settings.yolo_bbox_format,
            bbox_normalized=settings.yolo_bbox_normalized,
        )
        detections_by_frame.append(parsed_detections)

    tracked_detections = _track_detections(
        detections_by_frame,
        max_link_distance=settings.tracking_distance_threshold,
    )

    if not tracked_detections:
        return {
            "request_id": request.request_id,
            "session_id": str(request.session_id) if request.session_id else None,
            "course_id": str(request.course_id) if request.course_id else None,
            "room_id": str(request.room_id) if request.room_id else None,
            "camera_id": request.camera_id,
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "frame_count": len(frames),
            "detection_count": 0,
            "track_count": 0,
            "results": [],
        }

    tracked_with_crops: list[tuple[TrackedDetection, np.ndarray]] = []
    for tracked_detection in tracked_detections:
        frame_tensor = frames[tracked_detection.frame_index]
        try:
            face_crop = _crop_face(
                frame_tensor,
                tracked_detection.bbox,
                crop_size=settings.face_crop_size,
            )
        except ValueError:
            continue

        tracked_with_crops.append((tracked_detection, face_crop))

    if not tracked_with_crops:
        return {
            "request_id": request.request_id,
            "session_id": str(request.session_id) if request.session_id else None,
            "course_id": str(request.course_id) if request.course_id else None,
            "room_id": str(request.room_id) if request.room_id else None,
            "camera_id": request.camera_id,
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "frame_count": len(frames),
            "detection_count": len(tracked_detections),
            "track_count": len({item.track_id for item in tracked_detections}),
            "results": [],
        }

    tracked_items = [item[0] for item in tracked_with_crops]
    face_crops = [item[1] for item in tracked_with_crops]
    face_batch = _prepare_face_batch(face_crops)

    lvface_outputs = await client.infer_fp32_async(
        model_name=settings.lvface_model_name,
        tensors={settings.lvface_input_name: face_batch},
        output_names=_build_lvface_output_names(settings),
        request_id=request.request_id,
    )
    liveness_tensor = _select_output_tensor(
        lvface_outputs,
        preferred_name=settings.lvface_liveness_output_name,
        use_last_if_multiple=False,
    )
    liveness_scores = _decode_liveness_scores(liveness_tensor, expected_count=len(tracked_items))

    embedding_tensor = _select_output_tensor(
        lvface_outputs,
        preferred_name=settings.lvface_embedding_output_name,
        use_last_if_multiple=True,
    )
    embeddings = _decode_embeddings(embedding_tensor, expected_count=len(tracked_items))
    embedding_matches = await _resolve_vector_matches(embeddings)

    results: list[dict[str, Any]] = []
    for tracked_item, liveness_score, embedding, embedding_match in zip(
        tracked_items,
        liveness_scores,
        embeddings,
        embedding_matches,
        strict=True,
    ):
        frame_payload = request.frames[tracked_item.frame_index]
        student_id = str(embedding_match.student_id) if embedding_match.student_id is not None else None
        captured_at = None
        if frame_payload.captured_at is not None:
            if frame_payload.captured_at.tzinfo is None:
                captured_at = frame_payload.captured_at.replace(tzinfo=UTC).isoformat()
            else:
                captured_at = frame_payload.captured_at.astimezone(UTC).isoformat()

        serialized_result: dict[str, Any] = {
            "frame_index": tracked_item.frame_index,
            "frame_id": tracked_item.frame_id,
            "track_id": tracked_item.track_id,
            "student_id": student_id,
            "captured_at": captured_at,
            "class_id": tracked_item.class_id,
            "bbox": [float(round(value, 3)) for value in tracked_item.bbox],
            "detection_score": float(round(tracked_item.score, 6)),
            "liveness_score": float(round(liveness_score, 6)),
            "is_live": liveness_score >= request.liveness_threshold,
            "is_match": student_id is not None,
            "cosine_similarity": (
                float(round(embedding_match.cosine_similarity, 6))
                if embedding_match.cosine_similarity is not None
                else None
            ),
            "matched_embedding_id": (
                str(embedding_match.embedding_id) if embedding_match.embedding_id is not None else None
            ),
            "match_threshold": settings.match_threshold,
            "identity": _identity_from_embedding(embedding),
        }

        if request.include_embeddings:
            serialized_result["embedding"] = [float(value) for value in embedding.tolist()]

        results.append(serialized_result)

    return {
        "request_id": request.request_id,
        "session_id": str(request.session_id) if request.session_id else None,
        "course_id": str(request.course_id) if request.course_id else None,
        "room_id": str(request.room_id) if request.room_id else None,
        "camera_id": request.camera_id,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "frame_count": len(frames),
        "detection_count": len(tracked_detections),
        "track_count": len({item.track_id for item in tracked_detections}),
        "live_count": sum(1 for item in results if item["is_live"]),
        "results": results,
    }
