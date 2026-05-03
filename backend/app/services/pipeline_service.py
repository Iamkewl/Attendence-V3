"""Inference pipeline service chaining detection, tracking, liveness, and recognition."""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session_factory
from app.domain.models import Student, StudentEmbedding
from app.domain.schemas import InferenceBatchRequest, ImageTensorPayload
from app.worker.triton_client import (
    TritonGrpcClient,
    get_triton_client,
)


LOGGER = logging.getLogger(__name__)

STRICT_SIMILARITY_THRESHOLD = 0.85


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """Runtime settings for the asynchronous AI inference pipeline."""

    yolo_model_name: str
    yolo_input_name: str
    yolo_output_name: str | None
    lvface_model_name: str
    lvface_input_name: str
    lvface_liveness_output_name: str | None
    lvface_embedding_output_name: str | None
    face_crop_size: int
    tracking_distance_threshold: float


@dataclass(frozen=True, slots=True)
class Detection:
    """Single-frame object detection candidate returned from detector post-processing."""

    frame_index: int
    frame_id: str
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int


@dataclass(frozen=True, slots=True)
class TrackedDetection(Detection):
    """Detection enriched with temporal track identity."""

    track_id: int


@dataclass(frozen=True, slots=True)
class EmbeddingMatch:
    """Represents the nearest persisted enrollment template for a live embedding."""

    student_id: UUID | None
    embedding_id: UUID | None
    cosine_similarity: float | None


def _read_int_env(name: str, default: int, *, min_value: int) -> int:
    """Read integer environment configuration with explicit lower-bound validation."""
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer.") from exc

    if value < min_value:
        raise RuntimeError(
            f"Environment variable {name} must be greater than or equal to {min_value}."
        )

    return value


def _read_float_env(name: str, default: float, *, min_value: float) -> float:
    """Read floating-point environment configuration with lower-bound validation."""
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a floating-point number.") from exc

    if value < min_value:
        raise RuntimeError(
            f"Environment variable {name} must be greater than or equal to {min_value}."
        )

    return value


def _read_str_env(name: str, default: str) -> str:
    """Read string environment configuration and return default when unset."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _read_optional_str_env(name: str) -> str | None:
    """Read optional string environment configuration and normalize empty values to None."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


@lru_cache(maxsize=1)
def get_pipeline_settings() -> PipelineSettings:
    """Return cached pipeline runtime settings from environment variables."""
    return PipelineSettings(
        yolo_model_name=_read_str_env("ATTENDANCE_TRITON_YOLO_MODEL_NAME", "yolov12"),
        yolo_input_name=_read_str_env("ATTENDANCE_TRITON_YOLO_INPUT_NAME", "INPUT__0"),
        yolo_output_name=_read_optional_str_env("ATTENDANCE_TRITON_YOLO_OUTPUT_NAME"),
        lvface_model_name=_read_str_env("ATTENDANCE_TRITON_LVFACE_MODEL_NAME", "lvface"),
        lvface_input_name=_read_str_env("ATTENDANCE_TRITON_LVFACE_INPUT_NAME", "INPUT__0"),
        lvface_liveness_output_name=_read_optional_str_env(
            "ATTENDANCE_TRITON_LVFACE_LIVENESS_OUTPUT_NAME"
        ),
        lvface_embedding_output_name=_read_optional_str_env(
            "ATTENDANCE_TRITON_LVFACE_EMBEDDING_OUTPUT_NAME"
        ),
        face_crop_size=_read_int_env("ATTENDANCE_PIPELINE_FACE_CROP_SIZE", 112, min_value=32),
        tracking_distance_threshold=_read_float_env(
            "ATTENDANCE_PIPELINE_TRACKING_DISTANCE_THRESHOLD",
            96.0,
            min_value=1.0,
        ),
    )


def _decode_frame_tensor(frame: ImageTensorPayload) -> np.ndarray:
    """Decode a base64-encoded frame payload into an HWC float32 tensor."""
    try:
        frame_bytes = base64.b64decode(frame.data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Frame {frame.frame_id} contains invalid base64 data.") from exc

    if frame.dtype == "uint8":
        decoded = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            frame.height,
            frame.width,
            frame.channels,
        )
        tensor = decoded.astype(np.float32)
        if frame.normalize:
            tensor /= 255.0
    else:
        tensor = np.frombuffer(frame_bytes, dtype=np.float32).reshape(
            frame.height,
            frame.width,
            frame.channels,
        )
        tensor = tensor.astype(np.float32, copy=False)

    return np.ascontiguousarray(tensor)


def _frame_to_model_input(frame_tensor: np.ndarray) -> np.ndarray:
    """Convert HWC frame tensor to NCHW batch tensor for Triton model input."""
    if frame_tensor.ndim != 3:
        raise ValueError("Input frame tensor must be a rank-3 HWC tensor.")

    channels = frame_tensor.shape[2]
    if channels not in {1, 3, 4}:
        raise ValueError(f"Unsupported channel count {channels}; expected 1, 3, or 4.")

    if channels == 4:
        frame_tensor = frame_tensor[:, :, :3]

    if channels == 1:
        frame_tensor = np.repeat(frame_tensor, 3, axis=2)

    nchw = np.transpose(frame_tensor, (2, 0, 1))[np.newaxis, :, :, :]
    return np.ascontiguousarray(nchw, dtype=np.float32)


def _extract_detection_rows(raw_tensor: np.ndarray) -> np.ndarray:
    """Normalize detector output tensors into [num_detections, features] shape."""
    tensor = np.asarray(raw_tensor, dtype=np.float32)

    if tensor.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    if tensor.ndim == 4:
        tensor = tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])

    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = tensor[0]
        else:
            tensor = tensor.reshape(-1, tensor.shape[-1])

    if tensor.ndim == 1:
        if tensor.shape[0] < 6:
            return np.empty((0, 6), dtype=np.float32)
        feature_count = 6
        trimmed_size = tensor.shape[0] - (tensor.shape[0] % feature_count)
        tensor = tensor[:trimmed_size].reshape(-1, feature_count)

    if tensor.ndim != 2:
        return np.empty((0, 6), dtype=np.float32)

    if tensor.shape[1] < 6 and tensor.shape[0] >= 6:
        tensor = tensor.T

    if tensor.shape[1] > 256 and tensor.shape[0] <= 256:
        tensor = tensor.T

    if tensor.shape[1] < 5:
        return np.empty((0, 6), dtype=np.float32)

    return tensor


def _decode_bbox(
    row: np.ndarray,
    *,
    frame_height: int,
    frame_width: int,
) -> tuple[float, float, float, float]:
    """Decode bounding box row into clipped pixel-space x1,y1,x2,y2 coordinates."""
    x0, y0, x1, y1 = [float(value) for value in row[:4]]

    if x1 <= x0 or y1 <= y0:
        center_x, center_y, width, height = x0, y0, x1, y1
        x0 = center_x - (width / 2.0)
        y0 = center_y - (height / 2.0)
        x1 = center_x + (width / 2.0)
        y1 = center_y + (height / 2.0)

    max_coordinate = max(abs(x0), abs(y0), abs(x1), abs(y1))
    if max_coordinate <= 1.5:
        x0 *= frame_width
        y0 *= frame_height
        x1 *= frame_width
        y1 *= frame_height

    x0 = float(np.clip(x0, 0.0, frame_width - 1.0))
    y0 = float(np.clip(y0, 0.0, frame_height - 1.0))
    x1 = float(np.clip(x1, 0.0, frame_width - 1.0))
    y1 = float(np.clip(y1, 0.0, frame_height - 1.0))

    if x1 <= x0 or y1 <= y0:
        raise ValueError("Decoded bounding box is invalid after clipping.")

    return x0, y0, x1, y1


def _parse_detections(
    *,
    frame_index: int,
    frame_id: str,
    outputs: dict[str, np.ndarray],
    confidence_threshold: float,
    frame_height: int,
    frame_width: int,
    preferred_output_name: str | None,
) -> list[Detection]:
    """Parse detector outputs into normalized detection objects for a single frame."""
    if not outputs:
        return []

    if preferred_output_name is not None and preferred_output_name in outputs:
        detection_tensor = outputs[preferred_output_name]
    else:
        detection_tensor = next(iter(outputs.values()))

    rows = _extract_detection_rows(detection_tensor)
    detections: list[Detection] = []

    for row in rows:
        if row.shape[0] < 5:
            continue

        if row.shape[0] == 6:
            class_index = int(max(round(float(row[5])), 0))
            score = float(row[4])
        elif row.shape[0] > 6:
            class_logits = row[5:]
            if class_logits.size > 0:
                if np.any(class_logits < 0.0) or np.any(class_logits > 1.0):
                    stabilized = class_logits - float(np.max(class_logits))
                    exp_values = np.exp(stabilized)
                    denominator = float(np.sum(exp_values))
                    if denominator <= 0.0:
                        continue
                    class_probs = exp_values / denominator
                else:
                    class_probs = class_logits

                class_index = int(np.argmax(class_probs))
                class_confidence = float(class_probs[class_index])
                score = float(row[4] * class_confidence)
            else:
                class_index = 0
                score = float(row[4])
        else:
            class_index = 0
            score = float(row[4])

        if score < confidence_threshold:
            continue

        try:
            bbox = _decode_bbox(
                row,
                frame_height=frame_height,
                frame_width=frame_width,
            )
        except ValueError:
            continue

        detections.append(
            Detection(
                frame_index=frame_index,
                frame_id=frame_id,
                bbox=bbox,
                score=score,
                class_id=class_index,
            )
        )

    return detections


def _bbox_centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return bbox centroid coordinates for nearest-neighbor tracking."""
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _track_detections(
    detections_by_frame: list[list[Detection]],
    *,
    max_link_distance: float,
) -> list[TrackedDetection]:
    """Assign deterministic track IDs by nearest-centroid temporal association."""
    previous_tracks: dict[int, tuple[float, float]] = {}
    tracked_results: list[TrackedDetection] = []
    next_track_id = 1

    for frame_detections in detections_by_frame:
        available_previous_tracks = set(previous_tracks.keys())
        current_tracks: dict[int, tuple[float, float]] = {}

        for detection in sorted(frame_detections, key=lambda item: item.score, reverse=True):
            centroid = _bbox_centroid(detection.bbox)
            selected_track: int | None = None
            best_distance = float("inf")

            for candidate_track in available_previous_tracks:
                candidate_centroid = previous_tracks[candidate_track]
                distance = math.dist(centroid, candidate_centroid)
                if distance <= max_link_distance and distance < best_distance:
                    best_distance = distance
                    selected_track = candidate_track

            if selected_track is None:
                selected_track = next_track_id
                next_track_id += 1
            else:
                available_previous_tracks.remove(selected_track)

            current_tracks[selected_track] = centroid
            tracked_results.append(
                TrackedDetection(
                    track_id=selected_track,
                    frame_index=detection.frame_index,
                    frame_id=detection.frame_id,
                    bbox=detection.bbox,
                    score=detection.score,
                    class_id=detection.class_id,
                )
            )

        previous_tracks = current_tracks

    return tracked_results


def _resize_nearest(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Resize an HWC image tensor with nearest-neighbor sampling using NumPy indexing."""
    source_height, source_width = image.shape[:2]
    y_indices = np.linspace(0, source_height - 1, target_height, dtype=np.float32).round().astype(int)
    x_indices = np.linspace(0, source_width - 1, target_width, dtype=np.float32).round().astype(int)

    y_indices = np.clip(y_indices, 0, source_height - 1)
    x_indices = np.clip(x_indices, 0, source_width - 1)

    return image[y_indices][:, x_indices, :]


def _crop_face(
    frame_tensor: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    crop_size: int,
) -> np.ndarray:
    """Crop and normalize a face ROI to a fixed-size tensor for LVFace inference."""
    x0, y0, x1, y1 = bbox
    x0_i = int(max(math.floor(x0), 0))
    y0_i = int(max(math.floor(y0), 0))
    x1_i = int(min(math.ceil(x1), frame_tensor.shape[1]))
    y1_i = int(min(math.ceil(y1), frame_tensor.shape[0]))

    if x1_i <= x0_i or y1_i <= y0_i:
        raise ValueError("Face crop is empty after clipping.")

    face_crop = frame_tensor[y0_i:y1_i, x0_i:x1_i, :]
    resized = _resize_nearest(face_crop, crop_size, crop_size)
    return np.ascontiguousarray(resized.astype(np.float32, copy=False))


def _prepare_face_batch(face_crops: list[np.ndarray]) -> np.ndarray:
    """Convert face crops into an NCHW FP32 batch tensor for Triton inference."""
    if not face_crops:
        raise ValueError("Cannot build face batch from an empty list.")

    batch_hwc = np.stack(face_crops, axis=0).astype(np.float32, copy=False)
    batch_nchw = np.transpose(batch_hwc, (0, 3, 1, 2))
    return np.ascontiguousarray(batch_nchw, dtype=np.float32)


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


def _decode_liveness_scores(output_tensor: np.ndarray, expected_count: int) -> list[float]:
    """Decode model outputs into per-face liveness probabilities in [0, 1]."""
    tensor = np.asarray(output_tensor, dtype=np.float32)

    if tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)

    if tensor.shape[0] != expected_count and tensor.ndim == 2 and tensor.shape[1] == expected_count:
        tensor = tensor.T

    if tensor.shape[0] != expected_count:
        raise ValueError(
            "Liveness output batch size does not match requested face batch size."
        )

    scores: list[float] = []
    for row in tensor:
        if row.shape[0] == 1:
            value = float(row[0])
            if value < 0.0 or value > 1.0:
                value = 1.0 / (1.0 + math.exp(-value))
            scores.append(float(np.clip(value, 0.0, 1.0)))
            continue

        logits = row[:2]
        stabilized = logits - float(np.max(logits))
        exp_values = np.exp(stabilized)
        denominator = float(np.sum(exp_values))
        probability_live = 0.0 if denominator <= 0.0 else float(exp_values[1] / denominator)
        scores.append(float(np.clip(probability_live, 0.0, 1.0)))

    return scores


def _decode_embeddings(output_tensor: np.ndarray, expected_count: int) -> list[np.ndarray]:
    """Decode output tensor into L2-normalized embedding vectors per tracked face."""
    tensor = np.asarray(output_tensor, dtype=np.float32)

    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)

    if tensor.shape[0] != expected_count and tensor.ndim == 2 and tensor.shape[1] == expected_count:
        tensor = tensor.T

    if tensor.shape[0] != expected_count:
        raise ValueError(
            "Recognition output batch size does not match requested face batch size."
        )

    embeddings: list[np.ndarray] = []
    for row in tensor:
        norm = float(np.linalg.norm(row))
        if norm <= 1e-8:
            normalized = np.zeros_like(row, dtype=np.float32)
        else:
            normalized = (row / norm).astype(np.float32)
        embeddings.append(normalized)

    return embeddings


def extract_enrollment_embedding(
    face_tensor: np.ndarray,
    *,
    triton_client: TritonGrpcClient | None = None,
) -> tuple[np.ndarray, float]:
    """Extract one 512D normalized embedding plus quality proxy from a face image tensor."""
    settings = get_pipeline_settings()
    client = triton_client or get_triton_client()

    if face_tensor.ndim != 3:
        raise ValueError("Enrollment image tensor must be a rank-3 HWC tensor.")

    normalized_tensor = np.ascontiguousarray(face_tensor.astype(np.float32, copy=False))
    detector_input = _frame_to_model_input(normalized_tensor)
    detection_outputs = client.infer_fp32(
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
        aligned_face = _resize_nearest(
            normalized_tensor,
            settings.face_crop_size,
            settings.face_crop_size,
        )

    face_batch = _prepare_face_batch([aligned_face])

    liveness_outputs = client.infer_fp32(
        model_name=settings.lvface_model_name,
        tensors={settings.lvface_input_name: face_batch},
        output_names=(
            [settings.lvface_liveness_output_name] if settings.lvface_liveness_output_name else None
        ),
    )
    liveness_tensor = _select_output_tensor(
        liveness_outputs,
        preferred_name=settings.lvface_liveness_output_name,
        use_last_if_multiple=False,
    )
    quality_score = _decode_liveness_scores(liveness_tensor, expected_count=1)[0]

    recognition_outputs = client.infer_fp32(
        model_name=settings.lvface_model_name,
        tensors={settings.lvface_input_name: face_batch},
        output_names=(
            [settings.lvface_embedding_output_name]
            if settings.lvface_embedding_output_name
            else None
        ),
    )
    embedding_tensor = _select_output_tensor(
        recognition_outputs,
        preferred_name=settings.lvface_embedding_output_name,
        use_last_if_multiple=True,
    )
    embedding = _decode_embeddings(embedding_tensor, expected_count=1)[0]

    return embedding, float(np.clip(quality_score, 0.0, 1.0))


async def _resolve_nearest_embedding_match(
    session: AsyncSession,
    embedding: np.ndarray,
) -> EmbeddingMatch:
    """Resolve nearest active enrollment template using pgvector cosine distance."""
    vector = [float(value) for value in embedding.tolist()]
    distance_expr = StudentEmbedding.embedding.cosine_distance(vector)

    nearest_row = (
        await session.execute(
            select(
                StudentEmbedding.student_id,
                StudentEmbedding.id,
                (1 - distance_expr).label("cosine_similarity"),
            )
            .join(Student, Student.id == StudentEmbedding.student_id)
            .where(StudentEmbedding.is_active.is_(True))
            .where(Student.is_active.is_(True))
            .order_by(distance_expr)
            .limit(1)
        )
    ).first()

    if nearest_row is None:
        return EmbeddingMatch(student_id=None, embedding_id=None, cosine_similarity=None)

    cosine_similarity = float(nearest_row.cosine_similarity)
    if not math.isfinite(cosine_similarity):
        return EmbeddingMatch(student_id=None, embedding_id=None, cosine_similarity=None)

    if cosine_similarity < STRICT_SIMILARITY_THRESHOLD:
        return EmbeddingMatch(
            student_id=None,
            embedding_id=nearest_row.id,
            cosine_similarity=cosine_similarity,
        )

    return EmbeddingMatch(
        student_id=nearest_row.student_id,
        embedding_id=nearest_row.id,
        cosine_similarity=cosine_similarity,
    )


async def _resolve_vector_matches(embeddings: list[np.ndarray]) -> list[EmbeddingMatch]:
    """Resolve nearest-student matches for each embedding with one async DB session."""
    if not embeddings:
        return []

    session_factory = get_session_factory()
    async with session_factory() as session:
        matches: list[EmbeddingMatch] = []
        for embedding in embeddings:
            matches.append(await _resolve_nearest_embedding_match(session, embedding))
        return matches


def _identity_from_embedding(embedding: np.ndarray) -> str:
    """Generate deterministic identity token from embedding bytes for downstream matching."""
    digest = hashlib.sha1(embedding.tobytes(), usedforsecurity=False).hexdigest()
    return f"identity_{digest[:12]}"


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
        detection_outputs = client.infer_fp32(
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

    liveness_outputs = client.infer_fp32(
        model_name=settings.lvface_model_name,
        tensors={settings.lvface_input_name: face_batch},
        output_names=(
            [settings.lvface_liveness_output_name]
            if settings.lvface_liveness_output_name
            else None
        ),
        request_id=request.request_id,
    )
    liveness_tensor = _select_output_tensor(
        liveness_outputs,
        preferred_name=settings.lvface_liveness_output_name,
        use_last_if_multiple=False,
    )
    liveness_scores = _decode_liveness_scores(liveness_tensor, expected_count=len(tracked_items))

    recognition_outputs = client.infer_fp32(
        model_name=settings.lvface_model_name,
        tensors={settings.lvface_input_name: face_batch},
        output_names=(
            [settings.lvface_embedding_output_name]
            if settings.lvface_embedding_output_name
            else None
        ),
        request_id=request.request_id,
    )
    embedding_tensor = _select_output_tensor(
        recognition_outputs,
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
            "match_threshold": STRICT_SIMILARITY_THRESHOLD,
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


__all__ = [
    "PipelineSettings",
    "extract_enrollment_embedding",
    "get_pipeline_settings",
    "process_inference_batch",
]
