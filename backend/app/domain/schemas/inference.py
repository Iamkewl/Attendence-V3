"""Inference pipeline schemas: tensor payloads, batch requests, and recognition results."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import Field, StringConstraints, field_validator, model_validator

from .common import SchemaModel


class ImageTensorPayload(SchemaModel):
    """Validated frame payload carrying raw tensor bytes encoded as base64."""

    frame_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    data_base64: Annotated[str, StringConstraints(min_length=1)]
    height: Annotated[int, Field(ge=1, le=4096)]
    width: Annotated[int, Field(ge=1, le=4096)]
    channels: Annotated[int, Field(ge=1, le=4)] = 3
    dtype: Literal["uint8", "float32"] = "uint8"
    normalize: bool = True
    captured_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data_base64")
    @classmethod
    def validate_base64_data(cls, value: str) -> str:
        """Require valid non-empty base64 payload to prevent malformed tensor input."""
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("data_base64 must be valid base64 data.") from exc

        if len(decoded) == 0:
            raise ValueError("data_base64 must not decode to empty payload.")

        return value

    @model_validator(mode="after")
    def validate_tensor_shape(self) -> ImageTensorPayload:
        """Enforce byte-size consistency between payload and declared tensor dimensions."""
        raw = base64.b64decode(self.data_base64, validate=True)
        item_size = 1 if self.dtype == "uint8" else 4
        expected_size = self.height * self.width * self.channels * item_size

        if len(raw) != expected_size:
            raise ValueError(
                "data_base64 payload size does not match height * width * channels * dtype item size"
            )

        return self


class InferenceBatchRequest(SchemaModel):
    """Input schema used to enqueue one inference workload composed of one or more frames."""

    request_id: Annotated[str | None, StringConstraints(min_length=1, max_length=128)] = None
    session_id: UUID | None = None
    course_id: UUID | None = None
    room_id: UUID | None = None
    camera_id: Annotated[str | None, StringConstraints(min_length=1, max_length=128)] = None
    frames: Annotated[list[ImageTensorPayload], Field(min_length=1, max_length=128)]
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    liveness_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    include_embeddings: bool = False


class InferenceTaskAccepted(SchemaModel):
    """Response schema returned when an inference task is successfully queued."""

    task_id: str
    state: str
    queued_at: datetime
    frame_count: Annotated[int, Field(ge=1)]


class InferenceTaskStatus(SchemaModel):
    """Response schema exposing asynchronous task execution state and optional result data."""

    task_id: str
    state: str
    result: dict[str, Any] | None = None
    error: str | None = None


class RecognitionMatch(SchemaModel):
    """Matched student identity returned from one recognition detection."""

    student_id: UUID
    student_full_name: str
    student_number: str
    cosine_similarity: float


class RecognitionDetection(SchemaModel):
    """Single face detection with optional recognition result from synchronous inference."""

    track_id: int
    bbox: Annotated[list[float], Field(min_length=4, max_length=4)]
    confidence: float
    liveness_score: float
    is_live: bool
    match: RecognitionMatch | None


class RecognitionPhotoResponse(SchemaModel):
    """Response schema for synchronous photo inference returning all detected faces."""

    image_width: int
    image_height: int
    detection_count: int
    match_count: int
    processed_at: datetime
    detections: list[RecognitionDetection]
