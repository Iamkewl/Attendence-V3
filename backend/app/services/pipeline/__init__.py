"""pipeline subpackage — single-responsibility modules for the inference pipeline."""

from .detection import Detection
from .matching import EmbeddingMatch, STRICT_SIMILARITY_THRESHOLD
from .orchestrator import (
    NoFaceDetectedError,
    extract_enrollment_embedding,
    process_inference_batch,
)
from .settings import PipelineSettings, get_pipeline_settings
from .tracking import TrackedDetection

__all__ = [
    "Detection",
    "EmbeddingMatch",
    "NoFaceDetectedError",
    "PipelineSettings",
    "STRICT_SIMILARITY_THRESHOLD",
    "TrackedDetection",
    "extract_enrollment_embedding",
    "get_pipeline_settings",
    "process_inference_batch",
]
