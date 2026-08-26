"""pipeline subpackage — single-responsibility modules for the inference pipeline."""

from .detection import Detection
from .matching import EmbeddingMatch
from .orchestrator import (
    NoFaceDetectedError,
    analyze_enrollment_frame,
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
    "TrackedDetection",
    "analyze_enrollment_frame",
    "extract_enrollment_embedding",
    "get_pipeline_settings",
    "process_inference_batch",
]
