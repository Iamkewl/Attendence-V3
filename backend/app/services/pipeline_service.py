"""Backward-compatible facade — re-exports every public symbol from the pipeline subpackage."""

from .pipeline.detection import Detection, _decode_bbox, _parse_detections  # noqa: F401
from .pipeline.embedding import _decode_embeddings, _identity_from_embedding  # noqa: F401
from .pipeline.frame import (  # noqa: F401
    _crop_face,
    _decode_frame_tensor,
    _extract_detection_rows,
    _frame_to_model_input,
    _prepare_face_batch,
    _resize_nearest,
)
from .pipeline.liveness import _decode_liveness_scores  # noqa: F401
from .pipeline.matching import (  # noqa: F401
    EmbeddingMatch,
    STRICT_SIMILARITY_THRESHOLD,
    _classify_match,
    _format_pgvector_literal,
    _resolve_nearest_embedding_match,
    _resolve_vector_matches,
)
from .pipeline.orchestrator import (  # noqa: F401
    _build_lvface_output_names,
    _select_output_tensor,
    extract_enrollment_embedding,
    process_inference_batch,
)
from .pipeline.settings import PipelineSettings, get_pipeline_settings  # noqa: F401
from .pipeline.tracking import TrackedDetection, _bbox_centroid, _track_detections  # noqa: F401

__all__ = [
    "Detection",
    "EmbeddingMatch",
    "PipelineSettings",
    "STRICT_SIMILARITY_THRESHOLD",
    "TrackedDetection",
    "extract_enrollment_embedding",
    "get_pipeline_settings",
    "process_inference_batch",
]
