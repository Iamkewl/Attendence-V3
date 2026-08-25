"""Backward-compatible facade for the pipeline subpackage public surface.

Per ARCHITECTURE.md §2.4 and §4.2, this module contains no logic and
re-exports only the PUBLIC surface of the pipeline subpackage. Pre-fix it
also re-exported 18 underscore-prefixed private helpers
(`_decode_bbox`, `_parse_detections`, `_crop_face`, `_decode_frame_tensor`,
`_extract_detection_rows`, `_frame_to_model_input`, `_prepare_face_batch`,
`_resize_nearest`, `_decode_embeddings`, `_identity_from_embedding`,
`_decode_liveness_scores`, `_classify_match`, `_format_pgvector_literal`,
`_resolve_nearest_embedding_match`, `_resolve_vector_matches`,
`_build_lvface_output_names`, `_select_output_tensor`, `_bbox_centroid`,
`_track_detections`) with `# noqa: F401`, contradicting the facade's stated
contract and advertising an internal API surface as part of the public
contract (ATT-005).

The post-fix facade imports ONLY the symbols already re-exported by
``backend/app/services/pipeline/__init__.py`` (the subpackage's own public
surface). Downstream callers — ``app.worker.tasks``,
``app.api.v1.inference``, ``app.api.v1.students`` — only need
``process_inference_batch`` and ``extract_enrollment_embedding``, both
still re-exported here. The underscore-prefixed helpers remain importable
via direct subpackage paths (``from app.services.pipeline.detection
import _decode_bbox``) for code that genuinely needs them; the facade just
no longer advertises them as public.
"""

from .pipeline import (  # noqa: F401
    Detection,
    EmbeddingMatch,
    NoFaceDetectedError,
    PipelineSettings,
    STRICT_SIMILARITY_THRESHOLD,
    TrackedDetection,
    extract_enrollment_embedding,
    get_pipeline_settings,
    process_inference_batch,
)

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
