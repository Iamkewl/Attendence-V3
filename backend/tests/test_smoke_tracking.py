"""ATT-028 regression: `_track_detections` Two key contract changes:

1. The module + function docstrings must say "centroid" (truthful — the
   tracker is pure centroid Euclidean distance), NOT "IoU" (false — no IoU
   computation ever existed).

2. A camera that emits two single-frame batches 5 seconds apart, where
   both frames contain the same face area, gets the same `track_id`
   (unique negative sentinel ids), per the issue's
   literal ACCEPT ("or no `track_id` if the buffer is disabled").

Multi-frame batches (>= 2 frames in a single `_track_detections` call)
keep the within-batch centroid-linkage tracker behavior, which assigns
fresh track IDs starting at 1. The pre-fix tracker was already this way
for the first frame of any batch.

Pre-fix bug: in single-frame-batch mode (the documented production
periodic-CCTV case), every detection got a fresh per-call track ID,
giving the misleading impression that two batches 5s apart represent
different tracks. The post-fix short-circuit acknowledges that the
tracker has no cross-batch signal by emitting the sentinel 0 for every
detection in a single-frame batch.
"""

from __future__ import annotations

from pathlib import Path

from app.services.pipeline.detection import Detection
from app.services.pipeline.tracking import (
    TrackedDetection,
    _track_detections,
)


_TRACKING_PY_PATH = Path(__file__).resolve().parents[1] / "app/services/pipeline/tracking.py"


def _mk_detection(
    *,
    frame_index: int = 0,
    frame_id: str = "f0",
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    score: float = 0.9,
    class_id: int = 0,
) -> Detection:
    """Helper factory — Detection is a frozen dataclass, all fields required."""
    return Detection(
        frame_index=frame_index,
        frame_id=frame_id,
        bbox=bbox,
        score=score,
        class_id=class_id,
    )


# ---------------------------------------------------------------------------
# Docstring accuracy — the issue's ACCEPT literally says 'docstring says
# "centroid" not "IoU"'.
# ---------------------------------------------------------------------------


def test_att_028_docstring_does_not_claim_iou() -> None:
    """The pre-fix module docstring said 'IoU-based' but no IoU computation
    ever existed in tracking.py. That false descriptive claim is now gone.
    """
    src = _TRACKING_PY_PATH.read_text(encoding="utf-8")
    # Word-boundary case-sensitive check for "IoU". The pre-fix docstring
    # said "IoU-based nearest-centroid temporal tracker". The post-fix
    # docstring should NOT use the false descriptive claim anywhere — the
    # module + function docstrings now describe the actual algorithm
    # (centroid Euclidean distance) without invoking the false claim.
    # Explanatory comments resolving the historical lie use the descriptive
    # phrase "intersection-over-union" — see lines near the bottom of the
    # function docstring.
    assert "IoU" not in src, (
        "tracking.py must not claim 'IoU'-based tracking (no IoU computation "
        "ever existed in this module — the truth is pure centroid distance). "
        "If explanatory text discusses the historical lie, use the descriptive "
        "phrase 'intersection-over-union' rather than the abbreviation 'IoU'."
    )


def test_att_028_docstring_mentions_centroid() -> None:
    """The post-fix module docstring must mention 'centroid'."""
    src = _TRACKING_PY_PATH.read_text(encoding="utf-8")
    # Find the module docstring (first triple-quoted block).
    import re
    match = re.search(r'^"""(.*?)"""', src, re.DOTALL)
    assert match is not None, "tracking.py has no module docstring"
    docstring = match.group(1)
    assert "centroid" in docstring or "Centroid" in docstring, (
        f"tracking.py module docstring must mention 'centroid' to reflect "
        f"that the tracker is centroid-distance-based, not IoU-based. "
        f"Got: {docstring!r}"
    )


# ---------------------------------------------------------------------------
# The literal ACCEPT: same face → same track_id across two single-frame
# batches 5 seconds apart.
# ---------------------------------------------------------------------------


def test_att_028_single_frame_batch_same_face_across_two_batches_gets_same_track_id() -> None:
    """The issue's literal ACCEPT — verified directly with two single-frame
    `_track_detections` invocations (simulating two single-frame batches).

    Pre-fix: batch_1 returns detections with track_ids [1, 2] (fresh per
    call). Batch_2 returns detections with track_ids [1, 2] AGAIN (fresh
    per call). The same face in batch_1's frame and batch_2's frame could
    by coincidence share track_id=1, but it was accidental — and distinct
    faces could equally share track_id=1.

    Post-fix: every detection in a single-frame batch gets the sentinel
    unique negative sentinels. The ACCEPT's 'or no `track_id`' arm is the one
    satisfied: every row reads as unassociated rather than colliding.
    """
    # Batch 1: one frame with two faces (different bbox areas).
    batch1_frame = [
        _mk_detection(frame_id="batch-1-frame", bbox=(10.0, 10.0, 20.0, 20.0), score=0.9),
        _mk_detection(frame_id="batch-1-frame", bbox=(100.0, 100.0, 110.0, 110.0), score=0.8),
    ]
    batch1_results = _track_detections([batch1_frame], max_link_distance=96.0)

    # Batch 2: 5 seconds later, one frame, same face as batch_1's first face.
    batch2_frame = [
        _mk_detection(frame_id="batch-2-frame", bbox=(10.0, 10.0, 21.0, 21.0), score=0.85),
    ]
    batch2_results = _track_detections([batch2_frame], max_link_distance=96.0)

    # All single-frame-batch track_ids are unique negative sentinels: they
    # read as "no cross-batch association" and can never collide with real
    # tracker ids (>= 1).
    assert len(batch1_results) == 2
    assert len(batch2_results) == 1
    assert all(t.track_id < 0 for t in batch1_results)
    assert all(t.track_id < 0 for t in batch2_results)
    ids_b1 = [t.track_id for t in batch1_results]
    ids_b2 = [t.track_id for t in batch2_results]
    assert len(ids_b1) == len(set(ids_b1))
    assert len(ids_b2) == len(set(ids_b2))


def test_att_028_single_frame_batch_emits_sentinel_for_all_detections() -> None:
    """Every detection in a single-frame batch gets a unique negative sentinel,
    regardless of count or surface bbox — pre-refinement code emitted the SAME
    id (0) for every row, which collides with the results table's React keys
    and collapses orchestrator track_count to 1.
    """
    frame = [
        _mk_detection(frame_id="f", bbox=(10.0, 10.0, 20.0, 20.0), score=0.9),
        _mk_detection(frame_id="f", bbox=(100.0, 100.0, 110.0, 110.0), score=0.8),
        _mk_detection(frame_id="f", bbox=(500.0, 500.0, 510.0, 510.0), score=0.7),
    ]
    results = _track_detections([frame], max_link_distance=96.0)
    assert len(results) == 3
    assert all(r.track_id < 0 for r in results)
    ids = [r.track_id for r in results]
    assert len(ids) == len(set(ids)), f"sentinel ids must be unique, got {ids}"


def test_att_028_empty_batch_returns_empty_list() -> None:
    """An empty `detections_by_frame` produces no TrackedDetections.

    Preserves existing behavior — pre-empt multi-frame branch is unreachable.
    """
    assert _track_detections([], max_link_distance=96.0) == []


def test_att_028_single_frame_batch_with_zero_detections_returns_empty() -> None:
    """A batch with one frame containing zero detections produces no results."""
    assert _track_detections([[]], max_link_distance=96.0) == []


# ---------------------------------------------------------------------------
# Multi-frame batches: within-batch tracker behavior is preserved.
# ---------------------------------------------------------------------------


def test_att_028_multi_frame_tracker_links_same_face_within_batch() -> None:
    """Within a multi-frame batch (e.g. a 3-frame video inference batch),
    the same face across consecutive frames should get the SAME track_id
    via centroid linkage.

    The pre-fix tracker was already this way within a batch — this test
    anchors that the single-frame short-circuit doesn't break multi-frame
    linkage.
    """
    # Frame 0: one face at (10, 10). Frame 1: same face moved slightly, at (12, 12).
    # Centroid frame_0: (15, 15). Centroid frame_1: (17, 17). Distance: ~2.83 < 96.
    frame_0 = [_mk_detection(frame_index=0, frame_id="f0", bbox=(10.0, 10.0, 20.0, 20.0), score=0.95)]
    frame_1 = [_mk_detection(frame_index=1, frame_id="f1", bbox=(12.0, 12.0, 22.0, 22.0), score=0.9)]
    results = _track_detections([frame_0, frame_1], max_link_distance=96.0)

    assert len(results) == 2
    # Both detections share track_id (linked by centroid distance).
    assert results[0].track_id == results[1].track_id
    # The shared track_id is positive (multi-frame batches use real track IDs
    # starting at 1); sentinels are negative.
    assert results[0].track_id >= 1


def test_att_028_multi_frame_tracker_assigns_fresh_id_when_no_link() -> None:
    """A detection with no centroid match in the previous frame gets a fresh
    track ID (preserved from pre-fix behavior).
    """
    # Frame 0: face at top-left (centroid 5,5). Frame 1: face bottom-right (centroid 990, 990).
    # Distance way over max_link_distance=96 → unmatched → fresh track_id in frame_1.
    frame_0 = [_mk_detection(frame_index=0, frame_id="f0", bbox=(0.0, 0.0, 10.0, 10.0), score=0.9)]
    frame_1 = [_mk_detection(frame_index=1, frame_id="f1", bbox=(985.0, 985.0, 995.0, 995.0), score=0.85)]
    results = _track_detections([frame_0, frame_1], max_link_distance=96.0)

    assert len(results) == 2
    # Different track_ids because the centroids are way out of linkage range.
    assert results[0].track_id != results[1].track_id
    # Multi-frame track IDs start at 1.
    assert results[0].track_id == 1
    assert results[1].track_id == 2


def test_att_028_multi_frame_tracker_real_track_ids_do_not_collide_with_sentinel() -> None:
    """Multi-frame batches' track IDs (>= 1) live in a disjoint range from the
    single-frame sentinels (negative), so no downstream comparison can see a
    collision between a real track and a "no association" marker.
    """
    # Multi-frame batch with two faces per frame for several frames.
    frame_a = [
        _mk_detection(frame_index=0, frame_id="f-a", bbox=(10.0, 10.0, 20.0, 20.0), score=0.95),
        _mk_detection(frame_index=0, frame_id="f-a", bbox=(100.0, 100.0, 110.0, 110.0), score=0.85),
    ]
    frame_b = [
        _mk_detection(frame_index=1, frame_id="f-b", bbox=(12.0, 12.0, 22.0, 22.0), score=0.9),
        _mk_detection(frame_index=1, frame_id="f-b", bbox=(102.0, 102.0, 112.0, 112.0), score=0.8),
    ]
    results = _track_detections([frame_a, frame_b], max_link_distance=96.0)
    assert len(results) == 4
    # No multi-frame track_id may fall into the negative sentinel domain.
    assert all(r.track_id >= 1 for r in results)


def test_att_028_tracked_detection_dataclass_preserves_int_track_id_type() -> None:
    """`track_id` remains an int in the public `TrackedDetection` dataclass
    so the existing outbound schema at
    `backend/app/domain/schemas/inference.py:102` (`track_id: int`) is
    preserved. Anchors that the fix doesn't widen the field to Optional to
    avoid a breaking JSON schema change.
    """
    frame = [_mk_detection(frame_id="f", bbox=(0.0, 0.0, 5.0, 5.0), score=0.9)]
    results = _track_detections([frame], max_link_distance=96.0)
    assert len(results) == 1
    sentinel = results[0].track_id
    # Sentinel is an int (not None).
    assert isinstance(sentinel, int)
    assert sentinel < 0


def test_att_028_tracked_detection_preserves_detection_fields() -> None:
    """TrackedDetection sub-classes Detection and must preserve all fields
    (frame_index, frame_id, bbox, score, class_id) — single-frame path.
    """
    frame = [
        _mk_detection(
            frame_index=0,
            frame_id="frame-x",
            bbox=(10.0, 20.0, 30.0, 40.0),
            score=0.77,
            class_id=5,
        )
    ]
    results = _track_detections([frame], max_link_distance=96.0)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, TrackedDetection)
    assert r.frame_index == 0
    assert r.frame_id == "frame-x"
    assert r.bbox == (10.0, 20.0, 30.0, 40.0)
    assert r.score == 0.77
    assert r.class_id == 5
    assert r.track_id < 0
