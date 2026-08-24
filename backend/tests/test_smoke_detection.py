"""ATT-027 regression: `_decode_bbox` / `_parse_detections` "auto" mode must
be deterministic AND must not silently discard tiny XYXY pixel-space boxes
by re-scaling them past the frame and clipping them away.

Pre-fix "auto" branch in `_decode_bbox` (detection.py:71-85) had two per-row
heuristics that combined to drop detections silently:
  1. If `x1 <= x0 OR y1 <= y0`, flip the row to `cxcywh`.
  2. If `max(|coords|) <= 1.5`, multiply all four by frame dims.

Bug: a genuine XYXY detector output whose four corner coords all live
inside `[-1.5, +1.5]` *pixels* was treated as normalized and multiplied by
frame dims, pushing boxes outside the frame. Clipping then collapsed some
boxes (x1==x0 or y1==y0) and raised ValueError, swallowed to `continue` by
`_parse_detections`, silently dropping the detection.

Fix routes `bbox_format="auto"` through a per-frame resolver
(`_autoresolve_bbox_format`) that picks ONE format + ONE normalization flag
for the entire frame (the issue's recommended FIX option (b): "one-time
autodetection per model instead of per-row guessing").
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.pipeline.detection import (
    _autoresolve_bbox_format,
    _decode_bbox,
    _parse_detections,
)


# ---------------------------------------------------------------------------
# Explicit modes preserved (regression on the fix doesn't break the
# documented production-supported paths).
# ---------------------------------------------------------------------------


def test_att_027_explicit_xyxy_pixel_space_unchanged() -> None:
    """xyxy + normalized=False: input coords come back unchanged (clipped)."""
    bbox = _decode_bbox(
        [10.0, 20.0, 30.0, 40.0, 0.99, 0],
        frame_height=1080,
        frame_width=1920,
        format="xyxy",
        normalized=False,
    )
    assert bbox == (10.0, 20.0, 20.0, 20.0)


def test_att_027_explicit_xyxy_normalized_unaffected() -> None:
    """xyxy + normalized=True: each coord * frame dim (pre-fix behaviour)."""
    bbox = _decode_bbox(
        [0.1, 0.2, 0.3, 0.4, 0.99, 0],
        frame_height=100,
        frame_width=200,
        format="xyxy",
        normalized=True,
    )
    assert bbox == (20.0, 20.0, 40.0, 20.0)


def test_att_027_explicit_cxcywh_pixel_space_unchanged() -> None:
    """cxcywh + normalized=False: cxcywh decoding is preserved pre/post fix."""
    bbox = _decode_bbox(
        [50.0, 50.0, 20.0, 20.0, 0.99, 0],
        frame_height=1080,
        frame_width=1920,
        format="cxcywh",
        normalized=False,
    )
    assert bbox == (40.0, 40.0, 20.0, 20.0)


def test_att_027_unsupported_format_raises() -> None:
    """Unknown format strings raise ValueError (not silently pass through)."""
    with pytest.raises(ValueError, match="Unsupported bbox format"):
        _decode_bbox(
            [10.0, 20.0, 30.0, 40.0, 0.99, 0],
            frame_height=1080,
            frame_width=1920,
            format="xyz",
        )


# ---------------------------------------------------------------------------
# "auto" mode contract: must be resolved by _autoresolve_bbox_format (one-shot
# per frame); per-row _decode_bbox must refuse "auto" — pins the bug-trigger
# removal ("per-row guessing" is gone).
# ---------------------------------------------------------------------------


def test_att_027_auto_mode_runs_in_parse_detections_not_decode_bbox() -> None:
    """Calling _decode_bbox directly with auto raises (per-row guessing gone)."""
    with pytest.raises(ValueError, match="auto"):
        _decode_bbox(
            [0.4, 0.4, 0.6, 0.6, 0.99, 0],
            frame_height=100,
            frame_width=100,
            format="auto",
        )


# ---------------------------------------------------------------------------
# The literal ACCEPT: synthetic XYXY detector output with all boxes inside
# [-1.5, 1.5] pixel space decodes via "auto" mode to the SAME boxes (NOT
# rescaled).
# ---------------------------------------------------------------------------


def test_att_027_auto_mode_is_deterministic_for_xyxy_pixel_space() -> None:
    """The issue's literal ACCEPT — verified via _parse_detections.

    Pre-fix: a row [1.0, 1.0, 1.5, 1.5] XYXY-pixel passes the v2>v0 AND
    v3>v1 xyxy check, then hits max(|coord|) <= 1.5, multiplies all four
    by frame dims (1.5 * 100 = 150), clips to (1,1,100,100). Resulting box
    is rescaled, NOT what the operator asked for.

    Post-fix: auto mode resolves to xyxy + pixel-space (max_abs > 1.0 →
    not normalized), then _decode_bbox decodes [1,1,1.5,1.5] unchanged
    (clipped within frame).
    """
    rows = np.array([[1.0, 1.0, 1.5, 1.5, 0.9, 0]], dtype=np.float32)
    outputs = {"yolov12:0": rows}
    detections = _parse_detections(
        frame_index=0,
        frame_id="frame-0",
        outputs=outputs,
        confidence_threshold=0.5,
        frame_height=100,
        frame_width=100,
        preferred_output_name=None,
        bbox_format="auto",
    )
    assert len(detections) == 1
    assert detections[0].bbox == (1.0, 1.0, 0.5, 0.5)


def test_att_027_auto_mode_no_silently_dropped_detections_for_tiny_xyxy_box() -> None:
    """The silent-drop bug-trigger: a row [0.4, 0.4, 1.5, 1.5] xyxy-pixel that
    would have collapsed after pre-fix rescaling + clipping now decodes
    untouched (pixel-space, no rescale).
    """
    rows = np.array([[0.4, 0.4, 1.5, 1.5, 0.9, 0]], dtype=np.float32)
    outputs = {"yolov12:0": rows}
    detections = _parse_detections(
        frame_index=0,
        frame_id="frame-1",
        outputs=outputs,
        confidence_threshold=0.5,
        frame_height=100,
        frame_width=100,
        preferred_output_name=None,
        bbox_format="auto",
    )
    assert len(detections) == 1
    # Use approx for float32 rounding (raw rows come in as float32 from Triton).
    assert detections[0].bbox == pytest.approx(
        (0.4, 0.4, 1.1, 1.1), abs=1e-5
    )


# ---------------------------------------------------------------------------
# Autodetection contract: per-frame resolution, ONE format for ALL rows.
# ---------------------------------------------------------------------------


def test_att_027_auto_mode_resolved_once_per_frame_consistent_format() -> None:
    """All-xyxy rows → xyxy. Per-row flip heuristic is GONE.

    A frame where ALL rows look geometrically xyxy (`v2 > v0 AND v3 > v1`)
    resolves to xyxy + per-frame normalization. No row has its format
    flipped individually (the pre-fix per-row flip is the bug).
    """
    rows = np.array(
        [
            [10.0, 20.0, 30.0, 40.0, 0.9, 0],
            [50.0, 50.0, 80.0, 90.0, 0.85, 0],
            [5.0, 5.0, 15.0, 20.0, 0.95, 0],
        ],
        dtype=np.float32,
    )
    outputs = {"yolov12:0": rows}
    detections = _parse_detections(
        frame_index=2,
        frame_id="frame-2",
        outputs=outputs,
        confidence_threshold=0.5,
        frame_height=1080,
        frame_width=1920,
        preferred_output_name=None,
        bbox_format="auto",
    )
    assert len(detections) == 3
    # Each row decodes as xyxy pixel-space, no rescale.
    assert detections[0].bbox == (10.0, 20.0, 20.0, 20.0)
    assert detections[1].bbox == (50.0, 50.0, 30.0, 40.0)
    assert detections[2].bbox == (5.0, 5.0, 10.0, 15.0)


def test_att_027_auto_mode_majority_cxcywh_rows_pick_cxcywh() -> None:
    """A frame whose rows are GEOMETRICALLY xyxy-invalid resolves to cxcywh.

    Two rows with `v2 <= v0 OR v3 <= v1` flip the autodetect to cxcywh.
    Each row then decodes through the cxcywh branch in `_decode_bbox`.
    """
    # Rows: [cx, cy, w, h, score, class_id].
    # Row 1: cx=50, cy=50, w=20, h=20 → xyxy check v2=20 > v0=50? NO → cxcywh vote.
    # Row 2: cx=100, cy=100, w=30, h=40 → xyxy check v2=30 > v0=100? NO → cxcywh vote.
    rows = np.array(
        [
            [50.0, 50.0, 20.0, 20.0, 0.9, 0],
            [100.0, 100.0, 30.0, 40.0, 0.85, 0],
        ],
        dtype=np.float32,
    )
    outputs = {"yolov12:0": rows}
    detections = _parse_detections(
        frame_index=3,
        frame_id="frame-3",
        outputs=outputs,
        confidence_threshold=0.5,
        frame_height=1080,
        frame_width=1920,
        preferred_output_name=None,
        bbox_format="auto",
    )
    assert len(detections) == 2
    # Row 1: cx=50, cy=50, w=20, h=20 → xyxy=(40,40,60,60) → (40, 40, 20, 20)
    assert detections[0].bbox == (40.0, 40.0, 20.0, 20.0)
    # Row 2: xyxy=(85,80,115,120) → (85, 80, 30, 40)
    assert detections[1].bbox == (85.0, 80.0, 30.0, 40.0)


def test_att_027_autoresolve_empty_returns_production_default() -> None:
    """No rows means no autodetect signal — fall through to documented default.

    Production default per `pipeline/settings.py:95,120` is xyxy +
    un-normalized. _autoresolve_bbox_format must match that on empty input.
    """
    fmt, normalized = _autoresolve_bbox_format(np.array([]).reshape(0, 6))
    assert fmt == "xyxy"
    assert normalized is False


def test_att_027_autoresolve_single_normalized_row_pick_xyxy_normalized() -> None:
    """One row with coords strictly in [0,1] → xyxy + normalized=True."""
    rows = np.array([[0.1, 0.2, 0.3, 0.4, 0.9, 0]], dtype=np.float32)
    fmt, normalized = _autoresolve_bbox_format(rows)
    assert fmt == "xyxy"
    assert normalized is True


def test_att_027_auto_xyxy_normalized_full_frame_resolution() -> None:
    """A frame of normalized xyxy rows resolves to xyxy + normalized=True,
    so each row gets rescaled by frame dims in `_decode_bbox`.
    """
    rows = np.array(
        [
            [0.1, 0.1, 0.2, 0.2, 0.9, 0],
            [0.5, 0.5, 0.6, 0.6, 0.85, 0],
        ],
        dtype=np.float32,
    )
    outputs = {"yolov12:0": rows}
    detections = _parse_detections(
        frame_index=4,
        frame_id="frame-4",
        outputs=outputs,
        confidence_threshold=0.5,
        frame_height=100,
        frame_width=200,
        preferred_output_name=None,
        bbox_format="auto",
    )
    assert len(detections) == 2
    # Row 1: [0.1*200, 0.1*100, 0.2*200, 0.2*100] = [20, 10, 40, 20] → (20, 10, 20, 10)
    assert detections[0].bbox == pytest.approx(
        (20.0, 10.0, 20.0, 10.0), abs=1e-3
    )
    # Row 2: [0.5*200, 0.5*100, 0.6*200, 0.6*100] = [100, 50, 120, 60] → (100, 50, 20, 10)
    assert detections[1].bbox == pytest.approx(
        (100.0, 50.0, 20.0, 10.0), abs=1e-3
    )
