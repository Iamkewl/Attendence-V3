"""Object detection parsing: bbox decoding, output selection, and Detection dataclass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame import _extract_detection_rows


@dataclass(frozen=True, slots=True)
class Detection:
    """Single-frame object detection candidate returned from detector post-processing."""

    frame_index: int
    frame_id: str
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int


def _decode_bbox(
    row: np.ndarray | list[float],
    *,
    frame_height: int,
    frame_width: int,
    format: str = "auto",
    normalized: bool = False,
) -> tuple[float, float, float, float]:
    """Decode a detection row into a clipped pixel-space (x, y, w, h) tuple.

    Parameters
    ----------
    row:
        Raw detection values; only the first four elements are consumed.
    frame_height, frame_width:
        Pixel dimensions of the source frame used for clipping and de-normalization.
    format:
        ``"xyxy"``   — columns are [x1, y1, x2, y2, ...] (corner format).
        ``"cxcywh"`` — columns are [cx, cy, w, h, ...] (center format).
        ``"auto"``   — accepted only for backward-compatibility; resolved ONCE by
        ``_autoresolve_bbox_format()`` at the start of ``_parse_detections``
        per frame (one-shot per call), so per-row guessing is gone. When this
        function is reached directly with ``"auto"``, it raises ``ValueError`` —
        the resolution contract is that callers route ``"auto"`` through
        ``_parse_detections`` (or explicitly call ``_autoresolve_bbox_format``
        themselves).
    normalized:
        When ``True`` (and format is not ``"auto"``), multiply coordinates by
        frame_width / frame_height before clipping.  Ignored in ``"auto"`` mode,
        which applies its own scale heuristic (now one-shot, per-frame, not per-row).
    """
    if format == "auto":
        # Per-row guessing was the ATT-027 bug; auto mode must be resolved
        # to an explicit xyxy|cxcywh + normalization flag once per frame by
        # _autoresolve_bbox_format() before this per-row function is called.
        raise ValueError(
            "bbox_format='auto' must be resolved to 'xyxy' or 'cxcywh' by "
            "_autoresolve_bbox_format() before _decode_bbox is called per row."
        )

    v0, v1, v2, v3 = [float(value) for value in row[:4]]

    if format == "xyxy":
        x0, y0, x1, y1 = v0, v1, v2, v3
        if normalized:
            x0 *= frame_width
            y0 *= frame_height
            x1 *= frame_width
            y1 *= frame_height

    elif format == "cxcywh":
        if normalized:
            v0 *= frame_width
            v1 *= frame_height
            v2 *= frame_width
            v3 *= frame_height
        half_w = v2 / 2.0
        half_h = v3 / 2.0
        x0 = v0 - half_w
        y0 = v1 - half_h
        x1 = v0 + half_w
        y1 = v1 + half_h

    else:
        raise ValueError(f"Unsupported bbox format: {format!r} (expected 'xyxy', 'cxcywh').")

    x0 = float(np.clip(x0, 0.0, frame_width - 1.0))
    y0 = float(np.clip(y0, 0.0, frame_height - 1.0))
    x1 = float(np.clip(x1, 0.0, frame_width - 1.0))
    y1 = float(np.clip(y1, 0.0, frame_height - 1.0))

    if x1 <= x0 or y1 <= y0:
        raise ValueError("Decoded bounding box is invalid after clipping.")

    return x0, y0, x1 - x0, y1 - y0


# ---------------------------------------------------------------------------
# ATT-027: per-frame autoresolution of bbox format + normalization.
#
# Replaces the previous per-row guessing ("auto" branch in _decode_bbox) that
# silently discarded tiny XYXY pixel-space boxes (max abs coordinate <= 1.5):
# it inferred normalized scale and rescaled by frame dims, pushing boxes
# outside the frame and dropping them at the post-clip ValueError.
#
# The new contract is one-shot per _parse_detections frame call:
#   1. Collect the first 4 columns of every detection row.
#   2. Decide coordinate format:
#        - "xyxy"    if a strict majority of rows satisfy  v2 > v0  AND  v3 > v1
#          (i.e. the second corner is bottom-right of the first corner).
#        - "cxcywh"  otherwise.
#      Ties round to xyxy (the documented production default; see
#      `pipeline/settings.py:95`).
#   3. Decide normalization:
#        - normalized=True  if max |coord| across all rows <= 1.0  (typical
#          normalized detector output lives in [0, 1]).
#        - normalized=False if max |coord| > 1.0  — pixel-space regardless
#          of how small. Anchors the issue's literal ACCEPT: an XYXY detector
#          output with boxes inside [-1.5, 1.5] pixel space decodes to the
#          SAME boxes (NOT rescaled to frame_width * 1.5).
#
# This is "one-time autodetection per model (cache the format inferred from a
# sample output)" per the issue's recommended FIX option (b). The "cache" is
# the local variables inside _parse_detections — autodetection runs once per
# frame, then every row is decoded with the resolved (format, normalized) pair.
# Per-row guessing is gone.
# ---------------------------------------------------------------------------


def _autoresolve_bbox_format(
    rows: np.ndarray | list[np.ndarray],
) -> tuple[str, bool]:
    """Resolve bbox format and normalized flag from a sample of detector rows.

    Returns ``(format_str, normalized_bool)`` for use by ``_decode_bbox``.

    The empty-input case returns the documented production default
    (``"xyxy"``, un-normalized) — see ``pipeline/settings.py:95,120``.
    """
    # Materialize so we can iterate twice if needed (the input may be an ndarray
    # of ndarrays, or a true list). We only iterate once here, but converting
    # to a list guards against len() returning 0 on an uninitialized ndarray
    # view while iter() still yields rows.
    rows_list = list(rows) if not isinstance(rows, list) else rows

    if not rows_list:
        # No rows means no autodetect signal — fall through to the production
        # default (xyxy, pixel-space). _decode_bbox is still callable for an
        # empty list because _parse_detections iterates the loop zero times.
        return "xyxy", False

    xyxy_votes = 0
    cxcywh_votes = 0
    max_abs_coord = 0.0

    for row in rows_list:
        if row.shape[0] < 4:
            continue
        v0, v1, v2, v3 = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        max_abs_coord = max(max_abs_coord, abs(v0), abs(v1), abs(v2), abs(v3))
        # xyxy expects second corner strictly greater than first (positive area).
        # cxcywh has w,h > 0 in practice but doesn't impose a strict ordering
        # against v0/v1 (cx,cy). Use the geometric validity of xyxy as a vote.
        if v2 > v0 and v3 > v1:
            xyxy_votes += 1
        else:
            cxcywh_votes += 1

    if cxcywh_votes > xyxy_votes:
        format_str = "cxcywh"
    else:
        # Ties round to xyxy (production default).
        format_str = "xyxy"

    # Normalization: typical normalized detector outputs live in [0, 1]. A
    # max-abs <= 1.0 strongly suggests normalized coordinates. Anything else
    # (including the issue's cited [-1.5, 1.5] pixel-space case) is treated
    # as pixel-space and NOT rescaled.
    normalized = max_abs_coord <= 1.0

    return format_str, normalized


def _parse_detections(
    *,
    frame_index: int,
    frame_id: str,
    outputs: dict[str, np.ndarray],
    confidence_threshold: float,
    frame_height: int,
    frame_width: int,
    preferred_output_name: str | None,
    bbox_format: str = "auto",
    bbox_normalized: bool = False,
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

    # ATT-027: when bbox_format == "auto", resolve the format + normalized
    # flag ONCE per frame from a sample of detector rows (one-shot autodetect
    # per call, NOT per-row guessing). The resolved values replace the auto
    # mode for every subsequent per-row _decode_bbox call. When bbox_format
    # is already explicit ("xyxy" / "cxcywh"), keep using the operator's
    # explicit bbox_normalized flag directly (no autodetect).
    resolved_format = bbox_format
    resolved_normalized = bbox_normalized
    if bbox_format == "auto":
        resolved_format, resolved_normalized = _autoresolve_bbox_format(rows)

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
                format=resolved_format,
                normalized=resolved_normalized,
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
