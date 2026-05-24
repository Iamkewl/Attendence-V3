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
        ``"auto"``   — legacy heuristic: infer format and scale from runtime values.
    normalized:
        When ``True`` (and format is not ``"auto"``), multiply coordinates by
        frame_width / frame_height before clipping.  Ignored in ``"auto"`` mode,
        which applies its own scale heuristic.
    """
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
        x0, y0, x1, y1 = v0, v1, v2, v3
        if x1 <= x0 or y1 <= y0:
            center_x, center_y, width, height = v0, v1, v2, v3
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

    return x0, y0, x1 - x0, y1 - y0


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
                format=bbox_format,
                normalized=bbox_normalized,
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
