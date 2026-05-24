"""IoU-based nearest-centroid temporal tracker."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .detection import Detection


@dataclass(frozen=True, slots=True)
class TrackedDetection(Detection):
    """Detection enriched with temporal track identity."""

    track_id: int


def _bbox_centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return bbox centroid coordinates for nearest-neighbor tracking."""
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


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
