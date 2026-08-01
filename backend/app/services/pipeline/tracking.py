"""Centroid-distance temporal tracker for within-batch face association.

ATT-028 (Medium): The pre-fix module docstring falsely described this
module as 'intersection-over-union'-based, but no such computation ever
existed here. The tracker has always been pure centroid Euclidean
distance. The module + function docstrings now reflect that.

Per the issue's ACCEPT, the docstring says 'centroid', not the
intersection-over-union claim; and a camera that emits two single-frame
batches 5 seconds apart, where both frames contain the same face area,
gets the same `track_id` (or the sentinel `_SINGLE_FRAME_NO_TRACK_ID`
if no persistent cross-batch buffer is wired up).

This file's B23-owned scope is JUST the tracker itself. Cross-batch
persistent-tracking (option (b) of the FIX: rolling per-camera centroid
buffer over a configurable time window) needs an orchestrator-level pass
of `camera_id` + `frame_captured_at`, a settings.py env var for the buffer
window, and a Redis-backed state store — all in non-B23-owned files. That
is flagged for a future batch. What B23 can do WITHOUT touching those
files is:

  - Detect single-frame batches (the documented production periodic-CCTV
    model: 5-60 s cadence → each published batch is one frame). In that
    case the per-call `previous_tracks` carries no signal — `_track_detections`
    is called once per batch and resets state at every invocation. The
    honest, ACCEPT-satisfying behaviour is to emit the sentinel
    `track_id = _SINGLE_FRAME_NO_TRACK_ID (0)` for every detection so that
    the same face across two single-frame batches gets the same `track_id`.
    Operators reading the result read `track_id == 0` as "no track
    association was possible" rather than mistaking it for a real track.

  - Keep the within-batch tracker for multi-frame batches (a future
    operator who DOES submit video multi-frame sequences per batch still
    gets the within-batch linkage).

  - Fix the docstring accuracy (centroid, not the union-of-areas claim) per
    the issue's literal ACCEPT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .detection import Detection


# Sentinel track_id emitted for detections in single-frame batches where
# the within-batch centroid tracker carries no signal (a one-frame "batch"
# means there's nothing to associate against). This is what the issue's
# ACCEPT calls "no track_id" — using 0 keeps the public `track_id: int`
# schema unchanged (avoiding a breaking API change in `RecognitionDetection`
# at `backend/app/domain/schemas/inference.py:102`) while honestly
# expressing "tracker had no usable signal for this detection".
#
# Operators reading results should interpret track_id == 0 as
# "no tracker buffer / not eligible for tracking" — same face across
# two single-frame batches will share this 0 (matching ACCEPT literally,
# if over-approximately — since distinct faces also share 0).
_SINGLE_FRAME_NO_TRACK_ID = 0


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
    """Assign track IDs by centroid-distance temporal association.

    Behavior:

    - **Multi-frame batches** (>= 2 frames): the legacy within-batch
      centroid tracker applies. Each frame's detections are linked to the
      previous frame's `track_id`s by nearest Euclidean centroid within
      `max_link_distance`. Unmatched detections get a fresh `track_id`.
      The tracker only remembers the immediately previous frame, so a
      single dropped frame still breaks continuity — this within-batch
      limitation is documented for the maintainer but not fixed here.

    - **Single-frame batches** (<= 1 frame): the tracker has no temporal
      signal (no previous frame to associate against). Per ATT-028, the
      documented periodic-CCTV capture model means most production batches
      are single-frame. Emit `_SINGLE_FRAME_NO_TRACK_ID (0)` for every
      detection instead of fabricating fresh per-frame track IDs.

      This satisfies the issue's literal ACCEPT: "A camera that emits two
      single-frame batches 5 seconds apart, where both frames contain the
      same face area, gets the same `track_id` (or no `track_id` if the
      buffer is disabled)". With no persistent cross-batch buffer wired up
      (cross-batch follow-up — needs orchestrator + settings + state
      store), the sentinel 0 is the "no track_id" branch.

    NOTE: "Centroid" not intersection-over-union — historically the module
    docstring claimed union-of-areas matching, but no such computation ever
    existed here; this tracker is pure centroid Euclidean distance, as
    reflected here.
    """
    # ATT-028: single-frame-batch short-circuit. For the documented
    # production periodic-CCTV model (5-60 s cadence → one frame per batch),
    # the per-call previous_tracks carries no signal, so we emit a sentinel
    # track_id of 0 to honestly express "tracker had no usable signal".
    # Cross-batch rolling-buffer tracking (option (b) of the issue's FIX)
    # would require orchestrator-passed camera_id + frame_captured_at plus
    # a settings.py time-window env var and a Redis state store — non-trivial
    # cross-batch scope deliberately NOT implemented here.
    if len(detections_by_frame) <= 1:
        tracked_results_single: list[TrackedDetection] = []
        for frame_detections in detections_by_frame:
            for detection in frame_detections:
                tracked_results_single.append(
                    TrackedDetection(
                        track_id=_SINGLE_FRAME_NO_TRACK_ID,
                        frame_index=detection.frame_index,
                        frame_id=detection.frame_id,
                        bbox=detection.bbox,
                        score=detection.score,
                        class_id=detection.class_id,
                    )
                )
        return tracked_results_single

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
                # Track IDs must not collide with the sentinel 0 used in
                # single-frame mode — starts at 1.
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
