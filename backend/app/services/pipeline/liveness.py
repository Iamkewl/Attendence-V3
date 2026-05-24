"""Liveness score decoding from LVFace model output tensors."""

from __future__ import annotations

import math

import numpy as np


def _decode_liveness_scores(output_tensor: np.ndarray, expected_count: int) -> list[float]:
    """Decode model outputs into per-face liveness probabilities in [0, 1]."""
    tensor = np.asarray(output_tensor, dtype=np.float32)

    if tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)

    if tensor.shape[0] != expected_count and tensor.ndim == 2 and tensor.shape[1] == expected_count:
        tensor = tensor.T

    if tensor.shape[0] != expected_count:
        raise ValueError(
            "Liveness output batch size does not match requested face batch size."
        )

    scores: list[float] = []
    for row in tensor:
        if row.shape[0] == 1:
            value = float(row[0])
            if value < 0.0 or value > 1.0:
                value = 1.0 / (1.0 + math.exp(-value))
            scores.append(float(np.clip(value, 0.0, 1.0)))
            continue

        logits = row[:2]
        stabilized = logits - float(np.max(logits))
        exp_values = np.exp(stabilized)
        denominator = float(np.sum(exp_values))
        probability_live = 0.0 if denominator <= 0.0 else float(exp_values[1] / denominator)
        scores.append(float(np.clip(probability_live, 0.0, 1.0)))

    return scores
