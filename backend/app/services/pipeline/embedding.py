"""Embedding decoding and deterministic identity token generation."""

from __future__ import annotations

import hashlib

import numpy as np


def _decode_embeddings(output_tensor: np.ndarray, expected_count: int) -> list[np.ndarray]:
    """Decode output tensor into L2-normalized embedding vectors per tracked face."""
    tensor = np.asarray(output_tensor, dtype=np.float32)

    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)

    if tensor.shape[0] != expected_count and tensor.ndim == 2 and tensor.shape[1] == expected_count:
        tensor = tensor.T

    if tensor.shape[0] != expected_count:
        raise ValueError(
            "Recognition output batch size does not match requested face batch size."
        )

    embeddings: list[np.ndarray] = []
    for row in tensor:
        norm = float(np.linalg.norm(row))
        if norm <= 1e-8:
            normalized = np.zeros_like(row, dtype=np.float32)
        else:
            normalized = (row / norm).astype(np.float32)
        embeddings.append(normalized)

    return embeddings


def _identity_from_embedding(embedding: np.ndarray) -> str:
    """Generate deterministic identity token from embedding bytes for downstream matching."""
    digest = hashlib.sha256(embedding.tobytes()).hexdigest()
    return f"identity_{digest[:16]}"
