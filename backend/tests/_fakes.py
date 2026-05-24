"""Fake Triton gRPC client for smoke-test injection."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _unit_norm_vector(dim: int = 512, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeTritonGrpcClient:
    """Deterministic stand-in for TritonGrpcClient that never touches the network."""

    def __init__(self, *, zero_detections: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._zero_detections = zero_detections

    def assert_server_ready(self) -> None:
        self.calls.append({"op": "assert_server_ready"})

    def assert_model_ready(self, model_name: str, model_version: str = "") -> None:
        self.calls.append({"op": "assert_model_ready", "model_name": model_name})

    def prime_readiness(self, model_names: Sequence[str]) -> None:
        self.calls.append({"op": "prime_readiness", "model_names": list(model_names)})

    def invalidate_readiness_cache(self) -> None:
        self.calls.append({"op": "invalidate_readiness_cache"})

    def close(self) -> None:
        self.calls.append({"op": "close"})

    async def assert_server_ready_async(self) -> None:
        self.assert_server_ready()

    async def assert_model_ready_async(self, model_name: str, model_version: str = "") -> None:
        self.assert_model_ready(model_name, model_version)

    async def prime_readiness_async(self, model_names: Sequence[str]) -> None:
        self.prime_readiness(model_names)

    async def infer_fp32_async(
        self,
        *,
        model_name: str,
        tensors: dict[str, np.ndarray],
        output_names: Sequence[str] | None = None,
        model_version: str = "",
        request_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, np.ndarray]:
        return self.infer_fp32(
            model_name=model_name,
            tensors=tensors,
            output_names=output_names,
            model_version=model_version,
            request_id=request_id,
            timeout_seconds=timeout_seconds,
        )

    def infer_fp32(
        self,
        *,
        model_name: str,
        tensors: dict[str, np.ndarray],
        output_names: Sequence[str] | None = None,
        model_version: str = "",
        request_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, np.ndarray]:
        self.calls.append(
            {
                "op": "infer_fp32",
                "model_name": model_name,
                "tensor_keys": list(tensors.keys()),
                "output_names": list(output_names) if output_names else None,
            }
        )

        if model_name == "yolov12":
            if self._zero_detections:
                return {"OUTPUT__0": np.empty((1, 0, 6), dtype=np.float32)}
            return {
                "OUTPUT__0": np.array(
                    [[[160.0, 120.0, 220.0, 200.0, 0.92, 0.0]]],
                    dtype=np.float32,
                )
            }

        if model_name == "lvface":
            from app.services.pipeline.settings import get_pipeline_settings

            settings = get_pipeline_settings()
            liveness_key = settings.lvface_liveness_output_name or "liveness"
            embedding_key = settings.lvface_embedding_output_name or "embedding"
            batch_size = next(iter(tensors.values())).shape[0]
            return {
                liveness_key: np.array(
                    [[0.05, 2.5]] * batch_size, dtype=np.float32
                ),
                embedding_key: np.stack(
                    [_unit_norm_vector() for _ in range(batch_size)]
                ),
            }

        return {}
