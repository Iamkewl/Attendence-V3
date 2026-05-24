"""Robust Triton gRPC client wrapper with retry-aware inference helpers."""

from __future__ import annotations

import asyncio
import time
from functools import lru_cache
from typing import Callable, Sequence

import grpc
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from .settings import TritonClientSettings, get_triton_client_settings


class TritonClientError(RuntimeError):
    """Base exception raised for Triton client wrapper failures."""


class TritonServerUnavailableError(TritonClientError):
    """Raised when Triton server liveness/readiness checks fail."""


class TritonModelUnavailableError(TritonClientError):
    """Raised when the requested Triton model is unavailable."""


class TritonTimeoutError(TritonClientError):
    """Raised when Triton requests exceed configured timeout boundaries."""


class TritonInferenceError(TritonClientError):
    """Raised when inference execution fails and cannot be retried."""


class TritonGrpcClient:
    """High-level Triton gRPC helper that enforces readiness checks and retry policy."""

    def __init__(self, settings: TritonClientSettings | None = None) -> None:
        self._settings = settings or get_triton_client_settings()
        self._server_ready: bool = False
        self._ready_models: set[tuple[str, str]] = set()
        try:
            self._client = grpcclient.InferenceServerClient(
                url=self._settings.url,
                verbose=False,
                ssl=self._settings.ssl_enabled,
            )
        except Exception as exc:
            raise TritonServerUnavailableError(
                f"Failed to initialize Triton gRPC client for endpoint {self._settings.url}."
            ) from exc

    def close(self) -> None:
        """Close the underlying Triton client transport if supported by the client implementation."""
        close_method = getattr(self._client, "close", None)
        if callable(close_method):
            close_method()

    def assert_server_ready(self) -> None:
        """Ensure Triton server is live and ready before submitting inference requests."""
        if self._server_ready:
            return

        def check_live() -> bool:
            return bool(self._client.is_server_live())

        def check_ready() -> bool:
            return bool(self._client.is_server_ready())

        is_live = self._run_with_retries(check_live, "server liveness check")
        is_ready = self._run_with_retries(check_ready, "server readiness check")

        if not is_live:
            raise TritonServerUnavailableError("Triton server is not live.")

        if not is_ready:
            raise TritonServerUnavailableError("Triton server is live but not ready.")

        self._server_ready = True

    def assert_model_ready(self, model_name: str, model_version: str = "") -> None:
        """Ensure the specified model is available and ready for inference."""
        cache_key = (model_name, model_version)
        if cache_key in self._ready_models:
            return

        self.assert_server_ready()

        def check_model_ready() -> bool:
            return bool(
                self._client.is_model_ready(
                    model_name=model_name,
                    model_version=model_version,
                )
            )

        is_model_ready = self._run_with_retries(
            check_model_ready,
            f"model readiness check for {model_name}",
        )

        if not is_model_ready:
            if model_version:
                raise TritonModelUnavailableError(
                    f"Model {model_name} version {model_version} is not ready."
                )
            raise TritonModelUnavailableError(f"Model {model_name} is not ready.")

        self._ready_models.add(cache_key)

    def invalidate_readiness_cache(self) -> None:
        """Clear cached readiness state so subsequent calls re-verify against the server."""
        self._server_ready = False
        self._ready_models.clear()

    def prime_readiness(self, model_names: Sequence[str]) -> None:
        """Pre-verify server and named models once so inference paths skip readiness pings."""
        self.assert_server_ready()
        for model_name in model_names:
            self.assert_model_ready(model_name=model_name)

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
        """Run an FP32 Triton inference call and return all requested outputs as NumPy arrays."""
        if not tensors:
            raise ValueError("At least one input tensor must be provided.")

        self.assert_model_ready(model_name=model_name, model_version=model_version)

        infer_inputs: list[grpcclient.InferInput] = []
        for input_name, tensor in tensors.items():
            infer_inputs.append(self._numpy_to_infer_input(name=input_name, tensor=tensor))

        requested_outputs = None
        if output_names:
            requested_outputs = [grpcclient.InferRequestedOutput(name) for name in output_names]

        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else self._settings.request_timeout_seconds
        )

        def perform_inference() -> grpcclient.InferResult:
            try:
                return self._client.infer(
                    model_name=model_name,
                    model_version=model_version,
                    inputs=infer_inputs,
                    outputs=requested_outputs,
                    request_id=request_id,
                    client_timeout=effective_timeout,
                )
            except InferenceServerException as exc:
                if self._looks_like_timeout(exc):
                    raise TritonTimeoutError(
                        f"Inference timed out for model {model_name} after {effective_timeout}s."
                    ) from exc
                raise
            except grpc.RpcError as exc:
                if exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise TritonTimeoutError(
                        f"Inference timed out for model {model_name} after {effective_timeout}s."
                    ) from exc
                raise

        result = self._run_with_retries(perform_inference, f"inference call to model {model_name}")
        return self._parse_infer_outputs(result=result)

    async def assert_server_ready_async(self) -> None:
        """Async wrapper for assert_server_ready; safe to call from the FastAPI event loop."""
        await asyncio.to_thread(self.assert_server_ready)

    async def assert_model_ready_async(self, model_name: str, model_version: str = "") -> None:
        """Async wrapper for assert_model_ready; safe to call from the FastAPI event loop."""
        await asyncio.to_thread(self.assert_model_ready, model_name, model_version)

    async def prime_readiness_async(self, model_names: Sequence[str]) -> None:
        """Async wrapper for prime_readiness; safe to call from the FastAPI event loop."""
        await asyncio.to_thread(self.prime_readiness, model_names)

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
        """Async wrapper for infer_fp32; safe to call from the FastAPI event loop."""
        return await asyncio.to_thread(
            lambda: self.infer_fp32(
                model_name=model_name,
                tensors=tensors,
                output_names=output_names,
                model_version=model_version,
                request_id=request_id,
                timeout_seconds=timeout_seconds,
            )
        )

    def _numpy_to_infer_input(self, *, name: str, tensor: np.ndarray) -> grpcclient.InferInput:
        """Convert a NumPy tensor to Triton's FP32 InferInput payload."""
        contiguous_tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        if contiguous_tensor.size == 0:
            raise ValueError(f"Tensor '{name}' must not be empty.")

        infer_input = grpcclient.InferInput(name, contiguous_tensor.shape, "FP32")
        infer_input.set_data_from_numpy(contiguous_tensor, binary_data=True)
        return infer_input

    def _parse_infer_outputs(self, *, result: grpcclient.InferResult) -> dict[str, np.ndarray]:
        """Extract every output tensor from Triton inference result as NumPy arrays."""
        infer_response = result.get_response()
        output_names = [output.name for output in infer_response.outputs]
        if not output_names:
            raise TritonInferenceError("Triton response did not include any outputs.")

        parsed_outputs: dict[str, np.ndarray] = {}
        for output_name in output_names:
            output_tensor = result.as_numpy(output_name)
            if output_tensor is None:
                raise TritonInferenceError(
                    f"Triton response output '{output_name}' could not be decoded to NumPy."
                )
            parsed_outputs[output_name] = output_tensor

        return parsed_outputs

    def _run_with_retries(self, operation: Callable[[], object], operation_name: str) -> object:
        """Execute an operation with exponential backoff for retryable failures."""
        max_attempts = self._settings.max_retries + 1
        deadline = time.monotonic() + self._settings.total_retry_budget_seconds

        for attempt in range(1, max_attempts + 1):
            try:
                return operation()
            except TritonTimeoutError:
                if attempt >= max_attempts:
                    raise
                self.invalidate_readiness_cache()
                sleep_duration = self._settings.retry_backoff_seconds * (2 ** (attempt - 1))
                if time.monotonic() + sleep_duration >= deadline:
                    raise
                time.sleep(sleep_duration)
            except (InferenceServerException, grpc.RpcError, OSError) as exc:
                if not self._is_retryable_exception(exc) or attempt >= max_attempts:
                    raise TritonInferenceError(
                        f"{operation_name} failed after {attempt} attempt(s): {exc}"
                    ) from exc
                self.invalidate_readiness_cache()
                sleep_duration = self._settings.retry_backoff_seconds * (2 ** (attempt - 1))
                if time.monotonic() + sleep_duration >= deadline:
                    raise TritonInferenceError(
                        f"{operation_name} exceeded retry budget of"
                        f" {self._settings.total_retry_budget_seconds}s after {attempt} attempt(s): {exc}"
                    ) from exc
                time.sleep(sleep_duration)
            except Exception as exc:
                raise TritonInferenceError(f"{operation_name} failed: {exc}") from exc

        raise TritonInferenceError(f"{operation_name} failed unexpectedly.")

    @staticmethod
    def _is_retryable_exception(exc: InferenceServerException | grpc.RpcError | OSError) -> bool:
        """Return whether an exception likely represents transient infrastructure failure."""
        if isinstance(exc, grpc.RpcError):
            return exc.code() in {
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                grpc.StatusCode.INTERNAL,
                grpc.StatusCode.ABORTED,
            }

        normalized = str(exc).lower()
        if isinstance(exc, OSError):
            return True

        retryable_markers = (
            "unavailable",
            "deadline exceeded",
            "timeout",
            "temporarily",
            "connection refused",
            "connection reset",
            "resource exhausted",
        )
        return any(marker in normalized for marker in retryable_markers)

    @staticmethod
    def _looks_like_timeout(exc: InferenceServerException) -> bool:
        """Detect timeout-oriented Triton server exceptions by normalized error message."""
        normalized = str(exc).lower()
        return "deadline" in normalized or "timeout" in normalized


@lru_cache(maxsize=1)
def _build_triton_client() -> TritonGrpcClient:
    return TritonGrpcClient()


# Explicit test seam — only ever set by test fixtures, never in production code.
_test_client_override: TritonGrpcClient | None = None


def set_triton_client_override(client: TritonGrpcClient | None) -> None:
    """Replace the module-level singleton for test isolation.

    Call with a fake client before each test, restore with None after yield.
    Works for both FastAPI-path tests and Celery eager-mode tests since it
    bypasses the lru_cache without monkeypatching.
    """
    global _test_client_override
    _test_client_override = client


def get_triton_client() -> TritonGrpcClient:
    """Return the active Triton client — the test override if set, the singleton otherwise."""
    if _test_client_override is not None:
        return _test_client_override
    return _build_triton_client()


def clear_triton_client_cache() -> None:
    """Clear cached Triton wrapper and settings to support dynamic runtime reconfiguration."""
    cached_client = _build_triton_client.cache_info().currsize
    if cached_client:
        client = _build_triton_client()
        client.close()
    _build_triton_client.cache_clear()
    get_triton_client_settings.cache_clear()


__all__ = [
    "TritonClientError",
    "TritonGrpcClient",
    "TritonInferenceError",
    "TritonModelUnavailableError",
    "TritonServerUnavailableError",
    "TritonTimeoutError",
    "clear_triton_client_cache",
    "get_triton_client",
    "get_triton_client_settings",
    "set_triton_client_override",
]
