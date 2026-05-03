"""Robust Triton gRPC client wrapper with retry-aware inference helpers."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

import grpc
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


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


@dataclass(frozen=True, slots=True)
class TritonClientSettings:
    """Runtime settings controlling Triton connectivity and retry behavior."""

    url: str
    request_timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    ssl_enabled: bool


def _read_required_env(name: str) -> str:
    """Read a required environment variable and fail fast when not configured."""
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Environment variable {name} must be set.")
    return value.strip()


def _read_float_env(name: str, default: float, *, min_value: float) -> float:
    """Read a floating-point environment variable with lower-bound validation."""
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a floating-point number.") from exc

    if value < min_value:
        raise RuntimeError(
            f"Environment variable {name} must be greater than or equal to {min_value}."
        )

    return value


def _read_int_env(name: str, default: int, *, min_value: int) -> int:
    """Read an integer environment variable with explicit lower-bound validation."""
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer.") from exc

    if value < min_value:
        raise RuntimeError(
            f"Environment variable {name} must be greater than or equal to {min_value}."
        )

    return value


def _read_bool_env(name: str, default: bool) -> bool:
    """Read a boolean environment variable using common truthy and falsy values."""
    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise RuntimeError(f"Environment variable {name} must be a boolean value.")


@lru_cache(maxsize=1)
def get_triton_client_settings() -> TritonClientSettings:
    """Return cached Triton client settings sourced from environment variables."""
    return TritonClientSettings(
        url=_read_required_env("ATTENDANCE_TRITON_URL"),
        request_timeout_seconds=_read_float_env(
            "ATTENDANCE_TRITON_REQUEST_TIMEOUT_SECONDS",
            10.0,
            min_value=0.1,
        ),
        max_retries=_read_int_env("ATTENDANCE_TRITON_MAX_RETRIES", 3, min_value=0),
        retry_backoff_seconds=_read_float_env(
            "ATTENDANCE_TRITON_RETRY_BACKOFF_SECONDS",
            0.35,
            min_value=0.05,
        ),
        ssl_enabled=_read_bool_env("ATTENDANCE_TRITON_SSL_ENABLED", False),
    )


class TritonGrpcClient:
    """High-level Triton gRPC helper that enforces readiness checks and retry policy."""

    def __init__(self, settings: TritonClientSettings | None = None) -> None:
        self._settings = settings or get_triton_client_settings()
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

    def assert_model_ready(self, model_name: str, model_version: str = "") -> None:
        """Ensure the specified model is available and ready for inference."""
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

        for attempt in range(1, max_attempts + 1):
            try:
                return operation()
            except TritonTimeoutError:
                if attempt >= max_attempts:
                    raise
                time.sleep(self._settings.retry_backoff_seconds * (2 ** (attempt - 1)))
            except (InferenceServerException, grpc.RpcError, OSError) as exc:
                if not self._is_retryable_exception(exc) or attempt >= max_attempts:
                    raise TritonInferenceError(
                        f"{operation_name} failed after {attempt} attempt(s): {exc}"
                    ) from exc
                time.sleep(self._settings.retry_backoff_seconds * (2 ** (attempt - 1)))
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
def get_triton_client() -> TritonGrpcClient:
    """Create and cache a singleton Triton gRPC wrapper instance."""
    return TritonGrpcClient()


def clear_triton_client_cache() -> None:
    """Clear cached Triton wrapper and settings to support dynamic test/runtime reconfiguration."""
    cached_client = get_triton_client.cache_info().currsize
    if cached_client:
        client = get_triton_client()
        client.close()
    get_triton_client.cache_clear()
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
]
