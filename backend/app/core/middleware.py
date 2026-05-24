"""Request ID injection and structured access logging middleware."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


_ACCESS_LOGGER = logging.getLogger("app.access")


class _RequestIDFilter(logging.Filter):
    """Inject the current request ID into every log record on the thread."""

    _request_id_var: str = "unknown"

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(record, "request_id", self._request_id_var)
        return True


_request_id_filter = _RequestIDFilter()


def _parse_incoming_request_id(header_value: str | None) -> str:
    if header_value:
        try:
            uuid.UUID(header_value)
            return header_value
        except ValueError:
            pass
    return str(uuid.uuid4())


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate or propagate a UUID4 request ID and attach it to state and response headers."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = _parse_incoming_request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        _request_id_filter._request_id_var = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class _MinimalJSONFormatter(logging.Formatter):
    """Emit each log record as a single compact JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        extra_keys = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in logging.LogRecord.__dict__
            and k
            not in {
                "message",
                "asctime",
                "request_id",
                "args",
                "msg",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "funcName",
                "lineno",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "name",
                "stack_info",
                "exc_info",
                "exc_text",
                "taskName",
            }
        }
        payload.update(extra_keys)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_json_access_log() -> None:
    """Attach a JSON formatter and request-ID filter to the app.access logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(_MinimalJSONFormatter())
    handler.addFilter(_request_id_filter)
    _ACCESS_LOGGER.addHandler(handler)
    _ACCESS_LOGGER.propagate = False
    if not _ACCESS_LOGGER.level:
        _ACCESS_LOGGER.setLevel(logging.INFO)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Emit one structured JSON line per completed HTTP request."""

    _SKIP_PATHS: frozenset[str] = frozenset({"/health", "/healthz"})

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self._SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        request_id = getattr(request.state, "request_id", "unknown")
        user_id = getattr(request.state, "user_id", None)

        log_record: dict[str, Any] = {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "request_id": request_id,
        }
        if user_id is not None:
            log_record["user_id"] = str(user_id)

        _ACCESS_LOGGER.info("", extra=log_record)
        return response


__all__ = [
    "AccessLogMiddleware",
    "RequestIDMiddleware",
    "configure_json_access_log",
]
