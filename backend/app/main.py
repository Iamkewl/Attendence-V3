"""FastAPI application factory and runtime middleware wiring for Attendance V3."""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.operations import router as operations_router
from app.api.v1 import api_router, realtime_router
from app.core.database import dispose_engine
from app.core.middleware import AccessLogMiddleware, RequestIDMiddleware, configure_json_access_log
from app.core.security import close_redis, initialize_redis


_LOCALHOST_ORIGIN_PATTERN = re.compile(r"^http://localhost(:\d+)?$")


def _parse_allowed_origins() -> list[str]:
    """Parse and validate explicit CORS origins for credentialed requests."""
    raw_value = os.getenv("ATTENDANCE_ALLOWED_ORIGINS")
    if raw_value is None or not raw_value.strip():
        raise RuntimeError(
            "Environment variable ATTENDANCE_ALLOWED_ORIGINS must define at least one origin."
        )

    origins = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
    if not origins:
        raise RuntimeError("At least one CORS origin must be configured.")

    if "*" in origins:
        raise RuntimeError("Wildcard CORS origins are forbidden when credentials are enabled.")

    for origin in origins:
        if origin.startswith("https://"):
            continue
        if _LOCALHOST_ORIGIN_PATTERN.fullmatch(origin):
            continue
        raise RuntimeError(
            "CORS origins must use HTTPS except for http://localhost or http://localhost:<port>."
        )

    return origins


def _content_security_policy() -> str:
    """Return a strict default CSP unless explicitly overridden by environment variable."""
    return os.getenv(
        "ATTENDANCE_CSP",
        "default-src 'none'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'; "
        "form-action 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "manifest-src 'self'; "
        "object-src 'none'; "
        "upgrade-insecure-requests",
    )


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Initialize and tear down runtime resources used by auth and database layers."""
    configure_json_access_log()
    await initialize_redis()
    try:
        yield
    finally:
        await close_redis()
        await dispose_engine()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Attendance V3 API",
        description=(
            "Secure attendance backend with cookie-based authentication, "
            "JWT rotation, and Redis token revocation."
        ),
        version="3.0.0",
        lifespan=_lifespan,
        openapi_tags=[
            {
                "name": "Authentication",
                "description": (
                    "Session management endpoints with HttpOnly cookie issuance, "
                    "refresh rotation, and logout revocation."
                ),
            },
            {
                "name": "Inference",
                "description": (
                    "Asynchronous AI inference queue endpoints for frame stream and batch processing."
                ),
            },
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_allowed_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )

    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIDMiddleware)

    @app.middleware("http")
    async def inject_security_headers(request: Request, call_next) -> Response:
        """Apply strict response headers to reduce browser-based attack surface."""
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = _content_security_policy()
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response

    @app.get("/healthz", include_in_schema=False)
    async def healthcheck() -> dict[str, str]:
        """Lightweight service health endpoint for probes."""
        return {"status": "ok"}

    app.include_router(operations_router)
    app.include_router(realtime_router)
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()


__all__ = ["app", "create_app"]
