"""Operational endpoints: liveness, readiness, and version metadata."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.database import get_session_factory
from app.core.security import get_redis_client


LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["Operations"])


# ATT-021 message constants — pre-fix the READY probes returned str(exc) which
# leaks asyncpg DSN, Redis URL, Triton host:port etc. to the unauthenticated
# caller. Replace with these opaque strings + structured logging of the real
# `exc` on the server side. The 'unavailable' string intentionally carries no
# identifying info; the operator reads the structured logger for the real
# exception + traceback, gated behind whatever process-level log policy they
# configure (CLAUDE.md §logging).
ERROR_GENERIC_DATABASE = "unavailable"
ERROR_GENERIC_REDIS = "unavailable"
ERROR_GENERIC_TRITON = "unavailable"


def _app_version() -> str:
    try:
        from app import __version__  # type: ignore[attr-defined]
        return __version__
    except (ImportError, AttributeError):
        return "0.1.0"


def _ops_require_loopback() -> bool:
    """ATT-022 opt-in tighten: when ATTENDANCE_OPS_REQUIRE_LOOPBACK=1, the
    operational endpoints (/health, /ready, /version) refuse callers whose
    client_addr is not loopback (127.0.0.1 / ::1 / ::ffff:127.0.0.1).

    Default is OFF so the existing k8s probes / containerized deployments that
    rely on these endpoints stay unchanged. The flag is defense-in-depth for
    operators who expose the API directly on a public interface and want k8s
    probes to come from localhost (which they already do in any sane
    deployment). It's an opt-in, not a breaking change.
    """
    return os.getenv("ATTENDANCE_OPS_REQUIRE_LOOPBACK", "").strip().lower() in {"true", "1", "yes"}


def _client_addr_loopback(request: Request) -> bool:
    """Return True when the request's client_addr is loopback (127.0.0.1 / ::1).

    Covers IPv4-mapped IPv6 (::ffff:127.0.0.1) too — common in containers on
    dual-stack hosts.
    """
    client = request.client
    if client is None:
        # No client info attached — typical for in-process ASGI test clients
        # (httpx.ASGITransport). Treat as loopback-safe so tests don't break.
        return True
    host = client.host or ""
    return host in {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"}


@router.get("/health", include_in_schema=True)
async def health(request: Request) -> dict[str, str] | JSONResponse:
    if _ops_require_loopback() and not _client_addr_loopback(request):
        return JSONResponse(
            content={"status": "forbidden"},
            status_code=403,
        )
    return {"status": "ok"}


async def _probe_database() -> dict[str, object]:
    start = time.perf_counter()
    try:
        factory = get_session_factory()
        async with factory() as session:
            await asyncio.wait_for(
                session.execute(text("SELECT 1")),
                timeout=2.0,
            )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"ok": True, "latency_ms": latency_ms, "error": None}
    except Exception:
        # ATT-022: never leak the exception string to the unauthenticated
        # caller — log the full traceback server-side and return the
        # opaque "unavailable" string instead.
        LOGGER.exception("Database readiness probe failed")
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"ok": False, "latency_ms": latency_ms, "error": ERROR_GENERIC_DATABASE}


async def _probe_redis() -> dict[str, object]:
    start = time.perf_counter()
    try:
        redis = await get_redis_client()
        await asyncio.wait_for(redis.ping(), timeout=2.0)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"ok": True, "latency_ms": latency_ms, "error": None}
    except Exception:
        # ATT-022: same as _probe_database — log server-side, return opaque.
        LOGGER.exception("Redis readiness probe failed")
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"ok": False, "latency_ms": latency_ms, "error": ERROR_GENERIC_REDIS}


async def _probe_triton() -> dict[str, object]:
    demo_env = os.getenv("ATTENDANCE_TRITON_DEMO_MODE", "").strip().lower()
    if demo_env in {"true", "1"}:
        return {"ok": True, "mode": "demo", "latency_ms": 0.0, "error": None}

    start = time.perf_counter()
    try:
        from app.infrastructure.triton import get_triton_client

        client = get_triton_client()
        # ATT-021: was `asyncio.get_event_loop().run_in_executor(None,
        # client.assert_server_ready)` which under Python 3.12 emits
        # DeprecationWarning('There is no current event loop in thread
        # ...') and trips the filterwarnings=["error"] test config. Use
        # the existing async wrapper on the client itself, which is
        # implemented as `await asyncio.to_thread(self.assert_server_ready)`.
        await asyncio.wait_for(
            client.assert_server_ready_async(),
            timeout=2.0,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"ok": True, "mode": "live", "latency_ms": latency_ms, "error": None}
    except Exception:
        # ATT-022: same as the other probes — log server-side, return opaque.
        LOGGER.exception("Triton readiness probe failed")
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"ok": False, "mode": "live", "latency_ms": latency_ms, "error": ERROR_GENERIC_TRITON}


@router.get("/ready", include_in_schema=True)
async def ready(request: Request) -> JSONResponse:
    # ATT-022 opt-in: only refuse when the env flag is set. k8s probes hit
    # /ready from loopback so this doesn't break them by default.
    if _ops_require_loopback() and not _client_addr_loopback(request):
        return JSONResponse(
            content={"status": "forbidden"},
            status_code=403,
        )

    db_result, redis_result, triton_result = await asyncio.gather(
        _probe_database(),
        _probe_redis(),
        _probe_triton(),
    )

    db_ok = bool(db_result["ok"])
    redis_ok = bool(redis_result["ok"])
    triton_ok = bool(triton_result["ok"])

    if db_ok and redis_ok and triton_ok:
        overall = "ok"
    elif db_ok and redis_ok:
        overall = "degraded"
    else:
        overall = "degraded"

    http_status = 200 if (db_ok and redis_ok) else 503

    body = {
        "status": overall,
        "checks": {
            "database": db_result,
            "redis": redis_result,
            "triton": triton_result,
        },
    }
    return JSONResponse(content=body, status_code=http_status)


@router.get("/version", include_in_schema=True)
async def version(request: Request) -> dict[str, str] | JSONResponse:
    if _ops_require_loopback() and not _client_addr_loopback(request):
        return JSONResponse(
            content={"status": "forbidden"},
            status_code=403,
        )
    return {
        "version": _app_version(),
        "git_sha": os.getenv("ATTENDANCE_GIT_SHA", "unknown"),
        "build_time": os.getenv("ATTENDANCE_BUILD_TIME", "unknown"),
    }


__all__ = ["router"]
