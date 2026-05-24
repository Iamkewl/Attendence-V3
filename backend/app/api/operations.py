"""Operational endpoints: liveness, readiness, and version metadata."""

from __future__ import annotations

import asyncio
import os
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.database import get_session_factory
from app.core.security import get_redis_client


router = APIRouter(tags=["Operations"])


def _app_version() -> str:
    try:
        from app import __version__  # type: ignore[attr-defined]
        return __version__
    except (ImportError, AttributeError):
        return "0.1.0"


@router.get("/health", include_in_schema=True)
async def health() -> dict[str, str]:
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
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"ok": False, "latency_ms": latency_ms, "error": str(exc)}


async def _probe_redis() -> dict[str, object]:
    start = time.perf_counter()
    try:
        redis = await get_redis_client()
        await asyncio.wait_for(redis.ping(), timeout=2.0)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"ok": True, "latency_ms": latency_ms, "error": None}
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"ok": False, "latency_ms": latency_ms, "error": str(exc)}


async def _probe_triton() -> dict[str, object]:
    demo_env = os.getenv("ATTENDANCE_TRITON_DEMO_MODE", "").strip().lower()
    if demo_env in {"true", "1"}:
        return {"ok": True, "mode": "demo", "latency_ms": 0.0, "error": None}

    start = time.perf_counter()
    try:
        from app.infrastructure.triton import get_triton_client

        client = get_triton_client()
        await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, client.assert_server_ready),
            timeout=2.0,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"ok": True, "mode": "live", "latency_ms": latency_ms, "error": None}
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"ok": False, "mode": "live", "latency_ms": latency_ms, "error": str(exc)}


@router.get("/ready", include_in_schema=True)
async def ready() -> JSONResponse:
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
async def version() -> dict[str, str]:
    return {
        "version": _app_version(),
        "git_sha": os.getenv("ATTENDANCE_GIT_SHA", "unknown"),
        "build_time": os.getenv("ATTENDANCE_BUILD_TIME", "unknown"),
    }


__all__ = ["router"]
