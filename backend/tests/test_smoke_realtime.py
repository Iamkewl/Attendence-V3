"""Flows 5 and 6 — WebSocket ticket gate and SSE real-newline regression."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Flow 5 — WebSocket accepts a valid ticket without closing 1008
# ---------------------------------------------------------------------------

def test_websocket_accepts_valid_ticket() -> None:
    """Ticket issued in Redis is consumed by /ws/live without a 1008 rejection.

    This test is intentionally synchronous: starlette.testclient.TestClient
    uses anyio internally and cannot run inside an already-running asyncio
    event loop (the pytest-asyncio session loop). A sync test avoids the
    conflict.  We issue the ticket via a synchronous Redis connection instead
    of the session-scoped async redis_client fixture.

    The TestClient is NOT entered as a context manager: that would trigger
    FastAPI's lifespan (initialize_redis + dispose_engine on shutdown), which
    tries to close asyncpg connections from a different event loop than the
    one that created them. WebSocket routes don't require lifespan startup
    — the pubsub manager and connection manager are module-level singletons.

    Earlier async tests in the session populated the shared Redis client
    (`app.core.security._redis_client`) on the pytest-asyncio session loop.
    TestClient runs the ASGI app in its OWN anyio loop, so we must reset
    every cached async-Redis singleton beforehand. The fresh client is then
    lazily created on TestClient's loop where it can actually be awaited.
    """
    import redis
    import app.core.security as security
    import app.api.v1.websockets as ws_module
    from app.core.pubsub import websocket_ticket_key
    from app.main import create_app
    from starlette.testclient import TestClient

    ticket = str(uuid.uuid4())

    # Issue ticket synchronously (same Redis instance, different client type).
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    r.set(websocket_ticket_key(ticket), "smoke", ex=30, nx=True)

    # Drop cached async Redis clients so TestClient's loop creates fresh ones.
    security._redis_client = None
    ws_module._pubsub_manager._redis_client = None

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    try:
        with client.websocket_connect(f"/ws/live?ticket={ticket}") as ws:
            ws.close()
    except Exception as exc:
        pytest.fail(f"WebSocket connection raised an unexpected exception: {exc}")
    finally:
        # Drop the TestClient-loop client too — anything that runs after this
        # test in the session loop would otherwise inherit a dead reference.
        security._redis_client = None
        ws_module._pubsub_manager._redis_client = None


# ---------------------------------------------------------------------------
# Flow 6 — SSE regression: real LF bytes, not literal backslash-n
# ---------------------------------------------------------------------------

def test_sse_uses_real_newlines() -> None:
    # AST-walk websockets.py to find every constant or f-string yielded as an
    # SSE frame, evaluate it, and assert the resulting bytes contain real LF
    # (0x0a) and no literal backslash-n. We assert on the parsed source rather
    # than streaming because the TestClient + SSE generator teardown deadlocks:
    # the broadcast loop blocks on Redis pubsub and the generator's finally
    # cannot run until the next message arrives.
    import ast

    source = Path("backend/app/api/v1/websockets.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    def _extract_str(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                else:
                    parts.append("<expr>")  # placeholder for formatted expression
            return "".join(parts)
        return None

    sse_frames: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Yield) and node.value is not None:
            value = _extract_str(node.value)
            if value and ("event:" in value or "data:" in value):
                sse_frames.append(value)

    assert sse_frames, "Expected to find at least one SSE yield statement"

    for frame in sse_frames:
        assert "\\n" not in frame, (
            f"SSE yield evaluates to a literal backslash-n (regression): {frame!r}"
        )
        assert "\n" in frame, (
            f"SSE yield must contain real newline bytes: {frame!r}"
        )
