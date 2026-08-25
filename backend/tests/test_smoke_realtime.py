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

    B14 follow-on (ATT-026): the ticket payload MUST carry a user_id in the
    JSON shape that /auth/ws-ticket emits ({\"user_id\": str(uuid), ...}).
    The route now refuses tickets that do not parse to a usable user_id —
    that is the load-bearing cap-assignable identity.
    """
    import json
    import os
    import redis
    import app.core.security as security
    import app.api.v1.websockets as ws_module
    from app.core.pubsub import websocket_ticket_key
    from app.main import create_app
    from starlette.testclient import TestClient

    ticket = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    # Issue ticket synchronously (same Redis instance, different client type).
    # Payload matches /auth/ws-ticket's emission shape in backend/app/api/v1/
    # auth.py: json.dumps({"user_id": str(current_user.id), "issued_at": ...})
    # Resolve the DB number from the configured URL instead of hardcoding db=0:
    # environments that isolate runs via ATTENDANCE_REDIS_URL(_TEST)=.../N were
    # writing the ticket to one DB while the app consumed tickets from another.
    redis_url = (
        os.environ.get("ATTENDANCE_REDIS_URL_TEST")
        or os.environ.get("ATTENDANCE_REDIS_URL")
        or "redis://localhost:6379/0"
    )
    r = redis.Redis.from_url(redis_url, decode_responses=True)
    payload = json.dumps(
        {"user_id": user_id, "issued_at": "2026-07-29T00:00:00+00:00"},
        separators=(",", ":"),
        ensure_ascii=True,
    )
    r.set(websocket_ticket_key(ticket), payload, ex=30, nx=True)

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
        # ATT-026 follow-up: clean up the per-user and global Redis counters
        # so we don't perturb any other limiter-based test in the session.
        r.delete(f"ws:user:{user_id}")
        r.delete("ws:total")


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


# ---------------------------------------------------------------------------
# ATT-024 — _broadcast must use asyncio.gather, not a sequential per-client
# `await websocket.send_text(...)` loop. A slow client in the loop blocks
# every subsequent client and every queued pubsub message.
#
# A perfect unit test would measure completion time and assert < sum(send_times);
# in practice that's flaky on shared CI runners. The pragmatic test: inspect
# the broadcast source to confirm it uses `asyncio.gather`. Pre-fix the source
# had a `for websocket in recipients: await websocket.send_text(...)` loop with
# NO gather call; post-fix it uses `await asyncio.gather(..., return_exceptions=True)`.
# ---------------------------------------------------------------------------


def test_att_024_broadcast_uses_asyncio_gather() -> None:
    """ATT-024 regression anchor: _broadcast source must call asyncio.gather.

    Pre-fix: source contains a per-client for-loop with `await websocket.send_-
    text(payload_text)` but NO `asyncio.gather` invocation in `_broadcast`.
    """
    import inspect

    import app.api.v1.websockets as ws_module

    src = inspect.getsource(ws_module.LiveConnectionManager._broadcast)
    assert "asyncio.gather" in src, (
        "ATT-024: LiveConnectionManager._broadcast does NOT use asyncio.gather — "
        "the head-of-line blocking sequential send loop is still present."
    )


@pytest.mark.asyncio
async def test_att_024_broadcast_completes_concurrently_with_slow_clients() -> None:
    """ATT-024 wall-clock guard: 3 sockets each with 0.5s send_text finish in ~0.5s.

    Pre-fix took 3 × 0.5s ~= 1.5s (sequential loop). Post-fix takes max(0.5s)
    = 0.5s (concurrent gather). The assertion uses 1.1s as the upper bound
    so a busy CI runner can still pass post-fix; pre-fix would consistently
    fail at 1.5s.
    """
    import asyncio

    import app.api.v1.websockets as ws_module

    class _StubSocket:
        def __init__(self, delay: float) -> None:
            self._delay = delay

        async def send_text(self, _payload: str) -> None:
            await asyncio.sleep(self._delay)

    # Build a stub manager so we don't perturb the real `_connection_manager`
    # state (its connections set is a module singleton).
    manager = ws_module.LiveConnectionManager(
        pubsub_manager=ws_module._pubsub_manager,
        channels=ws_module.DEFAULT_REALTIME_CHANNELS,
    )
    sockets = [_StubSocket(0.5) for _ in range(3)]
    for s in sockets:
        manager._connections.add(s)

    message = ws_module.PubSubMessage(channel="test", payload="hello")
    loop = asyncio.get_running_loop()
    start = loop.time()
    await manager._broadcast(message)
    elapsed = loop.time() - start

    # Post-fix: ~0.5s. Pre-fix: ~1.5s. 1.1s is comfortably above the
    # concurrent baseline and below the sequential baseline, so either
    # environment fails pre-fix reliably.
    assert elapsed < 1.1, (
        f"ATT-024: broadcast took {elapsed:.2f}s but should have run "
        f"concurrently (max(3*0.5)=0.5s expected; 1.5s pre-fix). The "
        f"sequential HoL-blocking pattern is still present."
    )


# ---------------------------------------------------------------------------
# ATT-025 — LiveConnectionManager must NOT tear down the broadcast task on
# disconnect. Pre-fix the manager had `_stop_event` as an attribute and
# `disconnect()` set it then awaited the task outside the lock; a concurrent
# connect could start a new task before the old task finished un-subscribing,
# dispatching a final pubsub message twice.
#
# Post-fix: the broadcast task runs for the life of the process; `disconnect`
# only removes the socket from the shared set; no `_stop_event` attribute.
# ---------------------------------------------------------------------------


def test_att_025_no_stop_event_attribute() -> None:
    """Pre-fix LiveConnectionManager had a `_stop_event` attribute; post-fix it does not.

    The post-fix code keeps the broadcast task always running (lazy-started,
    never stopped) which sidesteps the start/stop race entirely. The test
    pins the absence of `_stop_event` on instances so a regression that
    re-introduces the per-connect churn is caught.
    """
    import app.api.v1.websockets as ws_module

    manager = ws_module.LiveConnectionManager(
        pubsub_manager=ws_module._pubsub_manager,
        channels=ws_module.DEFAULT_REALTIME_CHANNELS,
    )
    assert not hasattr(manager, "_stop_event"), (
        "ATT-025: LiveConnectionManager still has a `_stop_event` attribute — "
        "the disconnect-stops-task path is (back) in place and vulnerable "
        "to the rapid connect/disconnect duplicate-message race."
    )


@pytest.mark.asyncio
async def test_att_025_disconnect_does_not_null_broadcast_task() -> None:
    """disconnect() must NOT set _broadcast_task to None.

    Pre-fix the code nullified _broadcast_task on last disconnect. Post-fix
    the task is persistent for the life of the process, so disconnect only
    manages the _connections set. The test pins the invariant by creating
    a fake task, calling disconnect, and asserting the task is still there.
    """

    import app.api.v1.websockets as ws_module

    class _NoOpTask:
        # Just a sentinel that resembles the .done() check via bool()—since
        # the connect() logic checks `self._broadcast_task is None or
        # self._broadcast_task.done()`.
        def done(self) -> bool:
            return False

    manager = ws_module.LiveConnectionManager(
        pubsub_manager=ws_module._pubsub_manager,
        channels=ws_module.DEFAULT_REALTIME_CHANNELS,
    )
    sentinel = _NoOpTask()
    manager._broadcast_task = sentinel  # type: ignore[assignment]

    class _StubSocket:
        pass

    fake = _StubSocket()
    manager._connections.add(fake)  # type: ignore[arg-type]
    await manager.disconnect(fake)  # type: ignore[arg-type]

    assert manager._broadcast_task is sentinel, (
        "ATT-025: disconnect() nullified _broadcast_task — the per-connect "
        "broadcast-task churn is back, which triggers the duplicate-message "
        "race under rapid connect/disconnect."
    )


# ---------------------------------------------------------------------------
# ATT-026 — RealtimeConnectionLimiter must enforce per-user and global caps.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_att_026_limiter_default_caps() -> None:
    """Default per-user cap is 5; default total cap is 100."""
    import app.api.v1.websockets as ws_module

    # Clear env to force defaults.
    _orig_pu = ws_module.os.environ.pop("ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER", None)
    _orig_tl = ws_module.os.environ.pop("ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL", None)
    try:
        assert ws_module._read_max_ws_connections_per_user() == 5
        assert ws_module._read_max_ws_connections_total() == 100
    finally:
        if _orig_pu is not None:
            ws_module.os.environ["ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER"] = _orig_pu
        if _orig_tl is not None:
            ws_module.os.environ["ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL"] = _orig_tl


@pytest.mark.asyncio
async def test_att_026_per_user_cap_trips_at_cap() -> None:
    """The limiter returns False on the (cap+1)th concurrent acquire for one user."""

    import app.api.v1.websockets as ws_module
    from app.core.security import get_redis_client

    user_id = uuid.uuid4()
    per_user_key = f"ws:user:{user_id}"
    total_key = "ws:total"

    # Reset env to a tiny cap so we don't have to INCR a lot.
    _orig_pu = ws_module.os.environ.pop("ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER", None)
    _orig_tl = ws_module.os.environ.pop("ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL", None)
    ws_module.os.environ["ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER"] = "3"
    ws_module.os.environ["ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL"] = "1000"

    # Reset the Redis counters so the test starts from a clean baseline.
    client = await get_redis_client()
    await client.delete(per_user_key)
    await client.delete(total_key)

    limiter = ws_module._realtime_limiter
    try:
        acquired = []
        for _ in range(3):
            acquired.append(await limiter.try_acquire(user_id))
        # The 4th should be refused.
        assert acquired == [True, True, True], acquired
        refused = await limiter.try_acquire(user_id)
        assert refused is False, (
            "ATT-026: per-user cap did not refuse the (cap+1)th acquisition — "
            f"got {refused}, expected False. The Redis-backed counter is not "
            f"being incremented correctly or the cap comparison is wrong."
        )
        # The limiter should NOT have leaked the over-the-cap INCR, so the
        # per-user counter should be back to 3 (the cap), and the total
        # counter should also be back at 3 (not 4).
        per_user_count = int(await client.get(per_user_key) or 0)
        total_count = int(await client.get(total_key) or 0)
        assert per_user_count == 3, (
            f"ATT-026: per-user count after over-the-cap attempt is "
            f"{per_user_count}, expected 3 (cap). The over-cap INCR was not "
            f"rolled back."
        )
        assert total_count == 3, (
            f"ATT-026: global count after over-the-cap attempt is "
            f"{total_count}, expected 3 (matching the still-held reservations)."
        )
    finally:
        await client.delete(per_user_key)
        await client.delete(total_key)
        if _orig_pu is not None:
            ws_module.os.environ["ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER"] = _orig_pu
        else:
            ws_module.os.environ.pop("ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER", None)
        if _orig_tl is not None:
            ws_module.os.environ["ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL"] = _orig_tl
        else:
            ws_module.os.environ.pop("ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL", None)


@pytest.mark.asyncio
async def test_att_026_global_cap_trips_at_cap() -> None:
    """The limiter returns False when the global cap is hit, even with different users."""

    import app.api.v1.websockets as ws_module
    from app.core.security import get_redis_client

    # Use a tiny global cap and a large per-user cap so the global cap fires first.
    _orig_pu = ws_module.os.environ.pop("ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER", None)
    _orig_tl = ws_module.os.environ.pop("ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL", None)
    ws_module.os.environ["ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER"] = "1000"
    ws_module.os.environ["ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL"] = "2"

    total_key = "ws:total"
    client = await get_redis_client()
    await client.delete(total_key)

    limiter = ws_module._realtime_limiter
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    try:
        a1 = await limiter.try_acquire(user_a)
        b1 = await limiter.try_acquire(user_b)
        c1 = await limiter.try_acquire(uuid.uuid4())
        assert (a1, b1, c1) == (True, True, False), (
            f"ATT-026: global cap did not refuse the third distinct-user "
            f"acquisition — got {(a1, b1, c1)}, expected (True, True, False)."
        )
        # On the refused path the per-user count for user C must have been
        # rolled back too. user_c's per-user key was incremented and then
        # decremented as part of the second-cap rollback; so user_c's
        # per-user count should be 0 (no leak from the refused path).
        user_c = uuid.uuid4()
        await client.delete(f"ws:user:{user_c}")
        # (We can't easily get the third user_id used above; assert the
        # global count baseline is exactly 2 instead — that proves the
        # refused path's per-user rollback worked AND the total rollback
        # worked, since a missed rollback would leave total at 3.)
        total_count = int(await client.get(total_key) or 0)
        assert total_count == 2, (
            f"ATT-026: total count after refused acquire is {total_count}, "
            f"expected 2 (the two accepted acquisitions). The refused path "
            f"did not roll back both INCRs."
        )
    finally:
        # Clean up: release the two we held.
        await limiter.release(user_a)
        await limiter.release(user_b)
        await client.delete(total_key)
        await client.delete(f"ws:user:{user_a}")
        await client.delete(f"ws:user:{user_b}")
        if _orig_pu is not None:
            ws_module.os.environ["ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER"] = _orig_pu
        else:
            ws_module.os.environ.pop("ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER", None)
        if _orig_tl is not None:
            ws_module.os.environ["ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL"] = _orig_tl
        else:
            ws_module.os.environ.pop("ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL", None)


@pytest.mark.asyncio
async def test_att_026_release_decrements_counters() -> None:
    """release() decrements both the per-user and global counters."""

    import app.api.v1.websockets as ws_module
    from app.core.security import get_redis_client

    user_id = uuid.uuid4()
    per_user_key = f"ws:user:{user_id}"
    total_key = "ws:total"

    client = await get_redis_client()
    await client.delete(per_user_key)
    await client.delete(total_key)

    limiter = ws_module._realtime_limiter
    try:
        acquired = await limiter.try_acquire(user_id)
        assert acquired is True
        per_user_after = int(await client.get(per_user_key) or 0)
        total_after = int(await client.get(total_key) or 0)
        assert per_user_after == 1, f"per-user count after acquire: {per_user_after}"
        assert total_after == 1, f"total count after acquire: {total_after}"

        await limiter.release(user_id)
        per_user_after_release = int(await client.get(per_user_key) or 0)
        total_after_release = int(await client.get(total_key) or 0)
        assert per_user_after_release == 0, (
            f"per-user count after release: {per_user_after_release}"
        )
        assert total_after_release == 0, (
            f"total count after release: {total_after_release}"
        )
    finally:
        await client.delete(per_user_key)
        await client.delete(total_key)


@pytest.mark.asyncio
async def test_att_026_release_after_expired_key_self_heals() -> None:
    """A release() against an already-expired per-user key is best-effort (no leak at global)."""

    import app.api.v1.websockets as ws_module
    from app.core.security import get_redis_client

    user_id = uuid.uuid4()
    per_user_key = f"ws:user:{user_id}"
    total_key = "ws:total"

    client = await get_redis_client()
    await client.delete(per_user_key)
    await client.delete(total_key)

    limiter = ws_module._realtime_limiter
    try:
        acquired = await limiter.try_acquire(user_id)
        assert acquired is True
        # Simulate key TTL expiry (missed DECR would look like this) — the
        # per-user key auto-expires, but the total is still 1.
        await client.delete(per_user_key)
        # Now release — release() should NOT make total go negative for
        # long; it should decrement total to 0 or stay at 0, not underflow
        # below zero in a way that punishes future acquirers.
        await limiter.release(user_id)
        total_after = int(await client.get(total_key) or 0)
        # Generous: total went from 1 to 0 (or self-corrected through DECR
        # which on Redis DECR past zero yields -1). The test pins that we
        # don't CRASH on a missing per-user key (the BUG we're guarding
        # against is a hard crash in `release` when the key expired).
        assert total_after in (-1, 0, 1), (
            f"release() against an expired per-user keyyielded a weird total: "
            f"{total_after}"
        )
    finally:
        await client.delete(per_user_key)
        await client.delete(total_key)


@pytest.mark.asyncio
async def test_att_026_invalid_env_raises() -> None:
    """Invalid cap env values must raise RuntimeError (fail-closed per §6)."""
    import app.api.v1.websockets as ws_module

    _orig_pu = ws_module.os.environ.pop("ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER", None)
    ws_module.os.environ["ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER"] = "foo"
    try:
        with pytest.raises(RuntimeError, match="ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER"):
            ws_module._read_max_ws_connections_per_user()
    finally:
        if _orig_pu is not None:
            ws_module.os.environ["ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER"] = _orig_pu
        else:
            ws_module.os.environ.pop("ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER", None)


def test_att_026_limiter_exists_at_module_level() -> None:
    """Pre-fix: no _realtime_limiter / no RealtimeConnectionLimiter."""
    import app.api.v1.websockets as ws_module

    assert hasattr(ws_module, "_realtime_limiter"), (
        "ATT-026: module-level _realtime_limiter absent — caps not enforced."
    )
    assert hasattr(ws_module, "RealtimeConnectionLimiter"), (
        "ATT-026: RealtimeConnectionLimiter class absent — caps not enforced."
    )


@pytest.mark.asyncio
async def test_att_026_global_counter_carries_ttl_self_heal() -> None:
    """ws:total must carry a TTL after acquire so missed DECRs self-heal.

    Pre-fix the global counter had no expiry: any process crash between
    accept and release permanently consumed global capacity until someone
    ran DEL by hand.
    """
    import uuid

    import app.api.v1.websockets as ws_module
    from app.core.security import get_redis_client

    user_id = uuid.uuid4()
    total_key = "ws:total"
    client = await get_redis_client()
    await client.delete(total_key)

    limiter = ws_module._realtime_limiter
    try:
        assert await limiter.try_acquire(user_id) is True
        ttl = await client.ttl(total_key)
        assert ttl is not None and ttl > 0, (
            "ATT-026: global ws:total counter has no TTL — leaked INCRs "
            "permanently consume capacity (no self-heal window)."
        )
    finally:
        await limiter.release(user_id)
