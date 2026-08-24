"""Authenticated realtime transport endpoints backed by Redis Pub/Sub channels."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse

from app.core.pubsub import DEFAULT_REALTIME_CHANNELS, PubSubMessage, RedisPubSubManager
from app.core.security import get_redis_client


LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["Realtime"])


# ---------------------------------------------------------------------------
# ATT-026 — connection caps.
#
# The unbounded _connections set + no per-user cap let a single client (or a
# hostile session) open hundreds/thousands of WebSocket subscriptions, each
# pinned to the shared broadcast set. Combined with synchronous HoL blocking
# (ATT-024) this lets one client starve every other dashboard subscriber.
#
# Caps are configurable via env so operators can raise/lower per-deployment.
# Redis-backed INCR/DECR handles multi-process uvicorn deployments correctly;
# the per-user key has a TTL of the access-token-TTL so a missed DECR
# (client crash mid-test, etc.) self-heals within the access window.
#
# Two caps:
#   * ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER  (default 5)   — per user, across WS + SSE
#   * ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL      (default 100) — global, all ws endpoints
#
# On cap exceeded, the WS handler returns code 1011 ("try again later")
# after a brief reason string the client can read. Prior connections are
# not affected — they were already accepted and counted.
# ---------------------------------------------------------------------------


_WS_PER_USER_KEY = "ws:user:{user_id}"
_WS_TOTAL_KEY = "ws:total"


def _read_max_ws_connections_per_user() -> int:
    """Read the per-user connection cap from env; fail-closed on invalid input."""
    raw = os.getenv("ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER", "5")
    try:
        v = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER must be a "
            f"positive integer (got {raw!r})."
        ) from exc
    if v <= 0:
        raise RuntimeError(
            f"Environment variable ATTENDANCE_MAX_WS_CONNECTIONS_PER_USER must be a "
            f"positive integer (got {v})."
        )
    return v


def _read_max_ws_connections_total() -> int:
    """Read the global connection cap from env; fail-closed on invalid input."""
    raw = os.getenv("ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL", "100")
    try:
        v = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL must be a "
            f"positive integer (got {raw!r})."
        ) from exc
    if v <= 0:
        raise RuntimeError(
            f"Environment variable ATTENDANCE_MAX_WS_CONNECTIONS_TOTAL must be a "
            f"positive integer (got {v})."
        )
    return v


def _access_token_ttl_seconds() -> int:
    """Read ATTENDANCE_ACCESS_TOKEN_TTL_MINUTES for the per-user counter key TTL.

    A missed DECR (client crash, server OOM kill, GC pause past close) would
    leave the counter stuck at +1 forever, exhausting a user's allotment. A
    TTL keyed to the access-token's natural TTL bounds the leak to the access
    window so a missed DECR self-heals within minutes — the same treatment
    `blocklist_token` gives JWT JTIs.
    """
    raw = os.getenv("ATTENDANCE_ACCESS_TOKEN_TTL_MINUTES", "15")
    try:
        minutes = int(raw)
        return max(minutes * 60, 60)
    except (TypeError, ValueError):
        return 15 * 60


class RealtimeConnectionLimiter:
    """Redis-backed per-user + global caps for concurrent realtime connections.

    Redis counters are used (rather than a Python int) because a production
    uvicorn deployment may run multiple worker processes, and only Redis
    yields a consistent cross-process count.

    All operations are best-effort: if Redis is unavailable, the limiter
    surfaces the error to the caller rather than silently bypassing the cap
    (fail-closed per §6: a security control must not silently relax).
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def try_acquire(self, user_id: UUID) -> bool:
        """Atomically increment per-user and global counters if both have room.

        Returns True when the connection is acquired; False when a cap is hit
        (the counters are rolled back so a brief transient over-count does
        not pin the next caller).
        """
        per_user_cap = _read_max_ws_connections_per_user()
        total_cap = _read_max_ws_connections_total()
        per_user_key = _WS_PER_USER_KEY.format(user_id=user_id)
        ttl_seconds = _access_token_ttl_seconds()

        client = await get_redis_client()
        async with self._lock:
            existing_user = await client.incr(per_user_key)
            if existing_user > per_user_cap:
                # Over the cap — back the INCR out and refuse.
                await client.decr(per_user_key)
                return False
            # Refresh the TTL on every successful INCR — handles a missed
            # DECR by self-healing within the access-token window.
            await client.expire(per_user_key, ttl_seconds)
            existing_total = await client.incr(_WS_TOTAL_KEY)
            if existing_total > total_cap:
                await client.decr(_WS_TOTAL_KEY)
                await client.decr(per_user_key)
                return False
            # Same TTL safety net as the per-user key: without it a missed
            # DECR (process crash between accept and release) permanently
            # consumes global capacity and only a manual DEL restores it.
            # Refreshed on every successful acquire, mirroring per-user.
            await client.expire(_WS_TOTAL_KEY, ttl_seconds)
            return True

    async def release(self, user_id: UUID) -> None:
        """Decrement a previously-acquired per-user + global reservation.

        Best-effort: a DECR idempotency check is not done (Redis DECR is
        silent past 0); if a stale DECR slips in after the key has expired,
        the next INCR on a brand-new key just re-seeds at 1 then 1 = baseline.
        The TTL safety net ensures a missed DECR eventually resets the cap
        state to baseline.
        """
        per_user_key = _WS_PER_USER_KEY.format(user_id=user_id)
        client = await get_redis_client()
        async with self._lock:
            # Best-effort DECR; if the key has already expired (missed DECR),
            # this is a self-correcting surplus.
            exists = await client.exists(per_user_key)
            if exists:
                await client.decr(per_user_key)
            await client.decr(_WS_TOTAL_KEY)


_realtime_limiter = RealtimeConnectionLimiter()


class LiveConnectionManager:
    """Manage active websocket clients and one shared Redis broadcast worker."""

    def __init__(self, pubsub_manager: RedisPubSubManager, channels: Sequence[str]) -> None:
        self._pubsub_manager = pubsub_manager
        self._channels = tuple(channels)
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._broadcast_task: asyncio.Task[None] | None = None

    async def connect(self, websocket: WebSocket) -> None:
        """Accept one websocket and ensure the Redis broadcast worker is running.

        ATT-025: the broadcast task is lazy-started on first connect and NEVER
        stopped when the last client disconnects. Stopping the task on last
        disconnect raised a race — a concurrent connect could start a new
        task on a fresh `iter_messages` subscription while the old task was
        still tearing down, double-dispatching a final pubsub message. With
        one persistent per-process task we sidestep the race entirely.
        The trade-off is one idle task per process awaiting
        `pubsub.get_message(timeout=1.0)` when nobody is connected; that is
        a negligible cost (one await per second per process) for guaranteed
        duplicate-free delivery.
        """
        await websocket.accept()

        async with self._lock:
            self._connections.add(websocket)

            if self._broadcast_task is None or self._broadcast_task.done():
                self._broadcast_task = asyncio.create_task(
                    self._run_broadcast_loop(),
                    name="realtime-redis-broadcast-loop",
                )

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove one websocket from the broadcast set.

        ATT-025: notably, this method does NOT stop or null the broadcast
        task. The task runs for the life of the process. The reasoning and
        race description are in `connect`.
        """
        async with self._lock:
            self._connections.discard(websocket)

    async def _run_broadcast_loop(self) -> None:
        """Forward Redis pub/sub events to all connected websocket clients.

        The loop runs forever (per ATT-025). It only stops if the task is
        cancelled (process shutdown). When no clients are connected the
        broadcast still resolves to a no-op (`_broadcast` returns early),
        so the cost is just one `pubsub.get_message(timeout=1.0)` poll per
        second — negligible.
        """
        while True:
            try:
                async for message in self._pubsub_manager.iter_messages(
                    self._channels,
                ):
                    await self._broadcast(message)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Realtime broadcast loop failed; retrying subscription.")
                await asyncio.sleep(1)

    async def _broadcast(self, message: PubSubMessage) -> None:
        """Push one pub/sub message to every active websocket connection (concurrent).

        ATT-024: previously broadcast was a sequential per-client
        `await websocket.send_text(...)` loop. If client #1's TCP buffer
        filled (slow consumer) the loop stalled until the keepalive ping
        timed out — every subsequent client and every queued pubsub message
        was held back. With 100 dashboards, one stalled socket blocked all.

        Now we `asyncio.gather` every client send concurrently with
        `return_exceptions=True`, so a slow client only delays itself and is
        dropped from the set on the next broadcast (the disconnect branch
        removes dead sockets). Total broadcast latency is bounded by
        max(client send time) instead of sum(client send times).
        """
        payload_text = json.dumps(
            {
                "channel": message.channel,
                "payload": _decode_payload(message.payload),
            },
            separators=(",", ":"),
            ensure_ascii=True,
        )

        async with self._lock:
            recipients = tuple(self._connections)

        if not recipients:
            return

        async def _send_one(websocket: WebSocket) -> WebSocket | None:
            try:
                await websocket.send_text(payload_text)
                return None
            except (RuntimeError, WebSocketDisconnect):
                return websocket
            except Exception:
                LOGGER.exception("Failed to send realtime message to websocket client.")
                return websocket

        # Concurrent broadcast — see the ATT-024 reasoning above.
        results = await asyncio.gather(
            *(_send_one(ws) for ws in recipients),
            return_exceptions=True,
        )

        # Drop any sockets whose send raised; gather already gave them their
        # own per-client try, so the only thing left is to discard from the
        # shared set under the lock.
        disconnected: list[WebSocket] = []
        for ws, result in zip(recipients, results, strict=True):
            if isinstance(result, WebSocket) and result is ws:
                disconnected.append(ws)
            elif isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                # An unexpected exception (not raised through the inner
                # try) — treat the socket as suspect, drop it.
                disconnected.append(ws)

        if disconnected:
            async with self._lock:
                for websocket in disconnected:
                    self._connections.discard(websocket)


_pubsub_manager = RedisPubSubManager()
_connection_manager = LiveConnectionManager(_pubsub_manager, DEFAULT_REALTIME_CHANNELS)


def _parse_ticket_user_id(ticket_payload: str) -> UUID | None:
    """Extract the user_id from the consumed ws-ticket JSON payload.

    The ticket is issued by /auth/ws-ticket as
    ``json.dumps({"user_id": str(current_user.id), "issued_at": ...})``.
    Failing to parse or missing fields returns None — the caller treats
    that as an unauthenticated ticket (refuses the connection).
    """
    try:
        payload = json.loads(ticket_payload)
    except json.JSONDecodeError:
        return None
    user_id = payload.get("user_id") if isinstance(payload, dict) else None
    if not isinstance(user_id, str):
        return None
    try:
        return UUID(user_id)
    except (ValueError, AttributeError):
        return None


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    """Accept an authenticated live websocket stream after one-time ticket validation."""
    ticket = websocket.query_params.get("ticket")
    if ticket is None or not ticket.strip():
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Missing websocket ticket.",
        )
        return

    consumed_ticket = await _pubsub_manager.consume_ticket(ticket)
    if consumed_ticket is None:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Websocket ticket is invalid or expired.",
        )
        return

    user_id = _parse_ticket_user_id(consumed_ticket)
    if user_id is None:
        # The ticket payload didn't carry a usable user_id — this means the
        # ticket was issued by a buggy /auth/ws-ticket path or tampered with.
        # Either way, treat as unauthenticated.
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Websocket ticket is malformed.",
        )
        return

    # ATT-026: acquire a per-user and global connection-count reservation
    # before accepting the WebSocket. If either cap is hit, refuse with 1011
    # so the client retries later; prior connections are not affected.
    try:
        acquired = await _realtime_limiter.try_acquire(user_id)
    except Exception:
        LOGGER.exception("Realtime connection limiter unavailable; refusing new connection (fail-closed).")
        await websocket.close(
            code=status.WS_1011_INTERNAL_ERROR,
            reason="Realtime limiter unavailable; try again later.",
        )
        return

    if not acquired:
        await websocket.close(
            code=status.WS_1011_INTERNAL_ERROR,
            reason="Too many concurrent realtime connections; try again later.",
        )
        return

    await _connection_manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        await _connection_manager.disconnect(websocket)
        # Release the per-user and global reservation. Best-effort — a missed
        # release self-heals via the per-user key TTL.
        try:
            await _realtime_limiter.release(user_id)
        except Exception:
            LOGGER.warning("Realtime connection limiter release failed for user %s.", user_id)


@router.get(
    "/sse/live",
    summary="Live Event Stream",
    description="Authenticated server-sent event stream for realtime dashboard updates.",
)
async def sse_live(
    request: Request,
    ticket: str = Query(min_length=1),
) -> StreamingResponse:
    """Stream Redis channel events through SSE after one-time ticket validation."""
    consumed_ticket = await _pubsub_manager.consume_ticket(ticket)
    if consumed_ticket is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Realtime ticket is invalid or expired.",
        )

    user_id = _parse_ticket_user_id(consumed_ticket)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Realtime ticket is malformed.",
        )

    # ATT-026: SSE shares the same per-user / global connection cap so a
    # single user cannot exhaust the realtime count by opening SSE instead
    # of WebSocket. The global cap counts against the same Redis key.
    try:
        acquired = await _realtime_limiter.try_acquire(user_id)
    except Exception:
        LOGGER.exception("Realtime connection limiter unavailable; refusing SSE (fail-closed).")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Realtime limiter unavailable; try again later.",
        ) from None

    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many concurrent realtime connections; try again later.",
        )

    async def event_stream() -> AsyncIterator[str]:
        stop_event = asyncio.Event()

        async def watch_disconnect() -> None:
            while not stop_event.is_set():
                if await request.is_disconnected():
                    stop_event.set()
                    break
                await asyncio.sleep(0.5)

        disconnect_task = asyncio.create_task(watch_disconnect(), name="sse-disconnect-watch")
        try:
            yield "event: ready\ndata: {\"status\":\"connected\"}\n\n"
            async for message in _pubsub_manager.iter_messages(
                DEFAULT_REALTIME_CHANNELS,
                stop_event=stop_event,
            ):
                payload = json.dumps(
                    {
                        "channel": message.channel,
                        "payload": _decode_payload(message.payload),
                    },
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                yield f"event: {message.channel}\ndata: {payload}\n\n"
        finally:
            stop_event.set()
            disconnect_task.cancel()
            try:
                await disconnect_task
            except asyncio.CancelledError:
                pass
            # ATT-026: release the SSE-side reservation in the finally block
            # so a disconnect (TCP-close or server shutdown) restores the
            # count. Best-effort; a missed release self-heals via the
            # per-user key TTL.
            try:
                await _realtime_limiter.release(user_id)
            except Exception:
                LOGGER.warning("Realtime SSE limiter release failed for user %s.", user_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _decode_payload(payload: str) -> Any:
    """Decode JSON payloads when possible, falling back to raw text for opaque events."""
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload


__all__ = ["router"]
