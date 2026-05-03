"""Authenticated realtime transport endpoints backed by Redis Pub/Sub channels."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse

from app.core.pubsub import DEFAULT_REALTIME_CHANNELS, PubSubMessage, RedisPubSubManager


LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["Realtime"])


class LiveConnectionManager:
    """Manage active websocket clients and one shared Redis broadcast worker."""

    def __init__(self, pubsub_manager: RedisPubSubManager, channels: Sequence[str]) -> None:
        self._pubsub_manager = pubsub_manager
        self._channels = tuple(channels)
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._broadcast_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    async def connect(self, websocket: WebSocket) -> None:
        """Accept one websocket and ensure Redis broadcast worker is running."""
        await websocket.accept()

        async with self._lock:
            self._connections.add(websocket)

            if self._broadcast_task is None or self._broadcast_task.done():
                self._stop_event = asyncio.Event()
                self._broadcast_task = asyncio.create_task(
                    self._run_broadcast_loop(self._stop_event),
                    name="realtime-redis-broadcast-loop",
                )

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove one websocket and stop worker when no clients remain."""
        task_to_wait: asyncio.Task[None] | None = None

        async with self._lock:
            self._connections.discard(websocket)

            if (
                not self._connections
                and self._broadcast_task is not None
                and not self._broadcast_task.done()
                and self._stop_event is not None
            ):
                self._stop_event.set()
                task_to_wait = self._broadcast_task
                self._broadcast_task = None
                self._stop_event = None

        if task_to_wait is not None:
            try:
                await task_to_wait
            except asyncio.CancelledError:
                pass

    async def _run_broadcast_loop(self, stop_event: asyncio.Event) -> None:
        """Forward Redis pub/sub events to all connected websocket clients."""
        while not stop_event.is_set():
            try:
                async for message in self._pubsub_manager.iter_messages(
                    self._channels,
                    stop_event=stop_event,
                ):
                    await self._broadcast(message)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Realtime broadcast loop failed; retrying subscription.")
                if stop_event.is_set():
                    break
                await asyncio.sleep(1)

    async def _broadcast(self, message: PubSubMessage) -> None:
        """Push one pub/sub message to every active websocket connection."""
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

        disconnected: list[WebSocket] = []
        for websocket in recipients:
            try:
                await websocket.send_text(payload_text)
            except (RuntimeError, WebSocketDisconnect):
                disconnected.append(websocket)
            except Exception:
                LOGGER.exception("Failed to send realtime message to websocket client.")
                disconnected.append(websocket)

        if disconnected:
            async with self._lock:
                for websocket in disconnected:
                    self._connections.discard(websocket)


_pubsub_manager = RedisPubSubManager()
_connection_manager = LiveConnectionManager(_pubsub_manager, DEFAULT_REALTIME_CHANNELS)


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

    await _connection_manager.connect(websocket)
    try:
        while True:
            await websocket.receive()
    except WebSocketDisconnect:
        pass
    finally:
        await _connection_manager.disconnect(websocket)


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
                yield f"event: {message.channel}\\ndata: {payload}\\n\\n"
        finally:
            stop_event.set()
            disconnect_task.cancel()
            try:
                await disconnect_task
            except asyncio.CancelledError:
                pass

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
