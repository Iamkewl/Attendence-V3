"""Redis-backed pub/sub primitives and one-time ticket helpers for realtime channels."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis
from redis.asyncio.client import PubSub

from app.core.security import get_redis_client


LIVE_ATTENDANCE_CHANNEL = "live_attendance"
SYSTEM_ALERTS_CHANNEL = "system_alerts"
DEFAULT_REALTIME_CHANNELS: tuple[str, ...] = (
    LIVE_ATTENDANCE_CHANNEL,
    SYSTEM_ALERTS_CHANNEL,
)

WS_TICKET_KEY_PREFIX = "auth:ws_ticket"
WS_TICKET_TTL_SECONDS = 30


_CONSUME_TICKET_SCRIPT = """
local value = redis.call("GET", KEYS[1])
if not value then
    return nil
end
redis.call("DEL", KEYS[1])
return value
""".strip()


@dataclass(frozen=True, slots=True)
class PubSubMessage:
    """Normalized pub/sub payload emitted by Redis channel subscriptions."""

    channel: str
    payload: str


def websocket_ticket_key(ticket: str) -> str:
    """Build the Redis key used to store one-time websocket/SSE session tickets."""
    normalized_ticket = ticket.strip()
    if not normalized_ticket:
        raise ValueError("Ticket value must not be blank.")
    return f"{WS_TICKET_KEY_PREFIX}:{normalized_ticket}"


class RedisPubSubManager:
    """High-level async Redis pub/sub manager used by realtime transport endpoints."""

    def __init__(self, redis_client: Redis[str] | None = None) -> None:
        self._redis_client = redis_client

    async def publish(self, channel: str, message: str) -> int:
        """Publish a UTF-8 string message to a Redis pub/sub channel."""
        normalized_channel = channel.strip()
        if not normalized_channel:
            raise ValueError("Channel name must not be blank.")

        client = await self._get_client()
        return int(await client.publish(normalized_channel, message))

    async def publish_json(self, channel: str, payload: Mapping[str, Any]) -> int:
        """Serialize a mapping payload to JSON and publish it to a channel."""
        serialized_payload = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        return await self.publish(channel, serialized_payload)

    @asynccontextmanager
    async def subscribe(self, channels: Sequence[str]) -> AsyncIterator[PubSub]:
        """Create a managed pub/sub subscription that always unsubscribes and closes cleanly."""
        normalized_channels = _normalize_channels(channels)

        client = await self._get_client()
        pubsub = client.pubsub()
        await pubsub.subscribe(*normalized_channels)
        try:
            yield pubsub
        finally:
            try:
                await pubsub.unsubscribe(*normalized_channels)
            finally:
                await pubsub.aclose()

    async def iter_messages(
        self,
        channels: Sequence[str],
        *,
        stop_event: asyncio.Event | None = None,
        poll_timeout_seconds: float = 1.0,
    ) -> AsyncIterator[PubSubMessage]:
        """Iterate published channel messages until explicitly stopped."""
        if poll_timeout_seconds <= 0:
            raise ValueError("poll_timeout_seconds must be greater than zero.")

        async with self.subscribe(channels) as pubsub:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=poll_timeout_seconds,
                )
                if message is None:
                    await asyncio.sleep(0)
                    continue

                parsed_message = _parse_pubsub_message(message)
                if parsed_message is not None:
                    yield parsed_message

    async def issue_ticket(
        self,
        ticket: str,
        *,
        payload: str,
        ttl_seconds: int = WS_TICKET_TTL_SECONDS,
    ) -> bool:
        """Store a one-time realtime ticket in Redis with strict expiration semantics."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero.")

        client = await self._get_client()
        result = await client.set(
            websocket_ticket_key(ticket),
            payload,
            ex=ttl_seconds,
            nx=True,
        )
        return bool(result)

    async def consume_ticket(self, ticket: str) -> str | None:
        """Atomically read and delete a one-time realtime ticket value from Redis."""
        client = await self._get_client()
        consumed = await client.eval(_CONSUME_TICKET_SCRIPT, 1, websocket_ticket_key(ticket))
        if consumed is None:
            return None

        if isinstance(consumed, bytes):
            return consumed.decode("utf-8")

        return str(consumed)

    async def _get_client(self) -> Redis[str]:
        """Return a cached Redis client instance, initializing lazily when required."""
        if self._redis_client is None:
            self._redis_client = await get_redis_client()
        return self._redis_client


def _normalize_channels(channels: Sequence[str]) -> tuple[str, ...]:
    """Validate and normalize channel names before Redis subscription calls."""
    normalized_channels: list[str] = []
    for channel in channels:
        normalized = channel.strip()
        if not normalized:
            raise ValueError("Channel name must not be blank.")
        normalized_channels.append(normalized)

    if not normalized_channels:
        raise ValueError("At least one channel is required for pub/sub subscription.")

    return tuple(normalized_channels)


def _parse_pubsub_message(message: dict[str, Any]) -> PubSubMessage | None:
    """Normalize raw redis-py pub/sub envelopes into string-based message records."""
    message_type = message.get("type")
    if message_type not in {"message", "pmessage"}:
        return None

    raw_channel = message.get("channel")
    raw_payload = message.get("data")

    if isinstance(raw_channel, bytes):
        channel = raw_channel.decode("utf-8")
    elif isinstance(raw_channel, str):
        channel = raw_channel
    else:
        channel = str(raw_channel)

    if isinstance(raw_payload, bytes):
        payload = raw_payload.decode("utf-8")
    elif isinstance(raw_payload, str):
        payload = raw_payload
    else:
        payload = json.dumps(raw_payload, separators=(",", ":"), ensure_ascii=True)

    return PubSubMessage(channel=channel, payload=payload)


__all__ = [
    "DEFAULT_REALTIME_CHANNELS",
    "LIVE_ATTENDANCE_CHANNEL",
    "PubSubMessage",
    "RedisPubSubManager",
    "SYSTEM_ALERTS_CHANNEL",
    "WS_TICKET_KEY_PREFIX",
    "WS_TICKET_TTL_SECONDS",
    "websocket_ticket_key",
]
