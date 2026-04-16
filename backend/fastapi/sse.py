"""Server-Sent Events (SSE) endpoint and broadcaster for real-time updates.

This module implements the SSE streaming infrastructure:

    SSEBroadcaster: A singleton that maintains ONE Redis Pub/Sub subscription
    and fans out messages to all connected SSE clients. This avoids creating
    N Redis subscriptions for N browser tabs.

    sse_stream: The FastAPI endpoint at GET /api/stream. Each connected client
    gets a StreamingResponse that yields SSE-formatted events whenever the
    worker publishes new data.

SSE wire format (what the browser sees):
    event: locations
    data: {"vehicle_id_1": {...}, "vehicle_id_2": {...}}

    event: heartbeat
    data: {"time": "2026-02-24T12:00:00+00:00"}

Usage:
    # In backend/fastapi/__init__.py lifespan:
    from backend.fastapi.sse import broadcaster
    await broadcaster.start()   # on startup
    await broadcaster.stop()    # on shutdown

    # In backend/fastapi/routes.py:
    from backend.fastapi.sse import sse_stream
    router.add_api_route("/api/stream", sse_stream, methods=["GET"])
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import Request
from starlette.responses import StreamingResponse

from backend.pubsub import subscribe_updates
from backend.fastapi.utils import (
    build_locations_response,
    build_velocities_response,
    build_etas_response,
)

logger = logging.getLogger(__name__)


def format_sse(event: str, data: dict) -> str:
    """Format a dict as an SSE message string.

    SSE protocol requires:
        event: <event_name>\\n
        data: <json_payload>\\n
        \\n  (blank line terminates the message)

    Args:
        event: Event name (e.g., "locations", "heartbeat")
        data: Dict to serialize as JSON in the data field

    Returns:
        SSE-formatted string ready to yield from a StreamingResponse
    """
    json_str = json.dumps(data)
    return f"event: {event}\ndata: {json_str}\n\n"


class SSEBroadcaster:
    """Manages a single Redis Pub/Sub subscription shared across all SSE clients.

    Why a single subscription: If 50 users have the tracker open, we don't want
    50 separate Redis subscriptions. Instead, one background task subscribes to
    Redis and distributes messages to each client via asyncio.Queue instances.

    Lifecycle:
        broadcaster.start()  -> launches background listener task
        broadcaster.stop()   -> cancels the task (clean shutdown)

    Client lifecycle:
        async with broadcaster.subscribe() as queue:
            event_type = await queue.get()  # blocks until a message arrives
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[str]] = []
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background Pub/Sub listener task.

        Called once during FastAPI startup (in the lifespan context manager).
        """
        self._task = asyncio.create_task(self._listen())
        logger.info("SSE broadcaster started")

    async def stop(self) -> None:
        """Stop the background listener task.

        Called during FastAPI shutdown. Cancels the task and waits for it
        to finish cleaning up its Redis connection.
        """
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("SSE broadcaster stopped")

    async def _listen(self) -> None:
        """Background task: subscribe to Redis Pub/Sub and fan out messages.

        Runs continuously. If the Redis connection drops, subscribe_updates()
        handles reconnection internally (with exponential backoff).
        """
        try:
            async for event_type in subscribe_updates():
                async with self._lock:
                    dead_queues: list[asyncio.Queue[str]] = []
                    for queue in self._subscribers:
                        try:
                            queue.put_nowait(event_type)
                        except asyncio.QueueFull:
                            # Client is too slow to consume events — drop it
                            logger.warning("Dropping slow SSE client (queue full)")
                            dead_queues.append(queue)
                    for q in dead_queues:
                        self._subscribers.remove(q)
        except asyncio.CancelledError:
            logger.info("SSE broadcaster listener cancelled")
            raise
        except Exception as e:
            logger.error(f"SSE broadcaster listener error: {e}")

    @asynccontextmanager
    async def subscribe(self) -> AsyncGenerator[asyncio.Queue[str], None]:
        """Context manager for an SSE client to subscribe to events.

        Creates a Queue that receives event_type strings ("locations",
        "velocities", "etas") whenever the worker publishes an update.
        The queue is automatically removed when the client disconnects.

        Yields:
            asyncio.Queue that receives event type strings
        """
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=50)
        async with self._lock:
            self._subscribers.append(queue)
        logger.debug(f"SSE client subscribed (total: {len(self._subscribers)})")
        try:
            yield queue
        finally:
            async with self._lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)
            logger.debug(f"SSE client unsubscribed (total: {len(self._subscribers)})")


# Module-level singleton — imported by __init__.py and routes.py
broadcaster = SSEBroadcaster()


async def _fetch_data_for_event(event_type: str, session_factory) -> dict:
    """Fetch fresh data for a given event type.

    Reuses the same builder functions as the REST endpoints, which
    in turn use the Redis-cached query helpers. So if data is fresh
    in cache, this returns instantly without hitting PostgreSQL.

    Args:
        event_type: One of "locations", "velocities", "etas"
        session_factory: Async session factory from app.state

    Returns:
        Response data dict (same shape as the REST endpoint)
    """
    if event_type == "locations":
        response_data, _ = await build_locations_response(session_factory)
        return response_data
    elif event_type == "velocities":
        return await build_velocities_response(session_factory)
    elif event_type == "etas":
        return await build_etas_response(session_factory)
    else:
        logger.warning(f"Unknown SSE event type: {event_type}")
        return {}


async def sse_stream(request: Request) -> StreamingResponse:
    """SSE endpoint that streams real-time vehicle updates to the client.

    On connection:
        1. Sends an initial snapshot (locations, velocities, etas) so the
           client doesn't have to wait up to 5s for the first data.
        2. Subscribes to the broadcaster and waits for events.
        3. On each event, fetches fresh data and yields an SSE message.
        4. Sends a heartbeat every 15s to keep the connection alive.

    The response uses text/event-stream content type with buffering disabled
    (X-Accel-Buffering: no) for Nginx compatibility.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        session_factory = request.app.state.session_factory

        # 1. Send initial snapshot so client has data immediately
        try:
            locations_data, _ = await build_locations_response(session_factory)
            yield format_sse("locations", locations_data)

            velocities_data = await build_velocities_response(session_factory)
            yield format_sse("velocities", velocities_data)

            etas_data = await build_etas_response(session_factory)
            yield format_sse("etas", etas_data)
        except Exception as e:
            logger.error(f"SSE initial snapshot error: {e}")

        # 2. Enter event-driven loop
        async with broadcaster.subscribe() as queue:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.debug("SSE client disconnected")
                    break

                try:
                    # Wait for next event, with 15s timeout for heartbeat
                    event_type = await asyncio.wait_for(
                        queue.get(), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    # No events in 15s — send heartbeat to keep connection alive
                    yield format_sse(
                        "heartbeat",
                        {"time": datetime.now(timezone.utc).isoformat()},
                    )
                    continue

                # Drain any additional events that arrived in the same batch
                # (e.g., if "locations" and "velocities" arrive within ms)
                event_types = {event_type}
                try:
                    while True:
                        event_types.add(queue.get_nowait())
                except asyncio.QueueEmpty:
                    pass

                # Fetch and send each unique event type
                for et in event_types:
                    try:
                        data = await _fetch_data_for_event(et, session_factory)
                        yield format_sse(et, data)
                    except Exception as e:
                        logger.error(f"SSE data fetch error for {et}: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx response buffering
        },
    )
