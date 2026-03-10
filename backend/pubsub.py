"""Redis Pub/Sub helpers for real-time SSE updates.

This module provides publish/subscribe functionality using Redis Pub/Sub,
which is separate from the key-value caching in cache.py.

How it works:
    - The worker calls publish_update("locations") after inserting new GPS data.
    - Redis delivers that message to all active subscribers in real-time.
    - The FastAPI SSE broadcaster subscribes and fans messages out to connected clients.
    - Messages are fire-and-forget: if nobody is listening, they're discarded.

The publisher reuses the existing global Redis client from cache.py (publishing
is a normal command). The subscriber creates a dedicated connection because
Redis requires that a subscribed connection cannot be used for other commands.

Usage:
    # In the worker (after inserting data):
    from backend.pubsub import publish_update
    await publish_update("locations")

    # In the FastAPI SSE handler:
    from backend.pubsub import subscribe_updates
    async for event_type in subscribe_updates():
        # event_type is "locations", "velocities", or "etas"
        ...
"""
import asyncio
import logging
from typing import AsyncGenerator

from redis import asyncio as aioredis

from backend.cache import get_redis
from backend.config import settings

logger = logging.getLogger(__name__)

# Channel name for all real-time update notifications
CHANNEL = "shubble:updates"

# Event type constants
EVENT_LOCATIONS = "locations"
EVENT_VELOCITIES = "velocities"
EVENT_ETAS = "etas"


async def publish_update(event_type: str) -> None:
    """Publish an update notification to the Pub/Sub channel.

    Called by the worker after new data is committed to the database.
    Uses the existing global Redis client from cache.py.

    Args:
        event_type: One of "locations", "velocities", "etas"
    """
    redis = get_redis()
    if redis is None:
        logger.warning("Redis not initialized, cannot publish update")
        return

    try:
        await redis.publish(CHANNEL, event_type)
        logger.debug(f"Published update: {event_type}")
    except Exception as e:
        # Publishing failures should not crash the worker
        logger.error(f"Failed to publish update '{event_type}': {e}")


async def subscribe_updates() -> AsyncGenerator[str, None]:
    """Subscribe to the update notifications channel.

    Creates a dedicated Redis connection for the subscription (required
    because a subscribed connection cannot be used for other commands).

    Yields event_type strings ("locations", "velocities", "etas") as
    they are published by the worker.

    This generator handles reconnection internally. If the connection
    drops, it logs the error, waits with exponential backoff, and
    reconnects. The caller sees a continuous stream of events.

    Yields:
        Event type string for each published update.
    """
    backoff = 1  # seconds, doubles on each failure up to 30s

    while True:
        redis_sub = None
        pubsub = None
        try:
            # Create a dedicated connection for subscribing
            redis_sub = await aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
            )
            pubsub = redis_sub.pubsub()
            await pubsub.subscribe(CHANNEL)
            logger.info(f"Subscribed to Redis Pub/Sub channel: {CHANNEL}")

            # Reset backoff on successful connection
            backoff = 1

            # Listen for messages indefinitely
            async for message in pubsub.listen():
                # Redis Pub/Sub messages have type "subscribe" (confirmation),
                # "message" (actual data), etc. We only care about "message".
                if message["type"] == "message":
                    yield message["data"]

        except asyncio.CancelledError:
            # Clean shutdown requested (e.g., app is stopping)
            logger.info("Pub/Sub subscription cancelled")
            raise
        except Exception as e:
            logger.error(f"Pub/Sub connection error: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
        finally:
            # Always clean up the connection
            if pubsub:
                try:
                    await pubsub.unsubscribe(CHANNEL)
                    await pubsub.close()
                except Exception:
                    pass
            if redis_sub:
                try:
                    await redis_sub.close()
                except Exception:
                    pass
