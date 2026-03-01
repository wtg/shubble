"""Main Bubble agent loop."""
import asyncio
import logging

from redis.asyncio import Redis

from backend.cache import init_cache, soft_clear_namespace
from backend.database import create_async_db_engine, create_session_factory
from bubble.agent import (
    generate_announcements,
    get_current_announcements,
    update_bubble_announcements,
)
from bubble.config import settings
from bubble.sources import fetch_all_sources

logger = logging.getLogger(__name__)


async def run_once(session_factory, redis_client: Redis) -> None:
    """Run one Bubble cycle: fetch sources → generate → persist → invalidate cache."""
    sources = settings.sources_list
    if not sources:
        logger.warning("BUBBLE_SOURCES is empty — nothing to fetch")

    logger.info("Fetching %d source(s)", len(sources))
    source_data = await fetch_all_sources(sources)
    logger.info("Successfully fetched %d/%d source(s)", len(source_data), len(sources))

    bubble_announcements, manual_announcements, suggestions = await get_current_announcements(
        session_factory, redis_client
    )
    logger.debug(
        "Current announcements: %d bubble-managed, %d manual; %d suggestion(s)",
        len(bubble_announcements),
        len(manual_announcements),
        len(suggestions),
    )

    updates = await generate_announcements(
        source_data,
        settings.GEMINI_API_KEY,
        settings.BUBBLE_MODEL,
        bubble_announcements,
        manual_announcements,
        suggestions,
    )
    await update_bubble_announcements(session_factory, redis_client, updates, bubble_announcements)

    # Invalidate the backend's announcement cache so riders see fresh data immediately
    await soft_clear_namespace("announcements")
    logger.debug("Invalidated announcements cache")


async def _ticker(interval_seconds: int):
    """Async generator that yields immediately, then at fixed intervals."""
    yield
    while True:
        await asyncio.sleep(interval_seconds)
        yield


async def run_bubble() -> None:
    """Entry point for the Bubble agent process."""
    log_level = settings.get_log_level().upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Starting Bubble agent (interval=%ds)", settings.BUBBLE_INTERVAL)

    engine = create_async_db_engine(settings.DATABASE_URL)
    session_factory = create_session_factory(engine)

    redis_client: Redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)

    # Initialize the shared backend cache module so soft_clear_namespace works
    await init_cache(settings.REDIS_URL)

    try:
        async for _ in _ticker(settings.BUBBLE_INTERVAL):
            try:
                await run_once(session_factory, redis_client)
            except Exception:
                logger.exception("Bubble cycle failed")
    finally:
        await redis_client.aclose()
        await engine.dispose()
        logger.info("Bubble agent shut down")
