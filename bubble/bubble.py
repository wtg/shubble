"""Main Bubble agent loop."""
import asyncio
import logging
import time

from redis.asyncio import Redis

from backend.cache import init_cache, soft_clear_namespace
from backend.database import create_async_db_engine, create_session_factory
from bubble.agent import (
    generate_announcements,
    get_current_announcements,
    update_bubble_announcements,
)
from bubble.config import settings
from bubble.memory import ensure_table, load_exchanges, save_exchange
from bubble.sources import fetch_all_sources, fetch_live_locations

logger = logging.getLogger(__name__)

_BUBBLE_VOTES_CHANNEL = "bubble:votes"


async def run_once(session_factory, redis_client: Redis) -> None:
    """Run one Bubble cycle: fetch sources → generate → persist → invalidate cache."""
    sources = settings.sources_list
    if not sources:
        logger.warning("BUBBLE_SOURCES is empty — nothing to fetch")

    logger.info("Fetching %d source(s) + live locations", len(sources))
    source_data, live_location_summary, past_exchanges = await asyncio.gather(
        fetch_all_sources(sources),
        fetch_live_locations(settings.BACKEND_URL),
        load_exchanges(settings.DATABASE_URL, settings.BUBBLE_MEMORY_EXCHANGES),
    )
    logger.info("Successfully fetched %d/%d source(s)", len(source_data), len(sources))
    logger.debug("Live location summary: %s", live_location_summary)
    logger.debug("Loaded %d past exchange(s) from memory", len(past_exchanges))

    bubble_announcements, manual_announcements, suggestions = await get_current_announcements(
        session_factory, redis_client
    )
    logger.debug(
        "Current announcements: %d bubble-managed, %d manual; %d suggestion(s)",
        len(bubble_announcements),
        len(manual_announcements),
        len(suggestions),
    )

    updates, user_content, response_text = await generate_announcements(
        source_data,
        settings.GEMINI_API_KEY,
        settings.BUBBLE_MODEL,
        bubble_announcements,
        manual_announcements,
        suggestions,
        live_location_summary,
        past_exchanges,
    )
    await update_bubble_announcements(session_factory, redis_client, updates, bubble_announcements)

    # Invalidate the backend's announcement cache so riders see fresh data immediately
    await soft_clear_namespace("announcements")
    logger.debug("Invalidated announcements cache")

    # Persist this run to memory (skip if generation failed)
    if response_text:
        try:
            await save_exchange(settings.DATABASE_URL, user_content, response_text)
        except Exception:
            logger.warning("Failed to save exchange to memory", exc_info=True)


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

    logger.info(
        "Starting Bubble agent (interval=%ds, vote_threshold=%d, vote_cooldown=%ds, memory=%d exchanges)",
        settings.BUBBLE_INTERVAL,
        settings.BUBBLE_VOTE_THRESHOLD,
        settings.BUBBLE_VOTE_COOLDOWN,
        settings.BUBBLE_MEMORY_EXCHANGES,
    )

    engine = create_async_db_engine(settings.DATABASE_URL)
    session_factory = create_session_factory(engine)

    redis_client: Redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)

    # Initialize the shared backend cache module so soft_clear_namespace works
    await init_cache(settings.REDIS_URL)

    # Ensure the LangChain memory table exists before any run
    await ensure_table(settings.DATABASE_URL)

    # Shared last-run timestamp (monotonic); 0.0 means never run.
    # Stored in a list so both inner coroutines can mutate it.
    last_run_at: list[float] = [0.0]

    # Redis lock prevents concurrent runs from the ticker and vote listener.
    # timeout is a safety TTL that auto-releases the lock if Bubble crashes mid-run.
    run_lock = redis_client.lock("bubble:run_lock", timeout=60)

    async def _run_once_locked(trigger: str) -> None:
        """Acquire the run lock (non-blocking) and execute one Bubble cycle."""
        acquired = await run_lock.acquire(blocking=False)
        if not acquired:
            logger.info("%s: another Bubble run is already in progress — skipping", trigger)
            return
        try:
            await run_once(session_factory, redis_client)
            last_run_at[0] = time.monotonic()
        finally:
            await run_lock.release()

    async def _ticker_loop() -> None:
        async for _ in _ticker(settings.BUBBLE_INTERVAL):
            try:
                await _run_once_locked("ticker")
            except Exception:
                logger.exception("Bubble cycle failed")

    async def _vote_listener_loop() -> None:
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(_BUBBLE_VOTES_CHANNEL)
        logger.info("Subscribed to Redis channel '%s'", _BUBBLE_VOTES_CHANNEL)

        vote_count = 0
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue

                vote_count += 1
                logger.debug(
                    "Vote received (count=%d/%d)",
                    vote_count,
                    settings.BUBBLE_VOTE_THRESHOLD,
                )

                if vote_count < settings.BUBBLE_VOTE_THRESHOLD:
                    continue

                elapsed = time.monotonic() - last_run_at[0]
                if elapsed < settings.BUBBLE_VOTE_COOLDOWN:
                    logger.info(
                        "Vote threshold reached but cooldown active (%.0fs remaining) — keeping count",
                        settings.BUBBLE_VOTE_COOLDOWN - elapsed,
                    )
                    continue

                logger.info("Vote threshold reached — triggering Bubble rerun")
                try:
                    await _run_once_locked("vote_listener")
                except Exception:
                    logger.exception("Vote-triggered Bubble cycle failed")
                finally:
                    # Reset only after attempting a run, not during cooldown
                    vote_count = 0
        finally:
            await pubsub.unsubscribe(_BUBBLE_VOTES_CHANNEL)
            await pubsub.aclose()

    try:
        await asyncio.gather(_ticker_loop(), _vote_listener_loop())
    finally:
        await redis_client.aclose()
        await engine.dispose()
        logger.info("Bubble agent shut down")
