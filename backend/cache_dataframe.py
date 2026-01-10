"""Data loading utilities."""
import logging
import pickle
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import select
from redis import asyncio as aioredis
from fastapi_cache import FastAPICache

from backend.config import settings
from backend.database import create_async_db_engine, create_session_factory
from backend.models import VehicleLocation
from backend.time_utils import get_campus_start_of_day
from ml.pipelines import preprocess_pipeline, segment_pipeline, stops_pipeline

logger = logging.getLogger(__name__)


def get_cache_and_timestamp_key(date_str: str) -> tuple[str, str]:
    """Get Redis cache keys for data and timestamp."""
    return f"locations:{date_str}", f"locations:{date_str}:last_updated"


def process_raw_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw vehicle location data through the ML pipeline.

    Raw -> Preprocess -> Segment -> Stops
    """
    if raw_df.empty:
        return raw_df

    # Run pipelines in sequence, injecting the dataframe to bypass disk cache loading
    # We pass flag=True to ensure any internal checks know we want to compute
    # We pass cache=False to disable disk caching (Redis cache is sufficient)
    df = preprocess_pipeline(df=raw_df, preprocess=True, cache=False)
    df = segment_pipeline(df=df, segment=True, cache=False, min_segment_length=1)
    df = stops_pipeline(df=df, stops=True, cache=False)

    return df


async def load_today_dataframe(since: datetime | None = None) -> pd.DataFrame:
    """
    Fetch today's vehicle location data from the database.

    Args:
        since: Optional timestamp to fetch only records after this time.
               If None, fetches all records from start of day.

    Returns:
        DataFrame with columns: vehicle_id, latitude, longitude, timestamp
    """
    # Create database engine and session factory
    engine = create_async_db_engine(settings.DATABASE_URL, echo=False)
    session_factory = create_session_factory(engine)

    start_of_day = get_campus_start_of_day()

    async with session_factory() as session:
        # Query vehicle locations from today
        stmt = select(
            VehicleLocation.vehicle_id,
            VehicleLocation.latitude,
            VehicleLocation.longitude,
            VehicleLocation.timestamp
        )

        # Apply time filter based on whether 'since' is provided
        if since is not None:
            stmt = stmt.where(VehicleLocation.timestamp > since)
        else:
            stmt = stmt.where(VehicleLocation.timestamp >= start_of_day)

        stmt = stmt.order_by(VehicleLocation.timestamp)

        result = await session.execute(stmt)
        rows = result.fetchall()

    # Close engine
    await engine.dispose()

    # Convert to DataFrame
    df = pd.DataFrame(
        rows,
        columns=['vehicle_id', 'latitude', 'longitude', 'timestamp']
    )

    return df


async def get_today_dataframe() -> pd.DataFrame:
    """
    Get today's processed vehicle location data, using Redis cache if available.
    If cache miss, loads raw data, runs ML pipeline, and caches result.

    Returns:
        Processed DataFrame
    """
    today_str = get_campus_start_of_day().strftime('%Y-%m-%d')
    cache_key, timestamp_key = get_cache_and_timestamp_key(today_str)

    # Connect to Redis
    redis = await aioredis.from_url(settings.REDIS_URL)

    try:
        # Try to load from cache
        cached_data = await redis.get(cache_key)
        if cached_data:
            logger.info(f"Loaded {today_str} processed data from Redis cache")
            return pickle.loads(cached_data)

        # Load raw from database
        logger.info(f"Loading {today_str} raw data from database")
        raw_df = await load_today_dataframe()

        # Process data
        logger.info(f"Processing {len(raw_df)} records through ML pipeline...")
        processed_df = process_raw_dataframe(raw_df)

        # Save to cache
        # Serialize with pickle
        pickled_df = pickle.dumps(processed_df)

        # Store data and timestamp (expire in 24 hours)
        async with redis.pipeline() as pipe:
            # Set data and timestamp
            pipe.set(cache_key, pickled_df, ex=86400)
            pipe.set(timestamp_key, datetime.now(timezone.utc).isoformat(), ex=86400)
            await pipe.execute()

        # Invalidate smart_closest_point cache since dataframe was updated
        await FastAPICache.clear(namespace="smart_closest_point")

        logger.info(f"Saved {today_str} processed data to Redis cache")
        return processed_df

    finally:
        await redis.close()


async def update_today_dataframe() -> pd.DataFrame:
    """
    Update the cached processed dataframe with new data from the database.

    If cache exists:
        1. Gets new raw data since last update.
        2. Combines with previous raw data (reconstructed from cache).
        3. Re-runs ML pipeline on combined data.
        4. Updates cache.

    If cache doesn't exist:
        Calls get_today_dataframe() to load fresh data.

    Returns:
        Updated processed DataFrame
    """
    today_str = get_campus_start_of_day().strftime('%Y-%m-%d')
    cache_key, timestamp_key = get_cache_and_timestamp_key(today_str)

    redis = await aioredis.from_url(settings.REDIS_URL)

    try:
        # Check if cache exists
        cached_data = await redis.get(cache_key)
        last_updated_bytes = await redis.get(timestamp_key)

        if not cached_data or not last_updated_bytes:
            logger.info(f"No cache found for {today_str}, loading fresh data")
            await redis.close()
            return await get_today_dataframe()

        # Cache exists, perform incremental update
        logger.info(f"Cache found for {today_str}, checking for updates")
        last_updated = datetime.fromisoformat(last_updated_bytes.decode('utf-8'))

        # Load new raw data since last update
        new_raw_df = await load_today_dataframe(since=last_updated)
        logger.debug(f"Loaded {len(new_raw_df)} new raw records since {last_updated}")

        if new_raw_df.empty:
            logger.info("No new data since last update, returning cached data")
            return pickle.loads(cached_data)

        logger.info(f"Found {len(new_raw_df)} new raw records")

        # Load existing processed cache (already retrieved earlier)
        current_processed_df = pickle.loads(cached_data)

        # Reconstruct current raw dataframe from processed dataframe
        # We assume the pipeline preserves these original columns
        raw_cols = ['vehicle_id', 'latitude', 'longitude', 'timestamp']

        # Check if columns exist (safety check)
        if not all(col in current_processed_df.columns for col in raw_cols):
            logger.warning("Cached dataframe missing raw columns. Triggering full reload.")
            await redis.close()
            return await get_today_dataframe()

        current_raw_df = current_processed_df[raw_cols].copy()

        # Combine raw data
        combined_raw_df = pd.concat([current_raw_df, new_raw_df], ignore_index=True)
        # Sort and deduplicate
        combined_raw_df = combined_raw_df.sort_values('timestamp').drop_duplicates()

        # Run pipeline on combined raw data
        logger.info(f"Re-processing {len(combined_raw_df)} records...")
        updated_processed_df = process_raw_dataframe(combined_raw_df)

        # Update cache
        pickled_df = pickle.dumps(updated_processed_df)
        async with redis.pipeline() as pipe:
            pipe.set(cache_key, pickled_df, ex=86400)
            pipe.set(timestamp_key, datetime.now(timezone.utc).isoformat(), ex=86400)
            await pipe.execute()

        # Invalidate smart_closest_point cache since dataframe was updated
        await FastAPICache.clear(namespace="smart_closest_point")

        logger.info(f"Updated cache to {len(updated_processed_df)} processed records")

        return updated_processed_df

    finally:
        await redis.close()
