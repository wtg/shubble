"""Data loading utilities.

This module provides efficient caching and incremental updates for vehicle location data.

Key optimization: The update_today_dataframe() function uses a windowed approach to only
recompute affected rows instead of reprocessing the entire dataset:
  1. relevant_rows() identifies which rows need recomputation (new rows + context window)
  2. Only those rows are processed through the ML pipeline
  3. New processed rows are merged with existing cache

This significantly reduces computation time as the dataset grows throughout the day.
"""
import asyncio
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

    Uses additive mode to only compute missing values (NaN) and disables
    strict majority requirement for route cleaning to allow endpoint fills.

    Args:
        raw_df: Raw vehicle location DataFrame (may contain mix of raw and processed rows)
    """
    if raw_df.empty:
        return raw_df

    # Run pipelines in sequence, injecting the dataframe to bypass disk cache loading
    # We pass flag=True to ensure any internal checks know we want to compute
    # We pass cache=False to disable disk caching (Redis cache is sufficient)
    # additive=True: Only compute closest points for rows with NaN values
    # require_majority_valid=False: Use strict majority for route cleaning (default behavior)
    df = preprocess_pipeline(df=raw_df, preprocess=True, cache=False, additive=True)
    df = segment_pipeline(df=df, segment=True, cache=False, min_segment_length=1,
                         require_majority_valid=False, additive=True)
    df = stops_pipeline(df=df, stops=True, cache=False)

    return df


def relevant_rows(
    cached_df: pd.DataFrame,
    new_raw_df: pd.DataFrame,
    window_size: int = 5
) -> pd.DataFrame:
    """
    Get rows from cached dataframe that need recomputation when new data arrives.

    For each vehicle_id in new_raw_df, includes window_size // 2 previous points
    from cached_df to ensure windowed operations (like route cleaning) work correctly.

    Args:
        cached_df: Existing processed dataframe with raw columns
        new_raw_df: New raw rows to be added
        window_size: Window size used in ML pipeline (default 5)

    Returns:
        DataFrame containing:
        - Previous context rows (window_size // 2 per vehicle)
        - New raw rows
    """
    if cached_df.empty or new_raw_df.empty:
        return new_raw_df

    # Get unique vehicles in new data
    new_vehicle_ids = new_raw_df['vehicle_id'].unique()

    # Number of previous points needed per vehicle
    context_points = window_size // 2

    relevant_rows_list = []

    for vehicle_id in new_vehicle_ids:
        # Get previous points for this vehicle from cache
        vehicle_cached = cached_df[
            cached_df['vehicle_id'] == vehicle_id
        ].sort_values('timestamp')

        # Take last N points as context
        if len(vehicle_cached) > 0:
            context = vehicle_cached.tail(context_points)
            relevant_rows_list.append(context)

        # Add new points for this vehicle
        vehicle_new = new_raw_df[new_raw_df['vehicle_id'] == vehicle_id]
        relevant_rows_list.append(vehicle_new)

    # Combine all relevant rows
    if relevant_rows_list:
        result = pd.concat(relevant_rows_list, ignore_index=True)
        # Sort and deduplicate using (vehicle_id, timestamp) as key
        # In case context overlaps with new data
        result = result.sort_values('timestamp').drop_duplicates(
            subset=['vehicle_id', 'timestamp'], keep='last'
        )
        return result
    else:
        return new_raw_df


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

        # Process data in thread pool to avoid blocking the event loop
        logger.info(f"Processing {len(raw_df)} records through ML pipeline...")
        processed_df = await asyncio.to_thread(process_raw_dataframe, raw_df)

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


async def update_today_dataframe(window_size: int = 5) -> pd.DataFrame:
    """
    Update the cached processed dataframe with new data from the database.

    Optimized version that combines windowed context with additive processing
    to only compute missing values rather than reprocessing the entire dataset.

    If cache exists:
        1. Gets new raw data since last update.
        2. Uses relevant_rows() to extract new rows + context (window_size // 2 per vehicle).
        3. Sets processed columns to NaN for new rows only (preserves context values).
        4. Processes with additive=True (only computes NaN values) and require_majority_valid=True.
        5. Filters out genuinely new processed rows.
        6. Merges new processed rows with existing cache.
        7. Updates cache.

    If cache doesn't exist:
        Calls get_today_dataframe() to load fresh data.

    Args:
        window_size: Window size for ML pipeline operations (default 5)

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
        logger.info(f"Loaded {len(new_raw_df)} new raw records since {last_updated}")

        if new_raw_df.empty:
            logger.info("No new data since last update, returning cached data")
            return pickle.loads(cached_data)

        logger.info(f"Found {len(new_raw_df)} new raw records")

        # Load existing processed cache
        current_processed_df = pickle.loads(cached_data)

        # Check if required columns exist
        raw_cols = ['vehicle_id', 'latitude', 'longitude', 'timestamp']
        if not all(col in current_processed_df.columns for col in raw_cols):
            logger.warning("Cached dataframe missing raw columns. Triggering full reload.")
            await redis.close()
            return await get_today_dataframe()

        # Get rows that need recomputation (new rows + context)
        # This uses the windowed approach to include context for each vehicle
        rows_to_process = relevant_rows(current_processed_df, new_raw_df, window_size)
        logger.info(f"Extracted {len(rows_to_process)} rows to process (includes {window_size // 2} context points per vehicle)")

        # Get all processed columns (excluding raw columns)
        processed_cols = [col for col in current_processed_df.columns if col not in raw_cols]

        # Initialize processed columns with NaN if they don't exist
        for col in processed_cols:
            if col not in rows_to_process.columns:
                rows_to_process[col] = pd.NA

        # Identify which rows are genuinely new (not context)
        # Context rows will already have processed values, new rows will have NaN
        # We determine this by checking if 'route' column exists and has values
        if 'route' in rows_to_process.columns:
            # Rows from cache (context) will have route values
            # New raw rows will have NaN for route
            # Mark rows with NaN route as new (need processing)
            new_row_keys = set(zip(new_raw_df['vehicle_id'], new_raw_df['timestamp']))
            for idx, row in rows_to_process.iterrows():
                key = (row['vehicle_id'], row['timestamp'])
                if key in new_row_keys:
                    # This is a new row - set all processed columns to NaN
                    for col in processed_cols:
                        rows_to_process.at[idx, col] = pd.NA
        else:
            # If route column doesn't exist, all rows need processing
            for col in processed_cols:
                rows_to_process[col] = pd.NA

        logger.info(f"Processing {len(rows_to_process)} rows with additive mode (preserves context, computes new)")
        newly_processed_df = await asyncio.to_thread(process_raw_dataframe, rows_to_process)

        # Create a set of (vehicle_id, timestamp) pairs that are genuinely new
        # (not from the context window)
        new_keys = set(zip(new_raw_df['vehicle_id'], new_raw_df['timestamp']))

        # Filter newly_processed_df to only include truly new rows
        # by checking if (vehicle_id, timestamp) exists in new_keys
        newly_processed_keys = list(zip(
            newly_processed_df['vehicle_id'],
            newly_processed_df['timestamp']
        ))
        mask = [key in new_keys for key in newly_processed_keys]
        genuinely_new_processed = newly_processed_df[mask].copy()

        logger.info(f"Adding {len(genuinely_new_processed)} newly processed records to cache")

        # Combine with existing processed data
        updated_processed_df = pd.concat(
            [current_processed_df, genuinely_new_processed],
            ignore_index=True
        )

        # Sort and deduplicate using (vehicle_id, timestamp) as key
        # Keep 'last' to prefer newly processed rows over cached ones in case of updates
        updated_processed_df = updated_processed_df.sort_values('timestamp').drop_duplicates(
            subset=['vehicle_id', 'timestamp'], keep='last'
        )

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
