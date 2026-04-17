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
import time
from contextlib import aclosing
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import select

from backend.cache import soft_clear_namespace, get_redis
from backend.database import get_db
from backend.models import VehicleLocation
from backend.time_utils import get_campus_start_of_day
from ml.pipelines import preprocess_pipeline, segment_pipeline, stops_pipeline

logger = logging.getLogger(__name__)

# P3: process-local cache for the deserialized dataframe. Collapses concurrent
# get_today_dataframe() calls (worker's asyncio.gather of per_stop_etas + trips
# + predict_next_state, and concurrent API requests) into a single pickle.loads
# within the TTL window. Stale-by-5s is fine: Redis cache has a 15s write cycle
# anyway, and the worker repopulates this cache from its own update path.
_PROCESS_DF_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_PROCESS_DF_CACHE_TTL = 5.0  # seconds
_PROCESS_DF_CACHE_LOCK: asyncio.Lock | None = None


def _get_process_cache_lock() -> asyncio.Lock:
    """Lazy-init the cache lock. Must be called from within a running loop."""
    global _PROCESS_DF_CACHE_LOCK
    if _PROCESS_DF_CACHE_LOCK is None:
        _PROCESS_DF_CACHE_LOCK = asyncio.Lock()
    return _PROCESS_DF_CACHE_LOCK


def _read_process_cache(date_str: str) -> pd.DataFrame | None:
    entry = _PROCESS_DF_CACHE.get(date_str)
    if entry is None:
        return None
    ts, df = entry
    if time.monotonic() - ts > _PROCESS_DF_CACHE_TTL:
        return None
    return df


def _write_process_cache(date_str: str, df: pd.DataFrame) -> None:
    _PROCESS_DF_CACHE[date_str] = (time.monotonic(), df)


def _invalidate_process_cache(date_str: str | None = None) -> None:
    if date_str is None:
        _PROCESS_DF_CACHE.clear()
    else:
        _PROCESS_DF_CACHE.pop(date_str, None)


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

    # Defensive: ensure timestamp is datetime64 regardless of source
    # (old cached data pickled before dc1f5b5 may have object dtype)
    if 'timestamp' in raw_df.columns and not pd.api.types.is_datetime64_any_dtype(raw_df['timestamp']):
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])

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
    start_of_day = get_campus_start_of_day()

    async with aclosing(get_db()) as gen:
        session = await anext(gen)
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

    # Convert to DataFrame
    df = pd.DataFrame(
        rows,
        columns=['vehicle_id', 'latitude', 'longitude', 'timestamp']
    )

    # Ensure timestamp is datetime64 (SQLAlchemy rows come back as Python datetime
    # objects which pandas stores as object dtype without explicit conversion)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


async def get_today_dataframe() -> pd.DataFrame:
    """
    Get today's processed vehicle location data.

    Lookup order:
      1. Process-local cache (5s TTL) — collapses concurrent calls
      2. Redis cache — 15s worker update cycle
      3. Load + process from database (cold path)

    Returns:
        Processed DataFrame
    """
    today_str = get_campus_start_of_day().strftime('%Y-%m-%d')

    # Fast path: process-local cache hit
    cached = _read_process_cache(today_str)
    if cached is not None:
        return cached

    # Serialize concurrent misses so we only deserialize/load once per window
    lock = _get_process_cache_lock()
    async with lock:
        # Double-check after acquiring lock
        cached = _read_process_cache(today_str)
        if cached is not None:
            return cached

        cache_key, timestamp_key = get_cache_and_timestamp_key(today_str)

        # Connect to Redis
        redis = get_redis()

        # Try to load from Redis cache
        if redis:
            cached_data = await redis.get(cache_key)
            if cached_data:
                logger.info(f"Loaded {today_str} processed data from Redis cache")
                df = pickle.loads(cached_data)
                _write_process_cache(today_str, df)
                return df

        # Load raw from database
        logger.info(f"Loading {today_str} raw data from database")
        raw_df = await load_today_dataframe()

        # Process data in thread pool to avoid blocking the event loop
        logger.info(f"Processing {len(raw_df)} records through ML pipeline...")
        processed_df = await asyncio.to_thread(process_raw_dataframe, raw_df)

        # Save to Redis cache
        if redis:
            pickled_df = pickle.dumps(processed_df)
            async with redis.pipeline() as pipe:
                pipe.set(cache_key, pickled_df, ex=86400)
                pipe.set(timestamp_key, datetime.now(timezone.utc).isoformat(), ex=86400)
                await pipe.execute()

            # Invalidate smart_closest_point cache since dataframe was updated
            await soft_clear_namespace("smart_closest_point")

        logger.info(f"Saved {today_str} processed data to Redis cache")
        _write_process_cache(today_str, processed_df)
        return processed_df


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

    # Connect to Redis
    redis = get_redis()

    # Check if cache exists
    if not redis:
        logger.info(f"No Redis, loading fresh data for {today_str}")
        return await get_today_dataframe()

    cached_data = await redis.get(cache_key)
    last_updated_bytes = await redis.get(timestamp_key)

    if not cached_data or not last_updated_bytes:
        logger.info(f"No cache found for {today_str}, loading fresh data")
        return await get_today_dataframe()

    # Cache exists, perform incremental update
    logger.info(f"Cache found for {today_str}, checking for updates")
    last_updated = datetime.fromisoformat(last_updated_bytes.decode('utf-8'))

    # Load new raw data since last update
    new_raw_df = await load_today_dataframe(since=last_updated)
    logger.info(f"Loaded {len(new_raw_df)} new raw records since {last_updated}")

    if new_raw_df.empty:
        logger.info("No new data since last update, returning cached data")
        df = pickle.loads(cached_data)
        _write_process_cache(today_str, df)
        return df

    logger.info(f"Found {len(new_raw_df)} new raw records")

    # Load existing processed cache
    current_processed_df = pickle.loads(cached_data)

    # Check if required columns exist
    raw_cols = ['vehicle_id', 'latitude', 'longitude', 'timestamp']
    if not all(col in current_processed_df.columns for col in raw_cols):
        logger.warning("Cached dataframe missing raw columns. Triggering full reload.")
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

    # Identify which rows are genuinely new (not context) and null out their
    # processed columns so process_raw_dataframe(additive=True) recomputes them.
    # P5: vectorized mask instead of per-row iterrows + .at writes.
    new_raw_index = pd.MultiIndex.from_arrays(
        [new_raw_df['vehicle_id'].astype(str), new_raw_df['timestamp']]
    )
    row_index = pd.MultiIndex.from_arrays(
        [rows_to_process['vehicle_id'].astype(str), rows_to_process['timestamp']]
    )
    is_new_row = row_index.isin(new_raw_index)
    if 'route' in rows_to_process.columns and processed_cols:
        rows_to_process.loc[is_new_row, processed_cols] = pd.NA
    elif 'route' not in rows_to_process.columns:
        # Route column absent — fall back to nulling every row's processed cols
        for col in processed_cols:
            rows_to_process[col] = pd.NA

    logger.info(f"Processing {len(rows_to_process)} rows with additive mode (preserves context, computes new)")
    newly_processed_df = await asyncio.to_thread(process_raw_dataframe, rows_to_process)

    # P5: filter truly-new rows with the same vectorized MultiIndex.isin trick
    # instead of building a python set + list comprehension. Cheap on 30K rows.
    processed_index = pd.MultiIndex.from_arrays(
        [newly_processed_df['vehicle_id'].astype(str), newly_processed_df['timestamp']]
    )
    genuinely_new_processed = newly_processed_df[processed_index.isin(new_raw_index)].copy()

    logger.info(f"Adding {len(genuinely_new_processed)} newly processed records to cache")

    # Combine with existing processed data
    updated_processed_df = pd.concat(
        [current_processed_df, genuinely_new_processed],
        ignore_index=True
    )

    # P5: sort+drop_duplicates on near-sorted data is ~2x faster with
    # kind='mergesort' (timsort-backed). Sort by (vehicle_id, timestamp) so
    # the subsequent dedup is both stable and matches the index schema.
    updated_processed_df = updated_processed_df.sort_values(
        ['vehicle_id', 'timestamp'], kind='mergesort'
    ).drop_duplicates(
        subset=['vehicle_id', 'timestamp'], keep='last'
    )

    # Re-apply the co-located-stop remap across the ENTIRE merged
    # dataframe. Incremental updates only process a small context
    # window, so old rows keep whatever stop_name was assigned by
    # earlier pipeline versions. Running the remap on the full
    # merged df here is cheap (vectorized per route) and ensures new
    # rules (e.g. STUDENT_UNION_RETURN disambiguation) propagate to
    # historical rows without forcing a cold reprocess.
    from ml.data.stops import resolve_duplicate_stops
    if not updated_processed_df.empty and 'polyline_idx' in updated_processed_df.columns:
        resolve_duplicate_stops(
            updated_processed_df,
            route_column='route',
            polyline_index_column='polyline_idx',
            stop_column='stop_name',
        )

    # Update cache
    if redis:
        pickled_df = pickle.dumps(updated_processed_df)
        async with redis.pipeline() as pipe:
            pipe.set(cache_key, pickled_df, ex=86400)
            pipe.set(timestamp_key, datetime.now(timezone.utc).isoformat(), ex=86400)
            await pipe.execute()

        # Invalidate smart_closest_point cache since dataframe was updated
        await soft_clear_namespace("smart_closest_point")

    logger.info(f"Updated cache to {len(updated_processed_df)} processed records")

    # Populate process-local cache so the worker's subsequent
    # get_today_dataframe() calls (per_stop_etas / trips / predict_next_state
    # running under asyncio.gather) return instantly without re-deserializing.
    _write_process_cache(today_str, updated_processed_df)

    return updated_processed_df
