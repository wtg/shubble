"""Custom Redis cache implementation for Shubble.

This module provides a simple, async-first caching system using Redis.
It replaces fastapi-cache2 with a more lightweight implementation.

Supports soft/hard TTL:
    - soft_ttl: After this time, value is stale but still returned (logs STALE)
    - hard_ttl: After this time, value is deleted from Redis (cache miss)

Usage:
    from backend.cache import cache, soft_clear_namespace, init_cache

    # Initialize at startup
    await init_cache(redis_url)

    # Use as decorator with soft/hard TTL
    @cache(soft_ttl=15, hard_ttl=300, namespace="locations")
    async def get_locations():
        ...

    # Clear a namespace
    await soft_clear_namespace("locations")
"""
import asyncio
import functools
import logging
import pickle
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar, ParamSpec

from redis import asyncio as aioredis

logger = logging.getLogger(__name__)


@dataclass
class CachedValue:
    """Wrapper for cached values with soft TTL tracking."""
    value: Any
    soft_expiry: float  # Unix timestamp when soft TTL expires

# Global Redis client
_redis_client: Optional[aioredis.Redis] = None
_prefix: str = "shubble-cache"

P = ParamSpec('P')
T = TypeVar('T')


async def init_cache(redis_url: str, prefix: str = "shubble-cache") -> aioredis.Redis:
    """Initialize the cache with a Redis connection.

    Args:
        redis_url: Redis connection URL
        prefix: Key prefix for all cache entries

    Returns:
        Redis client instance
    """
    global _redis_client, _prefix
    _prefix = prefix
    _redis_client = await aioredis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=False,
    )
    logger.info(f"Cache initialized with prefix: {prefix}")
    return _redis_client


async def close_cache() -> None:
    """Close the Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Cache connection closed")


def get_redis() -> Optional[aioredis.Redis]:
    """Get the global Redis client."""
    return _redis_client


def _make_key(namespace: str, func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a cache key from function name and arguments.

    Args:
        namespace: Cache namespace
        func_name: Function name
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Cache key string
    """
    # Filter out non-serializable arguments (like session_factory)
    filtered_args = []
    for arg in args:
        if _is_serializable(arg):
            filtered_args.append(arg)

    filtered_kwargs = {}
    for k, v in kwargs.items():
        if _is_serializable(v):
            filtered_kwargs[k] = v

    # Build readable key parts
    key_parts = [_prefix, namespace, func_name]

    # Add args to key
    for arg in filtered_args:
        if isinstance(arg, (list, tuple)):
            # For lists, join elements with comma
            key_parts.append(",".join(str(x) for x in arg))
        else:
            key_parts.append(str(arg))

    # Add kwargs to key
    for k, v in sorted(filtered_kwargs.items()):
        if isinstance(v, (list, tuple)):
            key_parts.append(f"{k}={','.join(str(x) for x in v)}")
        else:
            key_parts.append(f"{k}={v}")

    return ":".join(key_parts)


def _is_serializable(obj: Any) -> bool:
    """Check if an object is JSON-serializable for cache key generation."""
    if obj is None:
        return True
    if isinstance(obj, (str, int, float, bool)):
        return True
    if isinstance(obj, (list, tuple)):
        return all(_is_serializable(item) for item in obj)
    if isinstance(obj, dict):
        return all(
            isinstance(k, str) and _is_serializable(v)
            for k, v in obj.items()
        )
    return False


def cache(
    soft_ttl: int = 60,
    hard_ttl: int = 300,
    namespace: str = "default",
    lock_timeout: float = 10.0,
    poll_interval: float = 0.1,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to cache async function results in Redis with soft/hard TTL.

    Includes stampede protection: when soft TTL expires, only one caller
    recomputes the value while others wait or return stale data.

    Args:
        soft_ttl: Soft TTL in seconds. After this, value is stale but still returned.
        hard_ttl: Hard TTL in seconds. After this, value is deleted from Redis.
        namespace: Cache namespace for grouping related keys.
        lock_timeout: Max seconds to wait for another caller to refresh the cache.
        poll_interval: Seconds between cache checks while waiting.

    Returns:
        Decorated function

    Example:
        @cache(soft_ttl=15, hard_ttl=300, namespace="locations")
        async def get_locations(vehicle_ids: list[str]):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            redis = get_redis()

            # If Redis not available, just call the function
            if redis is None:
                logger.warning(f"Cache not initialized, calling {func.__name__} directly")
                return await func(*args, **kwargs)

            # Generate cache key and lock key
            cache_key = _make_key(namespace, func.__name__, args, kwargs)
            lock_key = f"{cache_key}:lock"

            # Try to get from cache
            cached_data = None
            cached_value: Optional[CachedValue] = None
            try:
                cached_data = await redis.get(cache_key)
                if cached_data is not None:
                    cached_value = pickle.loads(cached_data)
            except Exception as e:
                logger.warning(f"Cache read error for {cache_key}: {e}")

            now = time.time()

            # Case 1: Fresh value - return immediately
            if cached_value is not None and now < cached_value.soft_expiry:
                logger.info(f"Cache HIT: {cache_key}")
                return cached_value.value

            # Case 2: Stale value or miss - need to refresh with stampede protection
            stale_value = cached_value.value if cached_value else None
            is_stale = cached_value is not None

            if is_stale:
                stale_seconds = int(now - cached_value.soft_expiry)
                logger.info(f"Cache STALE ({stale_seconds}s): {cache_key}")
            else:
                logger.info(f"Cache MISS: {cache_key}")

            # Try to acquire lock (SET NX with expiry)
            lock_acquired = await redis.set(
                lock_key, "1", nx=True, ex=int(lock_timeout) + 1
            )

            if lock_acquired:
                # We got the lock - recompute the value
                logger.info(f"Lock acquired, refreshing: {cache_key}")
                try:
                    result = await func(*args, **kwargs)

                    # Store in cache with soft expiry timestamp
                    new_cached = CachedValue(
                        value=result,
                        soft_expiry=time.time() + soft_ttl
                    )
                    pickled = pickle.dumps(new_cached)
                    await redis.set(cache_key, pickled, ex=hard_ttl)

                    return result
                finally:
                    # Release lock
                    await redis.delete(lock_key)
            else:
                # Another caller is refreshing - wait for fresh value or timeout
                logger.info(f"Lock held by another, waiting: {cache_key}")
                wait_start = time.time()

                while time.time() - wait_start < lock_timeout:
                    await asyncio.sleep(poll_interval)

                    # Check if fresh value is now available
                    try:
                        new_data = await redis.get(cache_key)
                        if new_data is not None:
                            new_cached = pickle.loads(new_data)
                            if new_cached.soft_expiry > now:
                                # Fresh value available
                                logger.info(f"Fresh value ready: {cache_key}")
                                return new_cached.value
                    except Exception:
                        pass

                # Timeout - return stale value if available, else compute
                if stale_value is not None:
                    logger.info(f"Lock timeout, returning stale: {cache_key}")
                    return stale_value
                else:
                    # No stale value and timeout - compute anyway
                    logger.info(f"Lock timeout, computing fallback: {cache_key}")
                    return await func(*args, **kwargs)

        return wrapper
    return decorator


async def clear_namespace(namespace: str) -> int:
    """Clear all cache entries in a namespace (hard delete).

    Args:
        namespace: Namespace to clear

    Returns:
        Number of keys deleted
    """
    redis = get_redis()
    if redis is None:
        logger.warning("Cache not initialized, cannot clear namespace")
        return 0

    pattern = f"{_prefix}:{namespace}:*"

    try:
        # Use SCAN to find keys (safer than KEYS for large datasets)
        deleted = 0
        async for key in redis.scan_iter(match=pattern):
            await redis.delete(key)
            deleted += 1

        if deleted > 0:
            logger.info(f"Cleared {deleted} keys from namespace '{namespace}'")
        return deleted
    except Exception as e:
        logger.error(f"Error clearing namespace '{namespace}': {e}")
        return 0


async def soft_clear_namespace(namespace: str) -> int:
    """Soft-clear all cache entries in a namespace by expiring their soft TTL.

    This marks all entries as stale without deleting them. The next request
    will log STALE but still return the cached value until the hard TTL expires.

    Args:
        namespace: Namespace to soft-clear

    Returns:
        Number of keys soft-cleared
    """
    redis = get_redis()
    if redis is None:
        logger.warning("Cache not initialized, cannot soft-clear namespace")
        return 0

    pattern = f"{_prefix}:{namespace}:*"
    now = time.time()

    try:
        cleared = 0
        async for key in redis.scan_iter(match=pattern):
            # Get current value and remaining TTL
            cached_data = await redis.get(key)
            remaining_ttl = await redis.ttl(key)

            if cached_data is None or remaining_ttl <= 0:
                continue

            try:
                cached: CachedValue = pickle.loads(cached_data)
                # Set soft_expiry to now (marking as stale)
                cached.soft_expiry = now
                # Save back with the same remaining hard TTL
                pickled = pickle.dumps(cached)
                await redis.set(key, pickled, ex=remaining_ttl)
                cleared += 1
            except Exception:
                # Skip malformed entries
                continue

        if cleared > 0:
            logger.info(f"Soft-cleared {cleared} keys in namespace '{namespace}'")
        return cleared
    except Exception as e:
        logger.error(f"Error soft-clearing namespace '{namespace}': {e}")
        return 0


async def delete_key(namespace: str, func_name: str, *args, **kwargs) -> bool:
    """Delete a specific cache key.

    Args:
        namespace: Cache namespace
        func_name: Function name
        *args: Function arguments used to generate the key
        **kwargs: Function keyword arguments

    Returns:
        True if key was deleted, False otherwise
    """
    redis = get_redis()
    if redis is None:
        return False

    cache_key = _make_key(namespace, func_name, args, kwargs)

    try:
        result = await redis.delete(cache_key)
        return result > 0
    except Exception as e:
        logger.error(f"Error deleting key {cache_key}: {e}")
        return False


async def get_cached(namespace: str, func_name: str, *args, **kwargs) -> Optional[Any]:
    """Get a cached value without calling the function.

    Args:
        namespace: Cache namespace
        func_name: Function name
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Cached value or None if not found
    """
    redis = get_redis()
    if redis is None:
        return None

    cache_key = _make_key(namespace, func_name, args, kwargs)

    try:
        cached_data = await redis.get(cache_key)
        if cached_data is not None:
            cached: CachedValue = pickle.loads(cached_data)
            return cached.value
    except Exception as e:
        logger.warning(f"Error getting cached value for {cache_key}: {e}")

    return None


async def set_cached(
    namespace: str,
    func_name: str,
    value: Any,
    soft_ttl: int = 60,
    hard_ttl: int = 300,
    *args,
    **kwargs
) -> bool:
    """Set a cache value directly.

    Args:
        namespace: Cache namespace
        func_name: Function name
        value: Value to cache
        soft_ttl: Soft TTL in seconds
        hard_ttl: Hard TTL in seconds (Redis expiry)
        *args: Function arguments for key generation
        **kwargs: Function keyword arguments

    Returns:
        True if set successfully, False otherwise
    """
    redis = get_redis()
    if redis is None:
        return False

    cache_key = _make_key(namespace, func_name, args, kwargs)

    try:
        cached_value = CachedValue(
            value=value,
            soft_expiry=time.time() + soft_ttl
        )
        pickled = pickle.dumps(cached_value)
        await redis.set(cache_key, pickled, ex=hard_ttl)
        return True
    except Exception as e:
        logger.error(f"Error setting cached value for {cache_key}: {e}")
        return False
