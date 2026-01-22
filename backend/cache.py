"""Custom Redis cache implementation for Shubble.

This module provides a simple, async-first caching system using Redis.
It replaces fastapi-cache2 with a more lightweight implementation.

Usage:
    from backend.cache import cache, clear_namespace, init_cache

    # Initialize at startup
    await init_cache(redis_url)

    # Use as decorator
    @cache(expire=60, namespace="locations")
    async def get_locations():
        ...

    # Clear a namespace
    await clear_namespace("locations")
"""
import functools
import logging
import pickle
from typing import Any, Callable, Optional, TypeVar, ParamSpec

from redis import asyncio as aioredis

logger = logging.getLogger(__name__)

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
    expire: int = 60,
    namespace: str = "default",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to cache async function results in Redis.

    Args:
        expire: Cache TTL in seconds (default: 60)
        namespace: Cache namespace for grouping related keys

    Returns:
        Decorated function

    Example:
        @cache(expire=300, namespace="locations")
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

            # Generate cache key
            cache_key = _make_key(namespace, func.__name__, args, kwargs)

            # Try to get from cache
            try:
                cached_data = await redis.get(cache_key)
                if cached_data is not None:
                    logger.info(f"Cache HIT: {cache_key}")
                    return pickle.loads(cached_data)
            except Exception as e:
                logger.warning(f"Cache read error for {cache_key}: {e}")

            # Cache miss - call function
            logger.info(f"Cache MISS: {cache_key}")
            result = await func(*args, **kwargs)

            # Store in cache
            try:
                pickled = pickle.dumps(result)
                await redis.set(cache_key, pickled, ex=expire)
            except Exception as e:
                logger.warning(f"Cache write error for {cache_key}: {e}")

            return result

        return wrapper
    return decorator


async def clear_namespace(namespace: str) -> int:
    """Clear all cache entries in a namespace.

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
            return pickle.loads(cached_data)
    except Exception as e:
        logger.warning(f"Error getting cached value for {cache_key}: {e}")

    return None


async def set_cached(
    namespace: str,
    func_name: str,
    value: Any,
    expire: int = 60,
    *args,
    **kwargs
) -> bool:
    """Set a cache value directly.

    Args:
        namespace: Cache namespace
        func_name: Function name
        value: Value to cache
        expire: TTL in seconds
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
        pickled = pickle.dumps(value)
        await redis.set(cache_key, pickled, ex=expire)
        return True
    except Exception as e:
        logger.error(f"Error setting cached value for {cache_key}: {e}")
        return False
