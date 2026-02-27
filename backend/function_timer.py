import asyncio
import functools
import logging
import time

logger = logging.getLogger(__name__)


def timed(func):
    """Decorator that logs the execution time of a function."""
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logger.info("%.3fs %s.%s", elapsed, func.__module__, func.__qualname__)
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logger.info("%.3fs %s.%s", elapsed, func.__module__, func.__qualname__)
        return sync_wrapper
