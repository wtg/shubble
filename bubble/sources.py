"""Async HTTP fetching for Bubble data sources."""
import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

# Maximum characters to keep per source before truncating.
# RPI shuttle page has schedule/closure info starting ~4,500 chars in, so keep plenty of headroom.
MAX_SOURCE_CHARS = 15000


async def _fetch_one(url: str, client: httpx.AsyncClient) -> tuple[str, str | None]:
    """Fetch a single URL. Returns (url, content) or (url, None) on failure."""
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        return url, response.text
    except Exception as e:
        logger.warning("Failed to fetch source %s: %s", url, e)
        return url, None


async def fetch_all_sources(urls: list[str]) -> dict[str, str]:
    """Fetch all source URLs concurrently.

    Returns a dict mapping URL -> content (truncated to MAX_SOURCE_CHARS).
    URLs that fail to fetch are omitted from the result.
    """
    if not urls:
        return {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        results = await asyncio.gather(*[_fetch_one(url, client) for url in urls])

    output: dict[str, str] = {}
    for url, content in results:
        if content is None:
            continue
        if len(content) > MAX_SOURCE_CHARS:
            content = content[:MAX_SOURCE_CHARS] + f"\n[truncated — {len(content) - MAX_SOURCE_CHARS} chars omitted]"
        output[url] = content

    return output
