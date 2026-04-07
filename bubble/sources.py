"""Async HTTP fetching for Bubble data sources."""
import asyncio
import logging
from html.parser import HTMLParser

import httpx

logger = logging.getLogger(__name__)

# Maximum characters to keep per source after text extraction.
MAX_SOURCE_CHARS = 15000

# Tags whose content should be skipped entirely during text extraction.
_SKIP_TAGS = frozenset({"head", "script", "style", "noscript", "nav", "footer", "aside"})


class _TextExtractor(HTMLParser):
    """Extracts visible text from HTML, skipping navigation/script/style sections."""

    def __init__(self) -> None:
        super().__init__()
        self._texts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:  # noqa: ARG002
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._texts.append(text)

    def get_text(self) -> str:
        return "\n".join(self._texts)


def _extract_text(html: str) -> str:
    """Return the visible text content of an HTML page."""
    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.get_text()


async def _fetch_one(url: str, client: httpx.AsyncClient) -> tuple[str, str | None]:
    """Fetch a single URL. Returns (url, text_content) or (url, None) on failure."""
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            text = _extract_text(response.text)
        else:
            text = response.text
        return url, text
    except Exception as e:
        logger.warning("Failed to fetch source %s: %s", url, e)
        return url, None


async def fetch_live_locations(backend_url: str) -> str | None:
    """Fetch /api/locations and return a human-readable summary for Gemini.

    Returns None if the fetch fails or the response is unexpected.
    """
    url = f"{backend_url.rstrip('/')}/api/locations"
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            data: dict = response.json()
    except Exception as e:
        logger.warning("Failed to fetch live locations from %s: %s", url, e)
        return None

    if not data:
        return "No shuttles are currently active in the geofence."

    lines = [f"{len(data)} shuttle(s) currently active in the geofence:"]
    for loc in data.values():
        name = loc.get("name", "Unknown")
        speed = loc.get("speed_mph")
        address = loc.get("formatted_location") or loc.get("address_name") or "unknown location"
        speed_str = f"{speed:.1f} mph" if speed is not None else "unknown speed"
        lines.append(f"  - {name}: {speed_str}, last seen at {address!r}")
    return "\n".join(lines)


async def fetch_all_sources(urls: list[str]) -> dict[str, str]:
    """Fetch all source URLs concurrently.

    Returns a dict mapping URL -> extracted text content (truncated to MAX_SOURCE_CHARS).
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
