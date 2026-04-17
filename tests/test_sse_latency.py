"""Regression guard for SSE push latency (Phase 2.4 of the latency pass).

These tests lock in the behavior of /api/trips/stream and
/api/locations/stream under Phase 2.2 — specifically that they forward a
Redis pub/sub notification to connected SSE clients within a negligible
budget (<1 s). The end-to-end "GPS ping to UI" latency target from the
handoff plan is <10 s worst case, but that budget also covers the
worker cycle (~5 s), DB insert, and compute; the SSE-forwarding layer
itself should be roughly instant and is what we guard here.

No real Redis is required — the tests mock the `get_redis()` module
lookup with an in-process `MagicMock` whose `pubsub()` behaves like the
real one for this test's purposes (returns one "message" dict, then
None). This lets the suite run in CI without any service dependencies
and still catches regressions in the SSE wiring (lost yields, wrong
data frames, missing initial emit, keepalive format drift).
"""

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# Latency budgets ------------------------------------------------------------
#
# The SSE forwarding layer runs in-process and should react to a pub/sub
# tick almost instantly. 1 s is a generous ceiling; a regression that
# drops below this threshold (e.g. an accidental 5 s sleep, a broken
# await chain, or a pubsub.get_message() path that swallows messages)
# would blow this budget.
SSE_FORWARD_LATENCY_BUDGET_SEC = 1.0


def _mock_pubsub_with_single_message():
    """Return a MagicMock pubsub that yields one 'message' event then None.

    Shape matches what the real redis.asyncio pubsub exposes: subscribe,
    unsubscribe, close, and get_message(ignore_subscribe_messages, timeout)
    are all awaitables. The first get_message() call returns a message
    dict (simulating a pub/sub notification arriving); subsequent calls
    sleep briefly then return None (simulating the keepalive timeout).
    """
    call_count = {"n": 0}

    async def get_message_sim(ignore_subscribe_messages=True, timeout=15.0):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"type": "message", "data": b"1", "channel": b"test"}
        # Subsequent calls: keepalive path — sleep and return None.
        await asyncio.sleep(0.05)
        return None

    mock = MagicMock()
    mock.subscribe = AsyncMock(return_value=None)
    mock.unsubscribe = AsyncMock(return_value=None)
    mock.close = AsyncMock(return_value=None)
    mock.get_message = get_message_sim
    return mock


def _mock_request_with_disconnect_after(n_checks: int):
    """Return a MagicMock Request that reports disconnected after N checks.

    The SSE generators poll `request.is_disconnected()` at the top of
    each loop iteration. Setting N=3 lets the generator do: initial
    emit → first get_message (returns message, yields data frame) →
    is_disconnected() → break. Adjust N if the generator's loop shape
    changes.
    """
    check_count = {"n": 0}

    async def is_disconnected():
        check_count["n"] += 1
        return check_count["n"] >= n_checks

    request = MagicMock()
    request.is_disconnected = is_disconnected
    return request


async def _collect_body_chunks(streaming_response, max_chunks: int, timeout: float):
    """Pull up to `max_chunks` chunks from a StreamingResponse body iterator.

    Returns a list of (monotonic_elapsed_sec, decoded_text) tuples. Stops
    when max_chunks is reached OR the generator terminates OR `timeout`
    is exceeded (whichever comes first).
    """
    events: list[tuple[float, str]] = []
    start = time.monotonic()

    async def _pull():
        async for chunk in streaming_response.body_iterator:
            text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
            events.append((time.monotonic() - start, text))
            if len(events) >= max_chunks:
                break

    try:
        await asyncio.wait_for(_pull(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_trips_initial_emit_and_pubsub_forward_fast():
    """stream_trips must emit the current payload on connect AND forward
    a subsequent pub/sub message without blocking.

    This is the core regression guard: if someone accidentally drops
    the `yield await _current_payload()` initial emit, or breaks the
    pubsub → yield chain, both assertions here fail loudly.
    """
    from backend.fastapi.routes import stream_trips

    fake_trips_live = b'[{"trip_id":"test","route":"TEST","departure_time":"2026-04-10T00:00:00+00:00","actual_departure":null,"scheduled":false,"vehicle_id":null,"status":"scheduled","stop_etas":{}}]'

    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value=fake_trips_live)
    mock_redis.pubsub = MagicMock(return_value=_mock_pubsub_with_single_message())

    request = _mock_request_with_disconnect_after(n_checks=2)

    with patch("backend.fastapi.routes.get_redis", return_value=mock_redis):
        response = await stream_trips(request)
        events = await _collect_body_chunks(response, max_chunks=2, timeout=5.0)

    # 1. Initial emit on connect.
    assert len(events) >= 1, f"no initial emit; events={events}"
    first_elapsed, first_text = events[0]
    assert first_text.startswith("data: "), f"bad initial frame: {first_text!r}"
    assert '"trip_id":"test"' in first_text
    assert first_elapsed < SSE_FORWARD_LATENCY_BUDGET_SEC, (
        f"initial emit took {first_elapsed:.3f}s, budget "
        f"{SSE_FORWARD_LATENCY_BUDGET_SEC}s"
    )

    # 2. Pushed update (the pubsub message → another data frame).
    assert len(events) >= 2, f"no pushed update; events={events}"
    second_elapsed, second_text = events[1]
    assert second_text.startswith("data: "), f"bad push frame: {second_text!r}"
    assert '"trip_id":"test"' in second_text
    assert second_elapsed < SSE_FORWARD_LATENCY_BUDGET_SEC, (
        f"pushed update took {second_elapsed:.3f}s, budget "
        f"{SSE_FORWARD_LATENCY_BUDGET_SEC}s"
    )


@pytest.mark.asyncio
async def test_stream_trips_without_redis_falls_back_to_empty_frame():
    """When Redis is down, stream_trips must still emit an initial data
    frame (with an empty array) and exit cleanly so the client can
    reconnect or fall back to polling — NOT hang the request forever.
    """
    from backend.fastapi.routes import stream_trips

    request = _mock_request_with_disconnect_after(n_checks=1)

    with patch("backend.fastapi.routes.get_redis", return_value=None):
        response = await stream_trips(request)
        events = await _collect_body_chunks(response, max_chunks=5, timeout=2.0)

    assert events, "expected at least the initial empty frame"
    first_text = events[0][1]
    assert first_text.startswith("data: ")
    # Empty-Redis path sends "data: []\n\n"
    assert "[]" in first_text


@pytest.mark.asyncio
async def test_stream_locations_initial_emit_and_pubsub_forward_fast():
    """Same latency contract for /api/locations/stream.

    The locations stream builds its payload via _build_locations_payload
    (which hits the DB through the cache), so we mock that helper too.
    """
    from backend.fastapi.routes import stream_locations

    fake_payload = {"v001": {"name": "001", "latitude": 42.7, "longitude": -73.6}}

    mock_redis = MagicMock()
    mock_redis.pubsub = MagicMock(return_value=_mock_pubsub_with_single_message())

    request = _mock_request_with_disconnect_after(n_checks=2)
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.session_factory = MagicMock()  # helper is mocked, unused

    with patch("backend.fastapi.routes.get_redis", return_value=mock_redis), \
         patch(
             "backend.fastapi.routes._build_locations_payload",
             new=AsyncMock(return_value=(fake_payload, None)),
         ):
        response = await stream_locations(request)
        events = await _collect_body_chunks(response, max_chunks=2, timeout=5.0)

    assert len(events) >= 1
    first_elapsed, first_text = events[0]
    assert first_text.startswith("data: ")
    assert '"v001"' in first_text
    assert first_elapsed < SSE_FORWARD_LATENCY_BUDGET_SEC, (
        f"initial emit took {first_elapsed:.3f}s"
    )

    assert len(events) >= 2
    second_elapsed, second_text = events[1]
    assert second_text.startswith("data: ")
    assert '"v001"' in second_text
    assert second_elapsed < SSE_FORWARD_LATENCY_BUDGET_SEC, (
        f"pushed update took {second_elapsed:.3f}s"
    )
