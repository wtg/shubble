"""Tests for the ETA API endpoint response shape."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


def make_mock_eta_row(vehicle_id: str, etas: dict, timestamp=None):
    """Create a mock ETA database row."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    row = MagicMock()
    row.vehicle_id = vehicle_id
    row.etas = etas
    row.timestamp = timestamp
    return row


@pytest.mark.asyncio
async def test_etas_returns_per_stop_shape():
    """Verify the API returns per-stop structure, not per-vehicle."""
    from backend.fastapi.utils import get_latest_etas

    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    mock_etas = [
        make_mock_eta_row("v1", {
            "COLONIE": {"eta": future, "route": "NORTH"},
            "GEORGIAN": {"eta": future, "route": "NORTH"},
        }),
    ]

    # Create a mock session factory
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_etas
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    # Bypass the cache decorator
    inner_fn = get_latest_etas.__wrapped__ if hasattr(get_latest_etas, '__wrapped__') else get_latest_etas

    result = await inner_fn(["v1"], mock_factory)

    # Keys should be stop names, not vehicle IDs
    assert "COLONIE" in result or "GEORGIAN" in result
    assert "v1" not in result  # vehicle_id should NOT be a top-level key

    # Each stop entry should have the right shape
    for stop_key, stop_data in result.items():
        assert "eta" in stop_data
        assert "vehicle_id" in stop_data
        assert "route" in stop_data


@pytest.mark.asyncio
async def test_etas_empty_when_no_vehicles():
    """Verify empty dict returned for no vehicles."""
    from backend.fastapi.utils import get_latest_etas

    inner_fn = get_latest_etas.__wrapped__ if hasattr(get_latest_etas, '__wrapped__') else get_latest_etas
    result = await inner_fn([], MagicMock())
    assert result == {}


@pytest.mark.asyncio
async def test_etas_filters_past_entries():
    """Verify past ETAs are excluded from response."""
    from backend.fastapi.utils import get_latest_etas

    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    mock_etas = [
        make_mock_eta_row("v1", {
            "COLONIE": {"eta": past, "route": "NORTH"},
            "GEORGIAN": {"eta": future, "route": "NORTH"},
        }),
    ]

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_etas
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    inner_fn = get_latest_etas.__wrapped__ if hasattr(get_latest_etas, '__wrapped__') else get_latest_etas
    result = await inner_fn(["v1"], mock_factory)

    # Past ETA should be excluded
    assert "COLONIE" not in result
    # Future ETA should be included
    assert "GEORGIAN" in result


@pytest.mark.asyncio
async def test_etas_picks_earliest_across_vehicles():
    """When multiple vehicles serve the same stop, earliest ETA wins."""
    from backend.fastapi.utils import get_latest_etas

    now = datetime.now(timezone.utc)
    early = (now + timedelta(minutes=5)).isoformat()
    late = (now + timedelta(minutes=15)).isoformat()

    mock_etas = [
        make_mock_eta_row("v1", {
            "STUDENT_UNION_RETURN": {"eta": late, "route": "NORTH"},
        }),
        make_mock_eta_row("v2", {
            "STUDENT_UNION_RETURN": {"eta": early, "route": "WEST"},
        }),
    ]

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_etas
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    inner_fn = get_latest_etas.__wrapped__ if hasattr(get_latest_etas, '__wrapped__') else get_latest_etas
    result = await inner_fn(["v1", "v2"], mock_factory)

    assert "STUDENT_UNION_RETURN" in result
    # v2 has earlier ETA
    assert result["STUDENT_UNION_RETURN"]["vehicle_id"] == "v2"
    assert result["STUDENT_UNION_RETURN"]["eta"] == early


@pytest.mark.asyncio
async def test_etas_handles_legacy_format():
    """Handle legacy ETA format (plain ISO string without route)."""
    from backend.fastapi.utils import get_latest_etas

    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    # Legacy format: {stop_key: iso_string} instead of {stop_key: {eta, route}}
    mock_etas = [
        make_mock_eta_row("v1", {
            "COLONIE": future,  # plain string, not dict
        }),
    ]

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_etas
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    inner_fn = get_latest_etas.__wrapped__ if hasattr(get_latest_etas, '__wrapped__') else get_latest_etas
    result = await inner_fn(["v1"], mock_factory)

    # Should still work — route will be empty string
    assert "COLONIE" in result
    assert result["COLONIE"]["route"] == ""
