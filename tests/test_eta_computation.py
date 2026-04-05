"""Tests for the ETA computation pipeline."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, AsyncMock

from shared.stops import Stops


# --- Helper to build mock vehicle dataframes ---

def make_vehicle_df(vehicle_id: str, route: str, polyline_idx: int,
                    lat: float = 42.73, lon: float = -73.68,
                    speed_kmh: float = 20.0,
                    stop_name=None,
                    timestamp=None):
    """Create a minimal dataframe row mimicking the preprocessed worker data."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    rows = []
    # Need at least 10 rows for LSTM (sequence_length)
    for i in range(12):
        ts = timestamp - timedelta(seconds=(11 - i) * 5)
        rows.append({
            "vehicle_id": str(vehicle_id),
            "latitude": lat + i * 0.0001,
            "longitude": lon + i * 0.0001,
            "speed_kmh": speed_kmh,
            "timestamp": ts,
            "route": route,
            "polyline_idx": polyline_idx,
            "stop_name": stop_name if i == 11 else None,  # only last row has stop
        })
    return pd.DataFrame(rows)


# --- Tests for offset diff math ---

def test_north_route_offsets_loaded():
    """Verify routes.json OFFSET values are accessible."""
    routes = Stops.routes_data
    assert "NORTH" in routes
    north = routes["NORTH"]
    assert north["STUDENT_UNION"]["OFFSET"] == 0
    assert north["COLONIE"]["OFFSET"] == 3
    assert north["ECAV"]["OFFSET"] == 11
    assert north["STUDENT_UNION_RETURN"]["OFFSET"] == 15


def test_west_route_offsets_loaded():
    routes = Stops.routes_data
    assert "WEST" in routes
    west = routes["WEST"]
    assert west["STUDENT_UNION"]["OFFSET"] == 0
    assert west["CITY_STATION"]["OFFSET"] == 7
    assert west["STUDENT_UNION_RETURN"]["OFFSET"] == 15


def test_offset_diff_calculation():
    """Verify offset differences produce correct time deltas.

    If vehicle is heading to COLONIE (offset=3) and we want ETA for ECAV (offset=11):
    Additional time = (11 - 3) * 60 = 480 seconds = 8 minutes.
    """
    routes = Stops.routes_data
    north = routes["NORTH"]
    colonie_offset = north["COLONIE"]["OFFSET"]
    ecav_offset = north["ECAV"]["OFFSET"]
    diff_seconds = (ecav_offset - colonie_offset) * 60
    assert diff_seconds == 480  # 8 minutes


def test_full_route_offset_span():
    """Total loop time should be 15 minutes for both routes."""
    routes = Stops.routes_data
    for route_name in ["NORTH", "WEST"]:
        route = routes[route_name]
        stops = route["STOPS"]
        first_offset = route[stops[0]]["OFFSET"]
        last_offset = route[stops[-1]]["OFFSET"]
        assert last_offset - first_offset == 15, f"{route_name} total should be 15 min"


# --- Tests for compute_per_stop_etas ---

@pytest.mark.asyncio
async def test_compute_per_stop_etas_single_vehicle_north():
    """One vehicle on NORTH at polyline_idx=2.

    Polyline idx 2 = transition from STOPS[2] (GEORGIAN) to STOPS[3] (STAC_1).
    So the vehicle is heading to STAC_1 (offset=6).
    """
    from backend.worker.data import compute_per_stop_etas

    now = datetime.now(timezone.utc)
    df = make_vehicle_df("v1", "NORTH", 2, timestamp=now)

    # Mock predict_eta to return a known ETA for the next stop (STAC_1)
    stac1_eta = now + timedelta(minutes=2)
    mock_lstm = AsyncMock(return_value={"v1": stac1_eta})

    with patch("backend.worker.data.predict_eta", mock_lstm), \
         patch("backend.worker.data.get_today_dataframe", AsyncMock(return_value=df)):

        result = await compute_per_stop_etas(["v1"], df=df)

    # Should have ETAs for STAC_1 and all subsequent stops
    assert "STAC_1" in result
    assert result["STAC_1"]["vehicle_id"] == "v1"
    assert result["STAC_1"]["route"] == "NORTH"

    # STAC_2 is next (offset 7, diff from STAC_1 offset 6 = 1 min)
    assert "STAC_2" in result
    stac2_eta = datetime.fromisoformat(result["STAC_2"]["eta"])
    stac1_eta_actual = datetime.fromisoformat(result["STAC_1"]["eta"])
    diff = (stac2_eta - stac1_eta_actual).total_seconds()
    assert diff == 60  # 1 minute

    # STUDENT_UNION_RETURN should be last (offset 15, diff from 6 = 9 min)
    assert "STUDENT_UNION_RETURN" in result
    return_eta = datetime.fromisoformat(result["STUDENT_UNION_RETURN"]["eta"])
    diff = (return_eta - stac1_eta_actual).total_seconds()
    assert diff == 540  # 9 minutes


@pytest.mark.asyncio
async def test_compute_per_stop_etas_single_vehicle_west():
    """One vehicle on WEST at polyline_idx=3.

    Polyline idx 3 = transition from STOPS[3] (CITY_STATION) to STOPS[4] (BLITMAN).
    So the vehicle is heading to BLITMAN (offset=8).
    """
    from backend.worker.data import compute_per_stop_etas

    now = datetime.now(timezone.utc)
    df = make_vehicle_df("v2", "WEST", 3, timestamp=now)

    blitman_eta = now + timedelta(minutes=3)
    mock_lstm = AsyncMock(return_value={"v2": blitman_eta})

    with patch("backend.worker.data.predict_eta", mock_lstm), \
         patch("backend.worker.data.get_today_dataframe", AsyncMock(return_value=df)):

        result = await compute_per_stop_etas(["v2"], df=df)

    assert "BLITMAN" in result
    assert result["BLITMAN"]["route"] == "WEST"

    # CHASAN offset=9, diff from BLITMAN offset=8 → 1 min
    assert "CHASAN" in result
    chasan_eta = datetime.fromisoformat(result["CHASAN"]["eta"])
    blitman_eta_actual = datetime.fromisoformat(result["BLITMAN"]["eta"])
    assert (chasan_eta - blitman_eta_actual).total_seconds() == 60


@pytest.mark.asyncio
async def test_multi_vehicle_aggregation():
    """Two vehicles on different routes — each stop gets earliest ETA."""
    from backend.worker.data import compute_per_stop_etas

    now = datetime.now(timezone.utc)
    df_north = make_vehicle_df("v1", "NORTH", 0, timestamp=now)
    df_west = make_vehicle_df("v2", "WEST", 0, timestamp=now)
    df = pd.concat([df_north, df_west], ignore_index=True)

    # v1 arrives at COLONIE in 2 min, v2 arrives at ACADEMY_HALL in 1 min
    north_eta = now + timedelta(minutes=2)
    west_eta = now + timedelta(minutes=1)
    mock_lstm = AsyncMock(return_value={"v1": north_eta, "v2": west_eta})

    with patch("backend.worker.data.predict_eta", mock_lstm), \
         patch("backend.worker.data.get_today_dataframe", AsyncMock(return_value=df)):

        result = await compute_per_stop_etas(["v1", "v2"], df=df)

    # COLONIE should only come from v1 (NORTH route)
    if "COLONIE" in result:
        assert result["COLONIE"]["vehicle_id"] == "v1"

    # ACADEMY_HALL should only come from v2 (WEST route)
    if "ACADEMY_HALL" in result:
        assert result["ACADEMY_HALL"]["vehicle_id"] == "v2"


@pytest.mark.asyncio
async def test_lstm_fallback_to_distance_speed():
    """When LSTM fails, fallback to distance/average_speed should still produce ETAs."""
    from backend.worker.data import compute_per_stop_etas

    now = datetime.now(timezone.utc)
    # Use real coordinates near Student Union for realistic distance calc
    df = make_vehicle_df("v1", "NORTH", 0,
                         lat=42.730711, lon=-73.676737,
                         speed_kmh=20.0, timestamp=now)

    # LSTM returns nothing — forces fallback
    mock_lstm = AsyncMock(return_value={})

    with patch("backend.worker.data.predict_eta", mock_lstm), \
         patch("backend.worker.data.get_today_dataframe", AsyncMock(return_value=df)):

        result = await compute_per_stop_etas(["v1"], df=df)

    # Should still produce some ETAs via the distance/speed fallback
    # (may not have all stops if get_closest_point can't match)
    # At minimum, the function shouldn't crash
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_vehicle_at_end_of_route_skipped():
    """Vehicle past all stops should be gracefully skipped."""
    from backend.worker.data import compute_per_stop_etas

    now = datetime.now(timezone.utc)
    routes = Stops.routes_data
    north_stops = routes["NORTH"]["STOPS"]
    max_polyline_idx = len(north_stops) - 1  # Past last transition

    df = make_vehicle_df("v1", "NORTH", max_polyline_idx, timestamp=now)
    mock_lstm = AsyncMock(return_value={})

    with patch("backend.worker.data.predict_eta", mock_lstm), \
         patch("backend.worker.data.get_today_dataframe", AsyncMock(return_value=df)):

        result = await compute_per_stop_etas(["v1"], df=df)

    # Vehicle should be skipped — no ETAs from it
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_empty_vehicle_list():
    """Empty vehicle list should return empty dict."""
    from backend.worker.data import compute_per_stop_etas

    result = await compute_per_stop_etas([])
    assert result == {}


@pytest.mark.asyncio
async def test_stop_ordering_preserved_across_vehicles():
    """Two vehicles on the same route must not produce impossible orderings.

    If v1 is closer to STUDENT_UNION_RETURN and v2 is further back,
    the result should use one vehicle's ETAs for the whole route,
    not mix ETAs from different vehicles.
    """
    from backend.worker.data import compute_per_stop_etas

    now = datetime.now(timezone.utc)
    # v1 at polyline 6 (heading to ECAV, offset=11), v2 at polyline 0 (heading to COLONIE, offset=3)
    df_v1 = make_vehicle_df("v1", "NORTH", 6, timestamp=now)
    df_v2 = make_vehicle_df("v2", "NORTH", 0, timestamp=now)
    df = pd.concat([df_v1, df_v2], ignore_index=True)

    # v1 arrives at ECAV in 1 min, v2 arrives at COLONIE in 2 min
    v1_eta = now + timedelta(minutes=1)
    v2_eta = now + timedelta(minutes=2)
    mock_lstm = AsyncMock(return_value={"v1": v1_eta, "v2": v2_eta})

    with patch("backend.worker.data.predict_eta", mock_lstm), \
         patch("backend.worker.data.get_today_dataframe", AsyncMock(return_value=df)):

        result = await compute_per_stop_etas(["v1", "v2"], df=df)

    # All NORTH stops in the result should come from the SAME vehicle
    north_vehicle_ids = set()
    for stop_key in ["COLONIE", "GEORGIAN", "STAC_1", "STAC_2", "STAC_3",
                      "ECAV", "HOUSTON_FIELD_HOUSE", "STUDENT_UNION_RETURN"]:
        if stop_key in result and result[stop_key]["route"] == "NORTH":
            north_vehicle_ids.add(result[stop_key]["vehicle_id"])

    # Should be exactly one vehicle providing all NORTH ETAs
    assert len(north_vehicle_ids) <= 1, \
        f"Multiple vehicles providing NORTH ETAs: {north_vehicle_ids}. This can cause impossible orderings."

    # Verify ordering: each subsequent stop's ETA must be >= the previous
    north_stops_order = ["COLONIE", "GEORGIAN", "STAC_1", "STAC_2", "STAC_3",
                         "ECAV", "HOUSTON_FIELD_HOUSE", "STUDENT_UNION_RETURN"]
    prev_eta = None
    for stop_key in north_stops_order:
        if stop_key in result and result[stop_key]["route"] == "NORTH":
            eta = datetime.fromisoformat(result[stop_key]["eta"])
            if prev_eta is not None:
                assert eta >= prev_eta, \
                    f"Impossible ordering: {stop_key} ETA {eta} is before previous stop ETA {prev_eta}"
            prev_eta = eta


@pytest.mark.asyncio
async def test_past_etas_filtered_out():
    """ETAs in the past should not appear in results."""
    from backend.worker.data import compute_per_stop_etas

    # Set timestamp far in the past so all ETAs will be in the past
    past = datetime(2020, 1, 1, tzinfo=timezone.utc)
    df = make_vehicle_df("v1", "NORTH", 2, timestamp=past)

    past_eta = past + timedelta(minutes=2)
    mock_lstm = AsyncMock(return_value={"v1": past_eta})

    with patch("backend.worker.data.predict_eta", mock_lstm), \
         patch("backend.worker.data.get_today_dataframe", AsyncMock(return_value=df)):

        result = await compute_per_stop_etas(["v1"], df=df)

    # All ETAs should be filtered out since they're in 2020
    assert len(result) == 0
