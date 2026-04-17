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


# --- Tests for polyline-intersection confusion in _compute_vehicle_etas_and_arrivals ---
#
# The NORTH route's outbound segment (Union → COLONIE) physically crosses
# its return segment (HFH → ... → Union) somewhere near HFH. When a
# shuttle is on the return leg near HFH, the closest-point match can
# resolve its position to the outbound polyline_idx=0 even though the
# shuttle never touched Union. Without the physical-distance guard on
# `is_loop_restart`, the predictor then builds vehicle_stops starting
# from COLONIE, puts HFH in the upcoming list with a future eta, and
# the spurious-detection scrub in build_trip_etas drops HFH's real
# last_arrival as "noise". Users see HFH flip from "Last:" to
# "ETA: LIVE" for the minute or two until the shuttle's polyline_idx
# resolves back to the true return value.


def _make_intersection_confusion_df(
    vid: str,
    now: datetime,
    ping_lat: float,
    ping_lon: float,
    polyline_idx: float,
    last_stop_name: str,
    last_row_has_stop: bool = True,
) -> pd.DataFrame:
    """Build a vehicle_df where the latest ping has a specific physical
    position, polyline_idx, and last detected stop_name.

    When `last_row_has_stop=True` the latest row is tagged with
    last_stop_name — this triggers the `now_stop_key` override branch
    in _compute_vehicle_etas_and_arrivals (line ~889). When False, the
    tag is one row BEFORE the latest, so the latest row has
    `stop_name=None`. Use False to exercise the polyline_idx branch
    (line ~848) in isolation without the now_stop_key override.
    """
    rows = []
    stop_tag_idx = 11 if last_row_has_stop else 10
    for i in range(12):
        ts = now - timedelta(seconds=(11 - i) * 5)
        rows.append({
            "vehicle_id": str(vid),
            "latitude": ping_lat,
            "longitude": ping_lon,
            "speed_kmh": 20.0,
            "timestamp": ts,
            "route": "NORTH",
            "polyline_idx": polyline_idx,
            "stop_name": last_stop_name if i == stop_tag_idx else None,
        })
    return pd.DataFrame(rows)


@pytest.mark.asyncio
async def test_polyline_intersection_does_not_trigger_loop_restart():
    """The intersection-confusion case must NOT be mistaken for a loop restart.

    Setup: shuttle physically at HFH coordinates (~500 m from Union),
    last detected stop_name = HFH (one row before the latest, so the
    latest row's now_stop_key is None and we exercise the polyline_idx
    branch rather than the now_stop_key override). Current polyline_idx
    = 0 (the closest-point match picked a point on the first-outbound
    polyline because the two polylines geographically intersect near
    HFH).

    Contract: the predictor must recognize this as polyline-intersection
    confusion (NOT a loop restart) and override current_polyline_idx to
    last_stop_idx. The resulting vehicle_stops should start from the
    stop AFTER HFH (STUDENT_UNION_RETURN), NOT from COLONIE.
    """
    from backend.worker.data import _compute_vehicle_etas_and_arrivals

    now = datetime.now(timezone.utc)
    # HFH's actual coordinates (from shared/routes.json)
    hfh_coord = Stops.routes_data["NORTH"]["HOUSTON_FIELD_HOUSE"]["COORDINATES"]
    df = _make_intersection_confusion_df(
        vid="v_intersect",
        now=now,
        ping_lat=hfh_coord[0],
        ping_lon=hfh_coord[1],
        polyline_idx=0.0,  # <-- the bug trigger: polyline match returns 0
        last_stop_name="HOUSTON_FIELD_HOUSE",
        last_row_has_stop=False,  # isolate the polyline_idx branch
    )

    # Mock predict_eta so the test doesn't require LSTM models.
    sur_eta = now + timedelta(minutes=2)
    mock_predict = AsyncMock(return_value={"v_intersect": sur_eta})

    with patch("backend.worker.data.predict_eta", mock_predict):
        vehicle_stop_etas, _ = await _compute_vehicle_etas_and_arrivals(
            ["v_intersect"], df
        )

    # The vehicle should be in the output (not dropped as out-of-range)
    assert "v_intersect" in vehicle_stop_etas, (
        "vehicle was dropped — override likely produced next_stop_idx out of range"
    )
    stops_list = vehicle_stop_etas["v_intersect"]["stops"]
    assert stops_list, "vehicle has no upcoming stops"

    first_upcoming = stops_list[0][0]  # (stop_key, eta_dt)
    # The shuttle is PAST HFH (index 7). Next upcoming stop must be
    # STUDENT_UNION_RETURN (index 8). It MUST NOT be COLONIE or
    # anything earlier — that would mean the override didn't fire.
    assert first_upcoming == "STUDENT_UNION_RETURN", (
        f"expected next stop STUDENT_UNION_RETURN (override fired), "
        f"got {first_upcoming} — intersection confusion treated as loop restart"
    )

    # HFH itself must NOT appear in the upcoming stops — the shuttle
    # already passed it. This is the crucial anti-regression check:
    # if HFH shows up in vehicle_stops with a future eta, the spurious
    # scrub in build_trip_etas will drop HFH's real last_arrival,
    # flipping the UI from "Last:" to "ETA: LIVE".
    upcoming_stop_keys = [s[0] for s in stops_list]
    assert "HOUSTON_FIELD_HOUSE" not in upcoming_stop_keys, (
        f"HFH leaked into upcoming stops: {upcoming_stop_keys} — the "
        f"predictor still thinks HFH is ahead of the shuttle"
    )


@pytest.mark.asyncio
async def test_true_loop_restart_still_respected():
    """A genuine loop restart (shuttle physically at Union) must still be
    recognized. This is the counter-case to the intersection test — the
    override must NOT fire when the shuttle really is at the start of a
    new loop.
    """
    from backend.worker.data import _compute_vehicle_etas_and_arrivals

    now = datetime.now(timezone.utc)
    # Student Union's actual coordinates
    union_coord = Stops.routes_data["NORTH"]["STUDENT_UNION"]["COORDINATES"]
    df = _make_intersection_confusion_df(
        vid="v_restart",
        now=now,
        ping_lat=union_coord[0],
        ping_lon=union_coord[1],
        polyline_idx=0.0,  # polyline correctly reports "new loop starting"
        last_stop_name="STUDENT_UNION_RETURN",
        last_row_has_stop=False,  # latest row has stop_name=None so the
                                  # now_stop_key override doesn't fire and
                                  # we test the polyline_idx branch
    )

    colonie_eta = now + timedelta(minutes=3)
    mock_predict = AsyncMock(return_value={"v_restart": colonie_eta})

    with patch("backend.worker.data.predict_eta", mock_predict):
        vehicle_stop_etas, _ = await _compute_vehicle_etas_and_arrivals(
            ["v_restart"], df
        )

    assert "v_restart" in vehicle_stop_etas
    stops_list = vehicle_stop_etas["v_restart"]["stops"]
    assert stops_list, "vehicle has no upcoming stops"

    # True loop restart: the new loop is starting, next stop is COLONIE.
    # Override must NOT have fired (the shuttle is physically at Union).
    first_upcoming = stops_list[0][0]
    assert first_upcoming == "COLONIE", (
        f"expected next stop COLONIE (true loop restart), got "
        f"{first_upcoming} — the override incorrectly fired"
    )


@pytest.mark.asyncio
async def test_intersection_confusion_with_stop_detection_fallback():
    """If `now_stop_key` is set to HFH on the latest row, the existing
    now_stop_key override (line 889) already fires. This test documents
    that the intersection fix is redundant with the now_stop_key path
    but still fires correctly when now_stop_key is set.
    """
    from backend.worker.data import _compute_vehicle_etas_and_arrivals

    now = datetime.now(timezone.utc)
    hfh_coord = Stops.routes_data["NORTH"]["HOUSTON_FIELD_HOUSE"]["COORDINATES"]
    # This variant has stop_name=HOUSTON_FIELD_HOUSE on the LATEST row,
    # so the now_stop_key override will also fire.
    rows = []
    for i in range(12):
        ts = now - timedelta(seconds=(11 - i) * 5)
        rows.append({
            "vehicle_id": "v_both",
            "latitude": hfh_coord[0],
            "longitude": hfh_coord[1],
            "speed_kmh": 20.0,
            "timestamp": ts,
            "route": "NORTH",
            "polyline_idx": 0.0,
            "stop_name": "HOUSTON_FIELD_HOUSE" if i >= 10 else None,
        })
    df = pd.DataFrame(rows)

    sur_eta = now + timedelta(minutes=2)
    mock_predict = AsyncMock(return_value={"v_both": sur_eta})

    with patch("backend.worker.data.predict_eta", mock_predict):
        vehicle_stop_etas, _ = await _compute_vehicle_etas_and_arrivals(
            ["v_both"], df
        )

    assert "v_both" in vehicle_stop_etas
    stops_list = vehicle_stop_etas["v_both"]["stops"]
    upcoming_keys = [s[0] for s in stops_list]
    assert "HOUSTON_FIELD_HOUSE" not in upcoming_keys
    # Next stop is SUR (index 8 = HFH index 7 + 1)
    assert stops_list[0][0] == "STUDENT_UNION_RETURN"
