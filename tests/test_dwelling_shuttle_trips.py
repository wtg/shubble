"""Tests for dwelling-shuttle handling in the per-vehicle ETA pipeline.

Locks in the fix for Bug 1 in quick task 260414-kg2: a shuttle whose
latest ping is tagged STUDENT_UNION_RETURN (the last stop in NORTH/WEST
STOPS) must still produce a `vehicle_stop_etas[vid]` entry. Without
the fix, the stop-detection override set
`next_stop_idx = now_stop_idx + 1 = len(STOPS)` and the vehicle fell
through the `next_stop_idx >= len(stops)` guard, disappearing from the
downstream /api/trips pipeline.

We also cover the same bug in `compute_per_stop_etas` (the sibling
function consumed by the /api/etas writer) with an analogous check.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from backend.worker.data import (
    _compute_vehicle_etas_and_arrivals,
    compute_per_stop_etas,
)
from shared.stops import Stops


NORTH = Stops.routes_data["NORTH"]
NORTH_STOPS = NORTH["STOPS"]
UNION_COORDS = tuple(NORTH[NORTH_STOPS[0]]["COORDINATES"])


def _mk_row(vid, ts, stop_name, lat, lon, route="NORTH", polyline_idx=0):
    return {
        "vehicle_id": vid,
        "timestamp": ts,
        "stop_name": stop_name,
        "latitude": lat,
        "longitude": lon,
        "route": route,
        "polyline_idx": polyline_idx,
        "speed_kmh": 0.0,
    }


def _dwelling_df(vid: str, base: datetime) -> pd.DataFrame:
    polyline_stops = NORTH.get("POLYLINE_STOPS", NORTH_STOPS)
    late_poly_idx = max(0, len(polyline_stops) - 2)
    rows = []
    for i in range(5):
        rows.append(
            _mk_row(
                vid,
                base + timedelta(seconds=i * 5),
                "STUDENT_UNION_RETURN",
                UNION_COORDS[0],
                UNION_COORDS[1],
                polyline_idx=late_poly_idx,
            )
        )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.mark.asyncio
async def test_dwelling_at_union_appears_in_vehicle_stop_etas():
    """A shuttle dwelling at STUDENT_UNION_RETURN on NORTH must produce
    a vehicle_stop_etas[vid] entry whose first `stops` entry is the
    stop at STOPS index 1 (COLONIE), not STUDENT_UNION_RETURN itself.
    """
    vid = "dwell_v1"
    base = datetime(2026, 4, 14, 15, 0, tzinfo=timezone.utc)
    df = _dwelling_df(vid, base)

    mock_eta = base + timedelta(minutes=2)
    mock_predict = AsyncMock(return_value={vid: mock_eta})

    with patch("backend.worker.data.predict_eta", mock_predict):
        vehicle_stop_etas, _ = await _compute_vehicle_etas_and_arrivals([vid], df)

    assert vid in vehicle_stop_etas, (
        f"Dwelling shuttle missing from vehicle_stop_etas. "
        f"Got keys: {list(vehicle_stop_etas.keys())}"
    )
    stops_list = vehicle_stop_etas[vid]["stops"]
    assert len(stops_list) >= 1
    first_stop_key = stops_list[0][0]
    assert first_stop_key == NORTH_STOPS[1], (
        f"Expected first stop to be {NORTH_STOPS[1]} (index 1), got {first_stop_key}"
    )
    assert first_stop_key != NORTH_STOPS[-1], (
        "First stop should not be STUDENT_UNION_RETURN (the last STOPS entry)"
    )


@pytest.mark.asyncio
async def test_dwelling_at_union_in_compute_per_stop_etas():
    """The sibling compute_per_stop_etas path must also surface ETAs for
    a dwelling-at-Union shuttle. The output is keyed by stop_name;
    assert at least the second stop (COLONIE) has a non-null eta
    attributed to the dwelling vehicle.
    """
    vid = "dwell_v2"
    base = datetime.now(timezone.utc)
    df = _dwelling_df(vid, base)

    mock_eta = base + timedelta(minutes=2)
    mock_predict = AsyncMock(return_value={vid: mock_eta})

    with patch("backend.worker.data.predict_eta", mock_predict):
        result = await compute_per_stop_etas([vid], df=df)

    # Expect the second stop (e.g. COLONIE on NORTH) to have an entry
    # whose vehicle_id matches the dwelling shuttle.
    second_stop = NORTH_STOPS[1]
    assert second_stop in result, (
        f"{second_stop} missing from result. Got: {list(result.keys())}"
    )
    entry = result[second_stop]
    # Some test-env paths may produce route-qualified keys only; accept
    # either. The essential property: SOMETHING in the result carries
    # the dwelling vid with a non-null eta.
    has_vid = entry.get("vehicle_id") == vid and entry.get("eta")
    if not has_vid:
        # Fallback: check route-qualified key
        rq_key = f"{second_stop}:NORTH"
        alt = result.get(rq_key)
        has_vid = bool(alt and alt.get("vehicle_id") == vid and alt.get("eta"))
    assert has_vid, (
        f"Dwelling vid {vid} did not produce an eta for {second_stop}. "
        f"Entry: {entry}"
    )


@pytest.mark.asyncio
async def test_multi_stop_lag_does_not_drop_vehicle():
    """Smoke check: a shuttle whose latest detected stop is deep into
    the route (not the last) must still produce upcoming stops. This
    exercises the `now_stop_idx >= next_stop_idx` branch — ensuring
    multi-stop polyline lag still advances next_stop_idx.
    """
    vid = "lag_v1"
    base = datetime(2026, 4, 14, 16, 0, tzinfo=timezone.utc)
    if len(NORTH_STOPS) < 5:
        pytest.skip("NORTH route too short for multi-stop lag scenario")
    late_stop = NORTH_STOPS[3]
    late_stop_coords = tuple(NORTH[late_stop]["COORDINATES"])

    rows = []
    for i in range(5):
        rows.append(
            _mk_row(
                vid,
                base + timedelta(seconds=i * 5),
                late_stop,
                late_stop_coords[0],
                late_stop_coords[1],
                polyline_idx=0,
            )
        )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    mock_eta = base + timedelta(minutes=2)
    mock_predict = AsyncMock(return_value={vid: mock_eta})

    with patch("backend.worker.data.predict_eta", mock_predict):
        vehicle_stop_etas, _ = await _compute_vehicle_etas_and_arrivals([vid], df)

    assert vid in vehicle_stop_etas, "Vehicle dropped unexpectedly"
    stops_list = vehicle_stop_etas[vid]["stops"]
    first_stop_key = stops_list[0][0]
    first_idx = NORTH_STOPS.index(first_stop_key)
    assert first_idx >= 4, (
        f"Expected first upcoming idx >= 4 (after {late_stop}), got {first_idx} ({first_stop_key})"
    )
