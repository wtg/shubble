"""Regression tests for the LATEST-close-approach dedup in
`_compute_vehicle_etas_and_arrivals`.

Locks in the fix for quick task 260414-m1p: the dedup that picks a
single row per (vehicle_id, stop_name) from the within-60m close_rows
must pick the LATEST timestamp, not the GLOBALLY-CLOSEST distance.

Why: every close_rows row has already passed the 60 m CLOSE_APPROACH_M
gate and is therefore a genuine arrival. Picking by min(_dist_m) over
a multi-loop day silently preferred whichever prior loop's ping
happened to land geometrically closest, then the downstream
`loop_cutoff` filter in `build_trip_etas` dropped that old timestamp
as a stale prior-loop leak. Net effect: stops the shuttle passes
every loop (Houston Field House on NORTH, the WEST-route stops on
return) never surfaced a "Last:" on the active trip even moments
after the shuttle passed them.

Picking by max(timestamp) instead returns the current loop's
detection by construction, which then passes `loop_cutoff` cleanly.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from backend.worker.data import _compute_vehicle_etas_and_arrivals
from shared.stops import Stops


NORTH = Stops.routes_data["NORTH"]
HFH_COORDS = tuple(NORTH["HOUSTON_FIELD_HOUSE"]["COORDINATES"])  # (lat, lon)


def _mk_row(
    vid: str,
    ts: datetime,
    stop_name: str,
    lat: float,
    lon: float,
    route: str = "NORTH",
    polyline_idx: int = 0,
    speed_kmh: float = 20.0,
) -> dict:
    return {
        "vehicle_id": vid,
        "timestamp": ts,
        "stop_name": stop_name,
        "latitude": lat,
        "longitude": lon,
        "route": route,
        "polyline_idx": polyline_idx,
        "speed_kmh": speed_kmh,
    }


def _offset_coord(base_coord: tuple[float, float], meters_north: float) -> tuple[float, float]:
    """Offset a (lat, lon) coord by roughly `meters_north` meters north.

    ~1 deg latitude == 111_139 m. Close enough for test geometry.
    """
    lat, lon = base_coord
    return (lat + meters_north / 111_139.0, lon)


@pytest.mark.asyncio
async def test_latest_close_approach_wins_over_closest():
    """Two pings for (v1, HFH) within 60 m but at different timestamps:
    the LATER one must win the dedup, even if the earlier one is
    geometrically closer to the stop.

    Mirrors the live-env evidence that motivated the fix:
      18:15:04 — 2 m from HFH  (earlier loop, silently preferred by old code)
      19:48:08 — 18 m from HFH (current loop, dropped by old code)
    """
    vid = "v1"
    earlier_ts = datetime(2026, 4, 10, 18, 15, 0, tzinfo=timezone.utc)
    later_ts = datetime(2026, 4, 10, 19, 48, 0, tzinfo=timezone.utc)

    # Earlier ping lands ~2 m from HFH.
    earlier_lat, earlier_lon = _offset_coord(HFH_COORDS, meters_north=2.0)
    # Later ping lands ~18 m from HFH — still inside the 60 m gate, but
    # farther than the earlier one.
    later_lat, later_lon = _offset_coord(HFH_COORDS, meters_north=18.0)

    # Include a third row at a different stop so the groupby has more
    # than one group to iterate.
    other_stop_coords = tuple(NORTH["COLONIE"]["COORDINATES"])
    other_lat, other_lon = _offset_coord(other_stop_coords, meters_north=5.0)

    rows = [
        _mk_row(vid, earlier_ts, "HOUSTON_FIELD_HOUSE", earlier_lat, earlier_lon),
        _mk_row(vid, later_ts, "HOUSTON_FIELD_HOUSE", later_lat, later_lon),
        _mk_row(vid, later_ts - timedelta(minutes=1), "COLONIE", other_lat, other_lon),
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    mock_predict = AsyncMock(return_value={vid: later_ts + timedelta(minutes=2)})
    with patch("backend.worker.data.predict_eta", mock_predict):
        _, last_arrivals_by_vehicle = await _compute_vehicle_etas_and_arrivals([vid], df)

    assert vid in last_arrivals_by_vehicle, (
        f"Vehicle missing from last_arrivals_by_vehicle: "
        f"{list(last_arrivals_by_vehicle.keys())}"
    )
    per_stop = last_arrivals_by_vehicle[vid]
    assert "HOUSTON_FIELD_HOUSE" in per_stop, (
        f"HFH missing from last_arrivals: {per_stop}"
    )
    got = datetime.fromisoformat(per_stop["HOUSTON_FIELD_HOUSE"])
    assert got == later_ts, (
        f"Expected latest ping {later_ts.isoformat()} to win dedup, "
        f"got {got.isoformat()} (earlier-but-closer would be "
        f"{earlier_ts.isoformat()})"
    )


@pytest.mark.asyncio
async def test_both_pings_out_of_range_yields_no_last_arrival():
    """When every ping for (vehicle, stop) is > 60 m from the stop's real
    coord, the CLOSE_APPROACH_M gate filters them out and the stop has
    no last_arrival entry. The gate still works independently of the
    LATEST-vs-CLOSEST tiebreaker change.
    """
    vid = "v2"
    ts1 = datetime(2026, 4, 10, 18, 15, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 4, 10, 19, 48, 0, tzinfo=timezone.utc)

    # Both pings ~150 m from HFH — well outside the 60 m gate.
    far_lat1, far_lon1 = _offset_coord(HFH_COORDS, meters_north=150.0)
    far_lat2, far_lon2 = _offset_coord(HFH_COORDS, meters_north=180.0)

    rows = [
        _mk_row(vid, ts1, "HOUSTON_FIELD_HOUSE", far_lat1, far_lon1),
        _mk_row(vid, ts2, "HOUSTON_FIELD_HOUSE", far_lat2, far_lon2),
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    mock_predict = AsyncMock(return_value={vid: ts2 + timedelta(minutes=2)})
    with patch("backend.worker.data.predict_eta", mock_predict):
        _, last_arrivals_by_vehicle = await _compute_vehicle_etas_and_arrivals([vid], df)

    # Either the vehicle is absent entirely, or its per-stop dict lacks HFH.
    per_stop = last_arrivals_by_vehicle.get(vid, {})
    assert "HOUSTON_FIELD_HOUSE" not in per_stop, (
        f"Both pings > 60 m — HFH should not appear in last_arrivals. "
        f"Got: {per_stop}"
    )


@pytest.mark.asyncio
async def test_single_within_range_ping_still_surfaces():
    """Smoke check: the trivial single-ping case still produces a
    last_arrival whose timestamp matches the input row. Ensures the
    refactor from keep='first' to keep='last' doesn't break the degenerate
    case where drop_duplicates has only one row to keep.
    """
    vid = "v3"
    ts = datetime(2026, 4, 10, 19, 48, 0, tzinfo=timezone.utc)

    close_lat, close_lon = _offset_coord(HFH_COORDS, meters_north=10.0)
    rows = [
        _mk_row(vid, ts, "HOUSTON_FIELD_HOUSE", close_lat, close_lon),
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    mock_predict = AsyncMock(return_value={vid: ts + timedelta(minutes=2)})
    with patch("backend.worker.data.predict_eta", mock_predict):
        _, last_arrivals_by_vehicle = await _compute_vehicle_etas_and_arrivals([vid], df)

    per_stop = last_arrivals_by_vehicle.get(vid, {})
    assert "HOUSTON_FIELD_HOUSE" in per_stop, (
        f"Single within-60m ping must surface; got per_stop={per_stop}"
    )
    got = datetime.fromisoformat(per_stop["HOUSTON_FIELD_HOUSE"])
    assert got == ts, (
        f"Expected last_arrival == input ts {ts.isoformat()}, got {got.isoformat()}"
    )
