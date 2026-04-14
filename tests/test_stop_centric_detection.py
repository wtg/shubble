"""Regression tests for the stop-centric close-approach pass in
`_compute_vehicle_etas_and_arrivals`.

Locks in the fix for quick task 260414-mxq: untagged GPS pings within
60 m of a stop's coordinate must produce a `last_arrival` for that
stop, even when the ML pipeline failed to assign a `stop_name` to the
row. The pre-existing tag-based pass (`stops_df = full_df.dropna(
subset=['stop_name', ...])`) drops every untagged ping — including all
of HOUSTON_FIELD_HOUSE's fast drive-by pings, which the 20 m
`add_stops` threshold and the incremental-cache filtering combine to
strip. Net effect before the fix: HFH never displayed a real
`last_arrival` in the UI, only the interpolated one.

The new stop-centric pass ignores the `stop_name` column and uses pure
geometry: for each route stop, find every ping within 60 m and keep
the LATEST timestamp per (vehicle, stop). It MUST skip
duplicate-coordinate stop pairs (e.g. STUDENT_UNION /
STUDENT_UNION_RETURN on NORTH and WEST) because cross-tagging both
names at the same physical location silently breaks trip-completion
logic — the existing tag-based + `resolve_duplicate_stops` path uses
polyline_idx context to disambiguate those, and remains the sole
source for them.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from backend.worker.data import _compute_vehicle_etas_and_arrivals
from shared.stops import Stops


NORTH = Stops.routes_data["NORTH"]
HFH_COORDS = tuple(NORTH["HOUSTON_FIELD_HOUSE"]["COORDINATES"])  # (lat, lon)
SU_COORDS = tuple(NORTH["STUDENT_UNION"]["COORDINATES"])
GEORGIAN_COORDS = tuple(NORTH["GEORGIAN"]["COORDINATES"])


def _mk_row(
    vid: str,
    ts: datetime,
    stop_name: str | None,
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
async def test_hfh_drive_by_with_untagged_pings_surfaces_last_arrival():
    """HFH drive-by repro: 3 untagged pings within 60 m of HFH must
    produce a last_arrival entry whose timestamp matches the LATEST
    within-60 m ping.

    Mirrors the live-env failure mode: the test shuttle moves at
    ~20 mph (~44 m per 5 s tick), routinely overshooting the 20 m ML
    `add_stops` threshold but landing inside the 60 m
    `CLOSE_APPROACH_M` gate. Before the stop-centric pass these pings
    were dropped by the tag-based filter (`dropna(subset=['stop_name',
    ...])`) and HFH never surfaced a real `last_arrival`.
    """
    vid = "vid1"
    t = datetime(2026, 4, 14, 18, 0, 0, tzinfo=timezone.utc)

    # Three untagged pings: ~29 m / ~40 m / ~10 m from HFH at t, t+5s, t+10s.
    lat1, lon1 = _offset_coord(HFH_COORDS, meters_north=29.0)
    lat2, lon2 = _offset_coord(HFH_COORDS, meters_north=40.0)
    lat3, lon3 = _offset_coord(HFH_COORDS, meters_north=10.0)

    rows = [
        _mk_row(vid, t, None, lat1, lon1),
        _mk_row(vid, t + timedelta(seconds=5), None, lat2, lon2),
        _mk_row(vid, t + timedelta(seconds=10), None, lat3, lon3),
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    mock_predict = AsyncMock(return_value={vid: t + timedelta(minutes=2)})
    with patch("backend.worker.data.predict_eta", mock_predict):
        _, last_arrivals_by_vehicle = await _compute_vehicle_etas_and_arrivals([vid], df)

    assert vid in last_arrivals_by_vehicle, (
        f"Vehicle missing from last_arrivals_by_vehicle: "
        f"{list(last_arrivals_by_vehicle.keys())}"
    )
    per_stop = last_arrivals_by_vehicle[vid]
    assert "HOUSTON_FIELD_HOUSE" in per_stop, (
        f"HFH missing from last_arrivals — stop-centric pass did not "
        f"detect the untagged drive-by pings. Got: {per_stop}"
    )
    expected_ts = t + timedelta(seconds=10)
    got = datetime.fromisoformat(per_stop["HOUSTON_FIELD_HOUSE"])
    assert got == expected_ts, (
        f"Expected LATEST within-60 m ping {expected_ts.isoformat()} "
        f"to win, got {got.isoformat()}"
    )


@pytest.mark.asyncio
async def test_duplicate_coord_stops_not_cross_tagged_by_stop_centric_pass():
    """STUDENT_UNION and STUDENT_UNION_RETURN share a coordinate on
    NORTH. The stop-centric pass MUST skip both members of that pair
    so it doesn't cross-tag a tagged STUDENT_UNION ping as also being
    a STUDENT_UNION_RETURN arrival (which would silently break
    trip-completion logic that treats the `_RETURN` detection as the
    loop end).

    Setup: ONE row tagged `stop_name='STUDENT_UNION'` at the
    Union coords. The tag-based pass should fire for STUDENT_UNION
    (the tag is right). If the stop-centric pass also ran for
    STUDENT_UNION_RETURN it would also tag this same ping (coords are
    identical) — the absence of STUDENT_UNION_RETURN in the result
    proves the duplicate-coord skip is working.
    """
    vid = "vid2"
    t = datetime(2026, 4, 14, 18, 30, 0, tzinfo=timezone.utc)

    # Single ping ~5 m from STUDENT_UNION coords, tagged with the tag-based
    # path's expected stop_name.
    lat, lon = _offset_coord(SU_COORDS, meters_north=5.0)
    rows = [
        _mk_row(vid, t, "STUDENT_UNION", lat, lon),
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    mock_predict = AsyncMock(return_value={vid: t + timedelta(minutes=2)})
    with patch("backend.worker.data.predict_eta", mock_predict):
        _, last_arrivals_by_vehicle = await _compute_vehicle_etas_and_arrivals([vid], df)

    per_stop = last_arrivals_by_vehicle.get(vid, {})
    assert "STUDENT_UNION" in per_stop, (
        f"Tag-based path failed to detect STUDENT_UNION: {per_stop}"
    )
    assert "STUDENT_UNION_RETURN" not in per_stop, (
        f"Stop-centric pass cross-tagged STUDENT_UNION_RETURN at "
        f"identical coords — duplicate-coord skip is broken. "
        f"Got: {per_stop}"
    )


@pytest.mark.asyncio
async def test_tag_based_detection_still_fires_when_stop_centric_pass_skips():
    """Existing tag-based detection must still surface for non-duplicate
    stops where both passes converge on the same answer.

    GEORGIAN is NOT a duplicate-coord stop on NORTH, so both the
    tag-based pass (the row is tagged with `stop_name='GEORGIAN'`) AND
    the stop-centric pass (within 60 m of GEORGIAN coords) detect it.
    The merge semantics must yield a single GEORGIAN entry with the
    row's timestamp — neither overwrites with a stale value, neither
    drops the detection.
    """
    vid = "vid3"
    t = datetime(2026, 4, 14, 19, 0, 0, tzinfo=timezone.utc)

    # Single ping within 20 m of GEORGIAN, tagged correctly.
    lat, lon = _offset_coord(GEORGIAN_COORDS, meters_north=12.0)
    rows = [
        _mk_row(vid, t, "GEORGIAN", lat, lon),
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    mock_predict = AsyncMock(return_value={vid: t + timedelta(minutes=2)})
    with patch("backend.worker.data.predict_eta", mock_predict):
        _, last_arrivals_by_vehicle = await _compute_vehicle_etas_and_arrivals([vid], df)

    per_stop = last_arrivals_by_vehicle.get(vid, {})
    assert "GEORGIAN" in per_stop, (
        f"GEORGIAN missing from last_arrivals — tag-based + stop-centric "
        f"merge dropped a real detection. Got: {per_stop}"
    )
    got = datetime.fromisoformat(per_stop["GEORGIAN"])
    assert got == t, (
        f"Expected GEORGIAN last_arrival == input ts {t.isoformat()}, "
        f"got {got.isoformat()}"
    )
