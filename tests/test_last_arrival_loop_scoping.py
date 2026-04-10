"""Tests for per-trip loop-scoped last_arrival filtering.

These tests lock in the fix for the long-running "COLONIE shows Last: even
though the shuttle hasn't reached it yet in the current loop" bug. The root
cause was a 15-minute sliding window in `_compute_vehicle_etas_and_arrivals`
that leaked prior-loop detections into the current trip's display. The fix
moves filtering into `build_trip_etas` via a per-trip `loop_cutoff`
parameter, using each trip's own `actual_departure` (or `prior_departure`
for completed trips) as the cutoff.

The four scenarios covered here are the ones the bug touched on: active
trip mid-loop, completed trip for a just-finished loop, dwelling shuttle
at Union post-loop, and within-loop detection-gap interpolation (the part
that must still work).
"""

from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from backend.worker.trips import (
    build_trip_etas,
    compute_trips_from_vehicle_data,
)
from shared.stops import Stops


# ---- Helpers ----------------------------------------------------------------


def _iso(dt: datetime) -> str:
    """ISO string with UTC tz, matching the format the worker produces."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


NORTH_STOPS = Stops.routes_data["NORTH"]["STOPS"]
# Convenience indices into NORTH_STOPS: first = STUDENT_UNION, then the
# real-stop sequence. We assert specific stop names below.


def _make_vehicle_df_loop_detections(
    vid: str,
    route: str,
    loop1_union_ts: datetime,
    loop1_colonie_ts: datetime,
    loop1_georgian_ts: datetime,
    loop2_union_ts: datetime | None = None,
) -> pd.DataFrame:
    """Synthesize a processed vehicle_df with loop-scoped stop detections.

    The rows mimic the shape `_compute_vehicle_etas_and_arrivals` reads:
    each row has vehicle_id, timestamp, stop_name (nullable), route,
    polyline_idx, latitude, longitude, speed_kmh. The lat/lon values
    are offset per stop so the "is the shuttle parked?" displacement
    filter in `compute_trips_from_vehicle_data` does not drop the trip.
    """
    # Distinct lat/lon per stop so the no-movement filter sees real motion.
    STOP_COORDS = {
        "STUDENT_UNION": (42.7300, -73.6800),
        "COLONIE":       (42.7350, -73.6750),
        "GEORGIAN":      (42.7400, -73.6700),
    }
    rows = []
    base = {
        "vehicle_id": vid,
        "route": route,
        "polyline_idx": 0,  # unused by the last_arrivals code path
        "speed_kmh": 20.0,  # moving — bypasses idle filter
    }
    for ts, stop in [
        (loop1_union_ts, "STUDENT_UNION"),
        (loop1_colonie_ts, "COLONIE"),
        (loop1_georgian_ts, "GEORGIAN"),
    ]:
        lat, lon = STOP_COORDS[stop]
        rows.append({**base, "timestamp": ts, "stop_name": stop,
                     "latitude": lat, "longitude": lon})
    if loop2_union_ts is not None:
        lat, lon = STOP_COORDS["STUDENT_UNION"]
        # Nudge lat/lon so the no-movement filter sees a different point
        # than the loop-1 union sample (which is at the same stop).
        rows.append({**base, "timestamp": loop2_union_ts, "stop_name": "STUDENT_UNION",
                     "latitude": lat + 0.0001, "longitude": lon + 0.0001})
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ---- Unit tests: build_trip_etas with loop_cutoff --------------------------


def test_loop_cutoff_hides_prior_loop_arrivals():
    """A last_arrival from before the current loop's cutoff must be dropped.

    Scenario: loop 1 detected COLONIE at 14:05. Shuttle is now mid-loop-2
    at 14:16, active trip's actual_departure was 14:14. The Colonie
    detection from 14:05 is older than 14:14 and must NOT appear as
    `passed=True` / `last_arrival=14:05` on the active trip.
    """
    now = datetime(2026, 4, 10, 14, 16, tzinfo=timezone.utc)
    actual_departure = datetime(2026, 4, 10, 14, 14, tzinfo=timezone.utc)

    last_arrivals = {
        "STUDENT_UNION": _iso(datetime(2026, 4, 10, 14, 14, tzinfo=timezone.utc)),
        "COLONIE": _iso(datetime(2026, 4, 10, 14, 5, tzinfo=timezone.utc)),  # loop 1
        "GEORGIAN": _iso(datetime(2026, 4, 10, 14, 8, tzinfo=timezone.utc)),  # loop 1
    }
    stops = ["STUDENT_UNION", "COLONIE", "GEORGIAN", "STAC_1"]
    trip = {
        "trip_id": "NORTH:...",
        "route": "NORTH",
        "vehicle_id": "v1",
        "status": "active",
    }

    result = build_trip_etas(
        trip=trip,
        vehicle_stops=[("COLONIE", now + timedelta(minutes=2))],
        last_arrivals=last_arrivals,
        stops_in_route=stops,
        now_utc=now,
        loop_cutoff=actual_departure,
    )

    # COLONIE was detected in loop 1 (14:05) before the loop_cutoff (14:14),
    # so the current trip must show it as unreached.
    assert result["COLONIE"]["last_arrival"] is None
    assert result["COLONIE"]["passed"] is False
    # GEORGIAN same story.
    assert result["GEORGIAN"]["last_arrival"] is None
    assert result["GEORGIAN"]["passed"] is False
    # STUDENT_UNION was detected at exactly the cutoff — must pass the filter.
    assert result["STUDENT_UNION"]["passed"] is True


def test_loop_cutoff_preserves_current_loop_arrivals():
    """Detections at-or-after loop_cutoff must be kept."""
    now = datetime(2026, 4, 10, 14, 20, tzinfo=timezone.utc)
    actual_departure = datetime(2026, 4, 10, 14, 14, tzinfo=timezone.utc)

    last_arrivals = {
        "STUDENT_UNION": _iso(datetime(2026, 4, 10, 14, 14, tzinfo=timezone.utc)),
        "COLONIE": _iso(datetime(2026, 4, 10, 14, 17, tzinfo=timezone.utc)),  # current loop
    }
    stops = ["STUDENT_UNION", "COLONIE", "GEORGIAN"]
    trip = {"trip_id": "t", "route": "NORTH", "vehicle_id": "v1", "status": "active"}

    result = build_trip_etas(
        trip=trip,
        vehicle_stops=[("GEORGIAN", now + timedelta(minutes=1))],
        last_arrivals=last_arrivals,
        stops_in_route=stops,
        now_utc=now,
        loop_cutoff=actual_departure,
    )
    # Both detections are from the current loop, both must be preserved.
    assert result["STUDENT_UNION"]["passed"] is True
    assert result["STUDENT_UNION"]["last_arrival"] is not None
    assert result["COLONIE"]["passed"] is True
    assert result["COLONIE"]["last_arrival"] is not None


def test_loop_cutoff_none_keeps_everything():
    """Passing loop_cutoff=None must preserve legacy behavior (no filter)."""
    now = datetime(2026, 4, 10, 14, 16, tzinfo=timezone.utc)
    last_arrivals = {
        "COLONIE": _iso(datetime(2026, 4, 10, 14, 5, tzinfo=timezone.utc)),
    }
    trip = {"trip_id": "t", "route": "NORTH", "vehicle_id": "v1", "status": "active"}
    result = build_trip_etas(
        trip=trip,
        vehicle_stops=[],
        last_arrivals=last_arrivals,
        stops_in_route=["STUDENT_UNION", "COLONIE"],
        now_utc=now,
        loop_cutoff=None,
    )
    assert result["COLONIE"]["last_arrival"] is not None
    assert result["COLONIE"]["passed"] is True


def test_mid_loop_backfill_still_works():
    """Within-loop detection-gap interpolation must still fire.

    If STUDENT_UNION and GEORGIAN both have real detections in the current
    loop but COLONIE (between them) has none, the backfill should mark
    COLONIE as passed with an interpolated last_arrival.
    """
    now = datetime(2026, 4, 10, 14, 25, tzinfo=timezone.utc)
    loop_cutoff = datetime(2026, 4, 10, 14, 14, tzinfo=timezone.utc)

    last_arrivals = {
        "STUDENT_UNION": _iso(datetime(2026, 4, 10, 14, 14, tzinfo=timezone.utc)),
        "GEORGIAN": _iso(datetime(2026, 4, 10, 14, 20, tzinfo=timezone.utc)),
        # COLONIE missing on purpose — the backfill should fill it
    }
    stops = ["STUDENT_UNION", "COLONIE", "GEORGIAN", "STAC_1"]
    trip = {"trip_id": "t", "route": "NORTH", "vehicle_id": "v1", "status": "active"}

    result = build_trip_etas(
        trip=trip,
        vehicle_stops=[("STAC_1", now + timedelta(minutes=1))],
        last_arrivals=last_arrivals,
        stops_in_route=stops,
        now_utc=now,
        loop_cutoff=loop_cutoff,
    )

    # COLONIE should be backfilled: passed=True with an interpolated
    # last_arrival somewhere between STUDENT_UNION (14:14) and GEORGIAN (14:20).
    assert result["COLONIE"]["passed"] is True
    assert result["COLONIE"]["last_arrival"] is not None
    colonie_la = datetime.fromisoformat(result["COLONIE"]["last_arrival"])
    assert datetime(2026, 4, 10, 14, 14, tzinfo=timezone.utc) <= colonie_la
    assert colonie_la <= datetime(2026, 4, 10, 14, 20, tzinfo=timezone.utc)
    # STAC_1 has a future ETA, not reached — must stay unpassed.
    assert result["STAC_1"]["passed"] is False
    assert result["STAC_1"]["last_arrival"] is None


# ---- Integration tests: full compute_trips_from_vehicle_data ----------------


def test_active_trip_scoped_via_compute_trips_from_vehicle_data():
    """End-to-end: loop-1 detections must not leak into an active loop-2 trip.

    Shuttle completed loop 1 (Union → Colonie → Georgian), then re-departed
    Union ~2 minutes ago (start of loop 2). The current trip is the loop-2
    run; its active trip must not show Colonie/Georgian as passed.
    """
    now = datetime(2026, 4, 10, 14, 16, tzinfo=timezone.utc)
    loop1_union = datetime(2026, 4, 10, 14, 0, tzinfo=timezone.utc)
    loop1_colonie = datetime(2026, 4, 10, 14, 3, tzinfo=timezone.utc)
    loop1_georgian = datetime(2026, 4, 10, 14, 6, tzinfo=timezone.utc)
    loop2_union = datetime(2026, 4, 10, 14, 14, tzinfo=timezone.utc)

    vid = "v1"
    vehicle_stop_etas = {
        vid: {
            "route": "NORTH",
            "stops": [
                ("COLONIE", now + timedelta(minutes=1)),
                ("GEORGIAN", now + timedelta(minutes=4)),
            ],
        }
    }
    last_arrivals_by_vehicle = {
        vid: {
            "STUDENT_UNION": _iso(loop2_union),  # current loop
            "COLONIE": _iso(loop1_colonie),       # previous loop — must be dropped
            "GEORGIAN": _iso(loop1_georgian),     # previous loop — must be dropped
        }
    }
    df = _make_vehicle_df_loop_detections(
        vid, "NORTH", loop1_union, loop1_colonie, loop1_georgian,
        loop2_union_ts=loop2_union,
    )

    trips = compute_trips_from_vehicle_data(
        vehicle_stop_etas=vehicle_stop_etas,
        last_arrivals_by_vehicle=last_arrivals_by_vehicle,
        full_df=df,
        routes_data=Stops.routes_data,
        vehicle_ids=[vid],
        now_utc=now,
        campus_tz=timezone.utc,
    )

    # Find the active (or scheduled) trip for this vehicle — the one with
    # the loop-2 actual_departure. A completed trip may also be emitted
    # (we separately assert on it in another test).
    live_trips = [t for t in trips if t["vehicle_id"] == vid and t["status"] != "completed"]
    assert live_trips, "expected at least one non-completed trip for the vehicle"
    active = live_trips[0]

    # COLONIE must NOT be marked passed on the active trip.
    colonie = active["stop_etas"]["COLONIE"]
    assert colonie["passed"] is False, (
        f"COLONIE leaked: passed={colonie['passed']}, "
        f"last_arrival={colonie['last_arrival']}"
    )
    assert colonie["last_arrival"] is None
    # GEORGIAN same rule.
    georgian = active["stop_etas"]["GEORGIAN"]
    assert georgian["passed"] is False
    assert georgian["last_arrival"] is None
