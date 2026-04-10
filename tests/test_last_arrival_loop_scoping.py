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


def test_spurious_detection_at_downstream_stop_rejected():
    """Detections at stops the predictor thinks are still upcoming must
    be rejected as spurious.

    Reproduces the bug where the WEST route's active trip showed
    passed=True + last_arrival + a simultaneous future eta on every
    stop after POLYTECHNIC. Root cause: the WEST polyline wraps back
    to Student Union, and add_stops occasionally mis-attributes a GPS
    ping near Union to STUDENT_UNION_RETURN. That post-departure
    "detection" passes the loop_cutoff filter, becomes the farthest
    anchor, and the monotonic clamp copies the earlier stops' real
    timestamps forward onto the unreached downstream stops.

    The scrub is: if the predictor has a FUTURE eta for a stop, any
    "last_arrival" for that same stop is physically impossible and
    must be dropped before the backfill runs.
    """
    # Shuttle physically at POLYTECHNIC (just arrived 15:56). Current
    # time is 15:58. actual_departure was 15:39.
    now = datetime(2026, 4, 10, 15, 58, tzinfo=timezone.utc)
    actual_departure = datetime(2026, 4, 10, 15, 39, tzinfo=timezone.utc)

    stops = [
        "STUDENT_UNION",
        "ACADEMY_HALL",
        "POLYTECHNIC",
        "CITY_STATION",
        "BLITMAN",
        "STUDENT_UNION_RETURN",
    ]
    last_arrivals = {
        "STUDENT_UNION": _iso(datetime(2026, 4, 10, 15, 39, tzinfo=timezone.utc)),
        "ACADEMY_HALL": _iso(datetime(2026, 4, 10, 15, 55, tzinfo=timezone.utc)),
        "POLYTECHNIC": _iso(datetime(2026, 4, 10, 15, 56, tzinfo=timezone.utc)),
        # SPURIOUS: polyline jitter at Union mis-attributed to STUDENT_UNION_RETURN.
        # Must be dropped because the predictor has a future eta for it.
        "STUDENT_UNION_RETURN": _iso(datetime(2026, 4, 10, 15, 56, tzinfo=timezone.utc)),
    }
    # Predictor's view: shuttle is upcoming at CITY_STATION, BLITMAN,
    # and STUDENT_UNION_RETURN (all future ETAs).
    vehicle_stops = [
        ("CITY_STATION", now + timedelta(minutes=1)),
        ("BLITMAN", now + timedelta(minutes=2)),
        ("STUDENT_UNION_RETURN", now + timedelta(minutes=8)),
    ]
    trip = {"trip_id": "WEST:t", "route": "WEST", "vehicle_id": "v4", "status": "active"}

    result = build_trip_etas(
        trip=trip,
        vehicle_stops=vehicle_stops,
        last_arrivals=last_arrivals,
        stops_in_route=stops,
        now_utc=now,
        loop_cutoff=actual_departure,
    )

    # STUDENT_UNION_RETURN has a future eta → detection is spurious,
    # must be dropped. The stop is not yet reached.
    surn = result["STUDENT_UNION_RETURN"]
    assert surn["passed"] is False, (
        f"STUDENT_UNION_RETURN should be unreached, got {surn}"
    )
    assert surn["last_arrival"] is None
    assert surn["eta"] is not None, "future eta must be preserved"

    # CITY_STATION and BLITMAN had no la in input — stay unreached with
    # their future ETAs. (Regression check — they must not acquire a
    # backfilled la from the now-absent spurious anchor.)
    for upcoming_stop in ("CITY_STATION", "BLITMAN"):
        entry = result[upcoming_stop]
        assert entry["passed"] is False, f"{upcoming_stop}: {entry}"
        assert entry["last_arrival"] is None, f"{upcoming_stop}: {entry}"
        assert entry["eta"] is not None, f"{upcoming_stop}: {entry}"

    # The real detections must still stand.
    assert result["STUDENT_UNION"]["passed"] is True
    assert result["ACADEMY_HALL"]["passed"] is True
    assert result["POLYTECHNIC"]["passed"] is True

    # Invariant: no stop may have both passed=True and a future eta.
    for stop_key, entry in result.items():
        if entry["passed"]:
            eta_iso = entry["eta"]
            if eta_iso is not None:
                eta_dt = datetime.fromisoformat(eta_iso)
                assert eta_dt <= now, (
                    f"{stop_key} is passed=True but eta {eta_iso} is in the future"
                )


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
    # And it must be flagged as interpolated so the frontend knows not
    # to render the synthesized timestamp as a real arrival.
    assert result["COLONIE"]["passed_interpolated"] is True
    colonie_la = datetime.fromisoformat(result["COLONIE"]["last_arrival"])
    assert datetime(2026, 4, 10, 14, 14, tzinfo=timezone.utc) <= colonie_la
    assert colonie_la <= datetime(2026, 4, 10, 14, 20, tzinfo=timezone.utc)
    # Real-detection stops must NOT be flagged interpolated.
    assert result["STUDENT_UNION"]["passed_interpolated"] is False
    assert result["GEORGIAN"]["passed_interpolated"] is False
    # STAC_1 has a future ETA, not reached — must stay unpassed.
    assert result["STAC_1"]["passed"] is False
    assert result["STAC_1"]["last_arrival"] is None
    # Unreached stops default to passed_interpolated=False.
    assert result["STAC_1"]["passed_interpolated"] is False


def test_spurious_scrub_allows_tick_boundary_race():
    """The spurious-detection scrub must not drop a real detection just
    because the predictor's polyline_idx hasn't advanced past it yet.

    Scenario: shuttle physically arrives at STAC_1 (Bryckwyck) at 14:22.
    add_stops registers the detection. The same worker cycle runs the
    predictor, which has STAC_1 in vehicle_stops with an eta a few
    seconds in the future (polyline projection lags the raw stop
    match). Without a grace window, the scrub would drop STAC_1 from
    last_arrivals as "spurious", which in turn collapses
    farthest_detected_idx and leaves GEORGIAN (the earlier stop) stuck
    with a future ETA for one worker cycle. Users see the stop flip
    back to "live ETA" briefly after passing, or see Georgian's update
    lag behind Bryckwyck's.

    Contract: an upcoming eta within the SPURIOUS_UPCOMING_GRACE_SEC
    window (60 s) is treated as a tick-boundary race, not noise, and
    the detection is preserved.
    """
    now = datetime(2026, 4, 10, 14, 22, 5, tzinfo=timezone.utc)
    loop_cutoff = datetime(2026, 4, 10, 14, 14, tzinfo=timezone.utc)

    # Shuttle just arrived at STAC_1 at 14:22:00 — 5s before "now".
    last_arrivals = {
        "STUDENT_UNION": _iso(datetime(2026, 4, 10, 14, 14, tzinfo=timezone.utc)),
        "COLONIE": _iso(datetime(2026, 4, 10, 14, 18, tzinfo=timezone.utc)),
        "STAC_1": _iso(datetime(2026, 4, 10, 14, 22, tzinfo=timezone.utc)),
    }
    # Predictor has STAC_1 still listed as "upcoming" with an eta ~15s
    # in the future — tick-boundary lag (polyline_idx not yet advanced
    # past STAC_1). This is a TRUE real detection, not noise.
    stops = ["STUDENT_UNION", "COLONIE", "GEORGIAN", "STAC_1", "STAC_2"]
    trip = {"trip_id": "t", "route": "NORTH", "vehicle_id": "v1", "status": "active"}

    result = build_trip_etas(
        trip=trip,
        vehicle_stops=[
            ("STAC_1", now + timedelta(seconds=15)),  # tick-boundary race
            ("STAC_2", now + timedelta(seconds=75)),
        ],
        last_arrivals=last_arrivals,
        stops_in_route=stops,
        now_utc=now,
        loop_cutoff=loop_cutoff,
    )

    # STAC_1 must be kept as passed despite the future eta from the
    # lagging predictor — the detection is real.
    assert result["STAC_1"]["passed"] is True, (
        f"tick-boundary race dropped: {result['STAC_1']}"
    )
    # And GEORGIAN (earlier, no real detection) must be backfilled as
    # passed via interpolation, not stuck with its upcoming ETA.
    assert result["GEORGIAN"]["passed"] is True, (
        f"Georgian not backfilled: {result['GEORGIAN']}"
    )
    assert result["GEORGIAN"]["passed_interpolated"] is True
    # STAC_2 is still truly upcoming (eta is 75s away, well past the
    # 60 s grace window — if it had been in last_arrivals, the scrub
    # should drop it because the predictor's "upcoming" signal is
    # clearer there. But we didn't put it in last_arrivals, so just
    # check that it stays unreached.)
    assert result["STAC_2"]["passed"] is False
    assert result["STAC_2"]["eta"] is not None


def test_spurious_scrub_still_kills_far_future_noise():
    """The scrub must still drop detections whose eta is >60 s future.

    This is the original WEST bug: a GPS ping near STUDENT_UNION got
    mis-attributed to STUDENT_UNION_RETURN (different stop, same
    physical location) with an "upcoming eta" several minutes away.
    That's noise and must be dropped.
    """
    now = datetime(2026, 4, 10, 15, 58, tzinfo=timezone.utc)
    actual_departure = datetime(2026, 4, 10, 15, 39, tzinfo=timezone.utc)

    stops = [
        "STUDENT_UNION",
        "ACADEMY_HALL",
        "POLYTECHNIC",
        "CITY_STATION",
        "STUDENT_UNION_RETURN",
    ]
    last_arrivals = {
        "STUDENT_UNION": _iso(datetime(2026, 4, 10, 15, 39, tzinfo=timezone.utc)),
        "POLYTECHNIC": _iso(datetime(2026, 4, 10, 15, 56, tzinfo=timezone.utc)),
        # Spurious: physical Union position mis-attributed to the loop-end stop
        "STUDENT_UNION_RETURN": _iso(datetime(2026, 4, 10, 15, 56, tzinfo=timezone.utc)),
    }
    vehicle_stops = [
        ("CITY_STATION", now + timedelta(minutes=1)),
        ("STUDENT_UNION_RETURN", now + timedelta(minutes=8)),  # 8 minutes out
    ]
    trip = {"trip_id": "t", "route": "WEST", "vehicle_id": "v4", "status": "active"}

    result = build_trip_etas(
        trip=trip,
        vehicle_stops=vehicle_stops,
        last_arrivals=last_arrivals,
        stops_in_route=stops,
        now_utc=now,
        loop_cutoff=actual_departure,
    )

    # STUDENT_UNION_RETURN must still be rejected — its eta is 8 min
    # out, well past the 60 s grace window.
    assert result["STUDENT_UNION_RETURN"]["passed"] is False
    assert result["STUDENT_UNION_RETURN"]["last_arrival"] is None
    # And the invariant must hold everywhere.
    for stop_key, entry in result.items():
        if entry["passed"] and entry["eta"]:
            eta_dt = datetime.fromisoformat(entry["eta"])
            assert eta_dt <= now, (
                f"{stop_key} is passed but eta is in the future: {entry}"
            )


def test_passed_interpolated_default_is_false_for_real_detections():
    """Every real-detection entry must have passed_interpolated=False.

    Defends against a regression where the flag is accidentally set
    on an anchor point because the anchor-clamp loop reuses `entry`
    from a previous iteration.
    """
    now = datetime(2026, 4, 10, 14, 20, tzinfo=timezone.utc)
    loop_cutoff = datetime(2026, 4, 10, 14, 14, tzinfo=timezone.utc)
    last_arrivals = {
        "STUDENT_UNION": _iso(datetime(2026, 4, 10, 14, 14, tzinfo=timezone.utc)),
        "COLONIE": _iso(datetime(2026, 4, 10, 14, 17, tzinfo=timezone.utc)),
        "GEORGIAN": _iso(datetime(2026, 4, 10, 14, 19, tzinfo=timezone.utc)),
    }
    stops = ["STUDENT_UNION", "COLONIE", "GEORGIAN", "STAC_1"]
    trip = {"trip_id": "t", "route": "NORTH", "vehicle_id": "v1", "status": "active"}

    result = build_trip_etas(
        trip=trip,
        vehicle_stops=[("STAC_1", now + timedelta(minutes=2))],
        last_arrivals=last_arrivals,
        stops_in_route=stops,
        now_utc=now,
        loop_cutoff=loop_cutoff,
    )

    # Every stop with a real detection has passed_interpolated=False.
    for passed_stop in ("STUDENT_UNION", "COLONIE", "GEORGIAN"):
        entry = result[passed_stop]
        assert entry["passed"] is True, f"{passed_stop} should be passed"
        assert entry["passed_interpolated"] is False, (
            f"{passed_stop} has a real detection and must not be "
            f"flagged interpolated: {entry}"
        )
    # Upcoming stop defaults to False too.
    assert result["STAC_1"]["passed_interpolated"] is False


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
