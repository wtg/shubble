"""Tests for the DWELL-LEAK GUARD in build_trip_etas.

Locks in the fix for quick task 260415-0vt. After the latest-close-approach
(260414-m1p) and stop-centric detection (260414-mxq) fixes shipped, a new
failure mode surfaced: when a shuttle dwells at STUDENT_UNION_RETURN at the
loop boundary and the dwell produces a GPS ping AT or just after the new
loop's actual_departure, the SUR last_arrival passes the loop_cutoff
filter (la >= actual_departure). The downstream backfill then takes that
SUR la as farthest_detected_idx and stamps every earlier stop as
passed_interpolated=True -- COLONIE / GEORGIAN / HFH all show "Passed" on
a loop that just started.

The guard: drop last_arrivals[stops_in_route[-1]] if its gap to
loop_cutoff is less than MIN_LOOP_TIME_SEC (5 minutes). A real loop
traversal takes 15+ minutes; anything faster is a dwell leak.
"""

from datetime import datetime, timezone, timedelta

from backend.worker.trips import build_trip_etas


# ---- Helpers ----------------------------------------------------------------


def _iso(dt: datetime) -> str:
    """ISO string with UTC tz, matching the format the worker produces."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


# Sample timestamps mirror the live evidence noted in 260415-0vt: NORTH
# vid=001 dep=04:33 actual=04:33, SUR la=04:34:16. We use a clean T0 of
# 04:33 UTC and offset from there.
T0 = datetime(2026, 4, 15, 4, 33, tzinfo=timezone.utc)

# Minimal valid NORTH-shaped stops list. The guard only requires that the
# boundary stop (here STUDENT_UNION_RETURN) is the LAST entry; the rest
# of the route shape is irrelevant to the guard.
NORTH_STOPS = ["STUDENT_UNION", "COLONIE", "GEORGIAN", "HFH", "STUDENT_UNION_RETURN"]

TRIP = {
    "trip_id": "NORTH:0433",
    "route": "NORTH",
    "vehicle_id": "v001",
    "status": "active",
}


# ---- Tests ------------------------------------------------------------------


def test_dwell_leak_dropped():
    """A SUR la dated 30 s after actual_departure is a dwell leak.

    Reproduces the live failure: SUR la=04:34:16 (~76 s after the 04:33
    departure cutoff). Without the guard, farthest_detected_idx becomes
    len(stops)-1 (SUR), and the backfill marks COLONIE / GEORGIAN /
    HFH as passed_interpolated=True even though the shuttle has only
    just left STUDENT_UNION.

    Contract: the guard drops the SUR la before the backfill runs, so
    every earlier stop stays unreached and the SUR row shows as
    passed=False with last_arrival=None.
    """
    last_arrivals = {
        "STUDENT_UNION_RETURN": _iso(T0 + timedelta(seconds=30)),
    }
    vehicle_stops = [
        ("COLONIE", T0 + timedelta(minutes=5)),
        ("GEORGIAN", T0 + timedelta(minutes=8)),
    ]

    result = build_trip_etas(
        trip=TRIP,
        vehicle_stops=vehicle_stops,
        last_arrivals=last_arrivals,
        stops_in_route=NORTH_STOPS,
        now_utc=T0 + timedelta(minutes=1),
        loop_cutoff=T0,
    )

    # SUR la must be dropped — it can't be a real completion 30s after dep.
    sur = result["STUDENT_UNION_RETURN"]
    assert sur["passed"] is False, f"SUR should not be marked passed: {sur}"
    assert sur["last_arrival"] is None, f"SUR la should be cleared: {sur}"

    # And, critically, COLONIE / GEORGIAN / HFH must NOT be backfilled
    # as passed/interpolated — without the guard, farthest_detected_idx
    # would anchor on SUR and stamp every earlier stop.
    colonie = result["COLONIE"]
    assert colonie["passed"] is False, f"COLONIE leaked: {colonie}"
    assert colonie["passed_interpolated"] is False, (
        f"COLONIE should not be interpolated: {colonie}"
    )

    georgian = result["GEORGIAN"]
    assert georgian["passed"] is False, f"GEORGIAN leaked: {georgian}"

    hfh = result["HFH"]
    assert hfh["passed"] is False, f"HFH leaked: {hfh}"


def test_legitimate_loop_completion_preserved():
    """A real loop completion (SUR la 18 min after dep) is preserved.

    All four real detections must survive the guard untouched: each
    stop has passed=True with a real (non-None) last_arrival and
    passed_interpolated=False (because each is a real detection).

    Confirms the 18-minute gap is well beyond MIN_LOOP_TIME_SEC, so the
    guard doesn't accidentally clobber legitimate completions.
    """
    last_arrivals = {
        "STUDENT_UNION":        _iso(T0 + timedelta(seconds=30)),
        "COLONIE":              _iso(T0 + timedelta(minutes=2)),
        "GEORGIAN":             _iso(T0 + timedelta(minutes=3)),
        "STUDENT_UNION_RETURN": _iso(T0 + timedelta(minutes=18)),
    }

    result = build_trip_etas(
        trip=TRIP,
        vehicle_stops=[],  # completed loop — no upcoming stops
        last_arrivals=last_arrivals,
        stops_in_route=NORTH_STOPS,
        now_utc=T0 + timedelta(minutes=19),
        loop_cutoff=T0,
    )

    for stop in ("STUDENT_UNION", "COLONIE", "GEORGIAN", "STUDENT_UNION_RETURN"):
        entry = result[stop]
        assert entry["passed"] is True, f"{stop} should be passed: {entry}"
        assert entry["last_arrival"] is not None, (
            f"{stop} la should be preserved: {entry}"
        )
        assert entry["passed_interpolated"] is False, (
            f"{stop} has a real detection — must not be interpolated: {entry}"
        )


def test_at_threshold_boundary():
    """SUR la at EXACTLY MIN_LOOP_TIME_SEC after loop_cutoff is kept.

    The guard uses strict `<` (gap < 300s triggers drop), so a la at
    exactly T0 + 300 s has gap == 300.0 and is kept. Documents the
    chosen `<` semantics: at-threshold means keep.

    Setup note: we deliberately use `vehicle_stops=[]` so the predictor
    has no future ETAs. Without an upcoming-stop frontier, the
    downstream cross-loop scrubs (Rules (a) and (b) at lines ~340-400
    of trips.py) cannot fire and second-guess the guard's at-threshold
    decision. This isolates the guard's `<` semantic so the test
    actually verifies what it claims to verify: that an la at gap ==
    300.0 survives the guard. With a non-empty vehicle_stops, Rule (a)
    "literal frontier rule" would drop SUR independently because SUR
    sits at-or-past the upcoming-stop frontier with no eta of its own
    -- that's its own scrub doing its own job, unrelated to the dwell
    guard's threshold semantics.
    """
    last_arrivals = {
        "STUDENT_UNION_RETURN": _iso(T0 + timedelta(seconds=300)),
    }

    result = build_trip_etas(
        trip=TRIP,
        vehicle_stops=[],  # see docstring: isolates guard's at-threshold semantic
        last_arrivals=last_arrivals,
        stops_in_route=NORTH_STOPS,
        now_utc=T0 + timedelta(minutes=6),
        loop_cutoff=T0,
    )

    # At-threshold SUR la must be kept (gap == 300.0, not < 300).
    sur = result["STUDENT_UNION_RETURN"]
    assert sur["passed"] is True, f"at-threshold SUR should be kept: {sur}"
    assert sur["last_arrival"] is not None, (
        f"at-threshold SUR la should survive: {sur}"
    )
