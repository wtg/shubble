"""Regression tests for duplicate-coord gating of the spurious-upcoming
la scrub in build_trip_etas (quick task 260414-nq4).

The scrub originally dropped any last_arrival whose corresponding
eta_lookup value was >60 s in the future, on the assumption the
detection was a polyline-projection artifact at a self-intersection
(the WEST STUDENT_UNION ≡ STUDENT_UNION_RETURN case).

Live monitoring on 2026-04-14 caught the scrub mis-firing on
HOUSTON_FIELD_HOUSE: a real 6.9 m close-approach detection at
20:58:44 was scrubbed 14 s later when the predictor's polyline_idx
hadn't yet advanced past HFH and projected an eta 61 s in the
future. Result: la dropped, UI flipped from "Last: 20:58" to a
live ETA, then flipped back the following cycle.

Fix: gate the scrub on the same `_ROUTE_REMAP_CACHE` set that
`resolve_duplicate_stops` consults — only duplicate-coord stops
are subject to the scrub.
"""

from datetime import datetime, timedelta, timezone

from backend.worker.trips import build_trip_etas


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def test_non_duplicate_stop_la_survives_future_eta():
    """HFH is NOT a duplicate-coord stop. A real close-approach
    detection (la set ~14 s ago) must survive even when the
    predictor's eta_lookup projects HFH 2 min in the future
    (polyline_idx tick-boundary lag). Without the fix the scrub
    drops the la and the UI flickers 'Last → live eta → Last' for
    one worker cycle.
    """
    now = datetime(2026, 4, 14, 20, 58, 58, tzinfo=timezone.utc)
    loop_cutoff = now - timedelta(minutes=10)
    hfh_la = now - timedelta(seconds=14)

    last_arrivals = {
        "STUDENT_UNION": _iso(loop_cutoff),
        "HOUSTON_FIELD_HOUSE": _iso(hfh_la),
    }
    # Predictor projects HFH 2 min out — well past the 60 s grace
    # window the scrub originally enforced. Without the duplicate-
    # coord gate, the scrub would drop HFH from last_arrivals.
    vehicle_stops = [
        ("HOUSTON_FIELD_HOUSE", now + timedelta(minutes=2)),
        ("BLITMAN", now + timedelta(minutes=4)),
    ]
    stops_in_route = [
        "STUDENT_UNION",
        "HOUSTON_FIELD_HOUSE",
        "BLITMAN",
        "STUDENT_UNION_RETURN",
    ]
    trip = {
        "trip_id": "NORTH:nq4",
        "route": "NORTH",
        "vehicle_id": "000000000000002",
        "status": "active",
    }

    result = build_trip_etas(
        trip=trip,
        vehicle_stops=vehicle_stops,
        last_arrivals=last_arrivals,
        stops_in_route=stops_in_route,
        now_utc=now,
        loop_cutoff=loop_cutoff,
    )

    hfh = result["HOUSTON_FIELD_HOUSE"]
    assert hfh["passed"] is True, (
        f"HFH close-approach detection was scrubbed: {hfh}"
    )
    assert hfh["last_arrival"] == _iso(hfh_la), hfh
    assert hfh["eta"] is None, (
        f"passed=True but eta still set — invariant violation: {hfh}"
    )


def test_duplicate_coord_stop_la_still_dropped():
    """STUDENT_UNION_RETURN IS a duplicate-coord stop on WEST
    (shares a coord with STUDENT_UNION). A polyline-jitter
    detection at the physical Union, mis-attributed to
    STUDENT_UNION_RETURN, must still be dropped — the predictor's
    8-minute future eta is the truth, the la is noise. This is
    the original WEST bug the scrub was written to fix.
    """
    now = datetime(2026, 4, 14, 15, 58, tzinfo=timezone.utc)
    loop_cutoff = now - timedelta(minutes=19)

    last_arrivals = {
        "STUDENT_UNION": _iso(loop_cutoff),
        "POLYTECHNIC": _iso(now - timedelta(minutes=2)),
        # SPURIOUS: jitter at physical Union mis-tagged as the
        # loop-end stop. Predictor places the real arrival 8 min
        # out, so this la is impossible.
        "STUDENT_UNION_RETURN": _iso(now - timedelta(minutes=2)),
    }
    vehicle_stops = [
        ("CITY_STATION", now + timedelta(minutes=1)),
        ("STUDENT_UNION_RETURN", now + timedelta(minutes=8)),
    ]
    stops_in_route = [
        "STUDENT_UNION",
        "ACADEMY_HALL",
        "POLYTECHNIC",
        "CITY_STATION",
        "STUDENT_UNION_RETURN",
    ]
    trip = {
        "trip_id": "WEST:nq4",
        "route": "WEST",
        "vehicle_id": "v4",
        "status": "active",
    }

    result = build_trip_etas(
        trip=trip,
        vehicle_stops=vehicle_stops,
        last_arrivals=last_arrivals,
        stops_in_route=stops_in_route,
        now_utc=now,
        loop_cutoff=loop_cutoff,
    )

    sur = result["STUDENT_UNION_RETURN"]
    assert sur["passed"] is False, (
        f"WEST STUDENT_UNION_RETURN scrub regression: {sur}"
    )
    assert sur["last_arrival"] is None, sur
    assert sur["eta"] is not None, sur
