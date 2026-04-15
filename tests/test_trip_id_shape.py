"""Regression tests for the canonical trip_id shape produced by
``compute_trips_from_vehicle_data``.

Locks in the fix from quick task 260415-emp: every trip_id in
``/api/trips`` must match the shape ``{route}:{iso_departure_time}``
with NO ``:vid`` or ``:done`` suffix. The frontend dedups rows by
trip_id, so suffixes create duplicate rows per ``(route, dep)`` slot.

Two invariants are tested:

1. **Shape** — trip_id has no ``:vid`` (all-digit segment after the
   timestamp) and no ``:done`` substring. The "no extra colon" check
   allows the ISO timestamp's own colons (e.g. ``NORTH:2026-04-15T19:30:00+00:00``
   has multiple colons but no vid/done suffix) by anchoring the match
   to recognizable suffix patterns rather than colon-count.

2. **Uniqueness** — ``Counter((t["route"], t["departure_time"]) for t in trips)``
   has no values ``> 1``. One row per slot.

Every trip in the output — active, completed, scheduled (including
idle-bound), and unassigned-scheduled — is exercised.
"""

import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import pandas as pd
import pytest

from backend.worker import trips as trips_module
from backend.worker.trips import compute_trips_from_vehicle_data
from shared.stops import Stops


# Regex that a canonical trip_id must match:
#   - Starts with a route name (letters/underscores/digits)
#   - Then exactly one colon separating the route from the ISO timestamp
#   - Then the ISO timestamp (which contains its own colons / `+` for tz)
# We avoid counting colons because ISO timestamps naturally carry 2-3
# of them (e.g., "14:30:00+00:00"). Instead, we explicitly reject the
# two banned suffixes:
#   - `:done`  (completed-trip tag)
#   - `:<digits>` tail that looks like a vid (Samsara IDs are all-digit)
# Anything else is allowed — including timezone suffixes.
_BANNED_DONE_RE = re.compile(r":done(?:$|[^a-zA-Z])")
_BANNED_VID_RE = re.compile(r":\d{12,}(?::done)?$")


def _has_banned_suffix(trip_id: str) -> bool:
    """True if trip_id contains the `:done` or `:<all-digit-vid>` tail."""
    if _BANNED_DONE_RE.search(trip_id):
        return True
    if _BANNED_VID_RE.search(trip_id):
        return True
    return False


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _loop_detections_df(
    vid: str,
    route: str,
    loop1_union_ts: datetime,
    loop1_colonie_ts: datetime,
    loop1_georgian_ts: datetime,
    loop2_union_ts: datetime,
) -> pd.DataFrame:
    """Build a processed vehicle_df that triggers both an active and a
    completed trip emission (loop 1 just finished, loop 2 just started).

    This exercises the two non-scheduled trip_id emission sites in
    ``compute_trips_from_vehicle_data``:
      1. The active trip for loop 2.
      2. The "just-completed" trip for loop 1 that runs alongside.
    """
    STOP_COORDS = {
        "STUDENT_UNION": (42.7300, -73.6800),
        "COLONIE":       (42.7350, -73.6750),
        "GEORGIAN":      (42.7400, -73.6700),
    }
    base = {
        "vehicle_id": vid,
        "route": route,
        "polyline_idx": 0,
        "speed_kmh": 20.0,
    }
    rows = []
    for ts, stop in [
        (loop1_union_ts, "STUDENT_UNION"),
        (loop1_colonie_ts, "COLONIE"),
        (loop1_georgian_ts, "GEORGIAN"),
    ]:
        lat, lon = STOP_COORDS[stop]
        rows.append({**base, "timestamp": ts, "stop_name": stop,
                     "latitude": lat, "longitude": lon})
    # Loop-2 Union detection nudged slightly so the no-movement filter
    # doesn't see both Union samples as the same point.
    ulat, ulon = STOP_COORDS["STUDENT_UNION"]
    rows.append({**base, "timestamp": loop2_union_ts, "stop_name": "STUDENT_UNION",
                 "latitude": ulat + 0.0001, "longitude": ulon + 0.0001})
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _idle_vehicle_df(
    vid: str,
    route_stops,
    first_stop_coords,
    *,
    dwell_start: datetime,
    dwell_end: datetime,
    step_sec: int = 30,
) -> pd.DataFrame:
    """Synthesize a vehicle_df for a shuttle that has been parked at
    first_stop for the full dwell window (triggers the idle-filter and
    the scheduled-slot binding side-channel)."""
    rows = []
    ts = dwell_start
    while ts <= dwell_end:
        rows.append({
            "vehicle_id": vid,
            "timestamp": ts,
            "stop_name": route_stops[0],
            "latitude": first_stop_coords[0],
            "longitude": first_stop_coords[1],
            "route": "NORTH",
            "polyline_idx": 0,
            "speed_kmh": 0.0,
        })
        ts = ts + timedelta(seconds=step_sec)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ---- Invariant 1: no banned suffixes ---------------------------------


def test_helper_regex_recognizes_banned_suffixes():
    """Lock in that the banned-suffix detector actually fires on the
    pre-fix shapes. Without this meta-test a false-negative in the
    regex (e.g. too-narrow anchoring) would silently pass the real
    invariant tests below."""
    # :done suffix — the completed-trip shape from pre-fix
    assert _has_banned_suffix("NORTH:2026-04-15T14:30:00+00:00:000000000000001:done")
    assert _has_banned_suffix("NORTH:2026-04-15T14:30:00:done")
    # :<vid> suffix where vid is an all-digit Samsara ID
    assert _has_banned_suffix("NORTH:2026-04-15T14:30:00+00:00:000000000000001")
    assert _has_banned_suffix("WEST:2026-04-15T14:25:00+00:00:000000000000004")
    # Canonical shapes must NOT match
    assert not _has_banned_suffix("NORTH:2026-04-15T14:30:00+00:00")
    assert not _has_banned_suffix("WEST:2026-04-15T14:25:00+00:00")
    assert not _has_banned_suffix("NORTH:2026-04-15T14:30:00")


def test_trip_id_has_no_vid_or_done_suffix(monkeypatch):
    """Every trip_id in the output must be ``{route}:{iso_dep}`` — no
    ``:vid`` and no ``:done`` anywhere.

    Scenario: mix of active + completed + scheduled (including
    idle-bound) trips, so every trip_id emission site in
    ``compute_trips_from_vehicle_data`` is exercised simultaneously.
    """
    route_data = Stops.routes_data["NORTH"]
    north_stops = route_data["STOPS"]
    first_stop = north_stops[0]
    first_stop_coords = tuple(route_data[first_stop]["COORDINATES"])

    now = datetime(2026, 4, 15, 14, 16, tzinfo=timezone.utc)
    loop1_union = datetime(2026, 4, 15, 14, 0, tzinfo=timezone.utc)
    loop1_colonie = datetime(2026, 4, 15, 14, 3, tzinfo=timezone.utc)
    loop1_georgian = datetime(2026, 4, 15, 14, 6, tzinfo=timezone.utc)
    loop2_union = datetime(2026, 4, 15, 14, 14, tzinfo=timezone.utc)

    # Running vehicle → active + completed emission.
    vid_running = "000000000000001"
    df_running = _loop_detections_df(
        vid_running, "NORTH",
        loop1_union, loop1_colonie, loop1_georgian,
        loop2_union_ts=loop2_union,
    )

    # Idle vehicle → triggers idle-binding side-channel on the scheduled
    # emission site. Dwell > IDLE_THRESHOLD_SEC (1200s).
    vid_idle = "000000000000099"
    dwell_start = now - timedelta(seconds=1800)
    dwell_end = now
    df_idle = _idle_vehicle_df(
        vid_idle, north_stops, first_stop_coords,
        dwell_start=dwell_start, dwell_end=dwell_end,
    )

    df = pd.concat([df_running, df_idle], ignore_index=True)

    # Schedule carries a future NORTH slot that the idle vid will bind to
    # (exercise the idle-bound scheduled-trip emission site) plus another
    # slot that stays unassigned (exercise the pure-scheduled site).
    scheduled = {
        "NORTH": [
            now + timedelta(minutes=10),
            now + timedelta(minutes=25),
        ],
    }
    monkeypatch.setattr(trips_module, "_load_today_schedule", lambda _tz: scheduled)

    vehicle_stop_etas = {
        vid_running: {
            "route": "NORTH",
            "stops": [
                ("COLONIE", now + timedelta(minutes=1)),
                ("GEORGIAN", now + timedelta(minutes=4)),
            ],
        },
        vid_idle: {"route": "NORTH", "stops": []},
    }
    last_arrivals_by_vehicle = {
        vid_running: {
            "STUDENT_UNION": _iso(loop2_union),
            "COLONIE": _iso(loop1_colonie),
            "GEORGIAN": _iso(loop1_georgian),
        },
        vid_idle: {},
    }

    trips = compute_trips_from_vehicle_data(
        vehicle_stop_etas=vehicle_stop_etas,
        last_arrivals_by_vehicle=last_arrivals_by_vehicle,
        full_df=df,
        routes_data=Stops.routes_data,
        vehicle_ids=[vid_running, vid_idle],
        now_utc=now,
        campus_tz=timezone.utc,
    )

    # Sanity: at least one scheduled trip WITH a bound vehicle_id is in
    # the output. That's the pre-fix regression surface: idle-binding
    # appended `:vid` to the scheduled trip's trip_id. If this assertion
    # fails, the shape check below isn't exercising anything meaningful.
    scheduled_with_vid = [
        t for t in trips if t["status"] == "scheduled" and t.get("vehicle_id")
    ]
    assert scheduled_with_vid, (
        f"no scheduled+idle-bound trip emitted — shape check would be a no-op: "
        f"{trips}"
    )

    # Core invariant: no banned suffix on ANY trip_id.
    bad = [t["trip_id"] for t in trips if _has_banned_suffix(t["trip_id"])]
    assert not bad, (
        f"Found trip_ids with :vid or :done suffix (pre-fix shape): {bad}"
    )

    # Shape check: every trip_id starts with "ROUTE:" where ROUTE matches
    # one of the known route names present in the output.
    for t in trips:
        tid = t["trip_id"]
        prefix = f"{t['route']}:"
        assert tid.startswith(prefix), (
            f"trip_id {tid!r} does not start with '{prefix}'"
        )
        # Remainder after "ROUTE:" must equal the canonical departure_time.
        assert tid[len(prefix):] == t["departure_time"], (
            f"trip_id tail {tid[len(prefix):]!r} != departure_time "
            f"{t['departure_time']!r}"
        )


# ---- Invariant 2: one row per (route, departure_time) -----------------


def test_no_duplicate_route_departure_pairs(monkeypatch):
    """Each ``(route, departure_time)`` pair must appear at MOST once in
    the /api/trips output.

    This is the single-row-per-slot invariant the frontend dedup keys on.
    The pre-fix :vid/:done suffixes collided on the same slot across
    active + completed + scheduled lifecycle states, producing 2 rows
    for the same departure (as seen in the live snapshot in
    quick-260415-emp's CONTEXT).
    """
    route_data = Stops.routes_data["NORTH"]
    north_stops = route_data["STOPS"]
    first_stop = north_stops[0]
    first_stop_coords = tuple(route_data[first_stop]["COORDINATES"])

    now = datetime(2026, 4, 15, 14, 16, tzinfo=timezone.utc)
    loop1_union = datetime(2026, 4, 15, 14, 0, tzinfo=timezone.utc)
    loop1_colonie = datetime(2026, 4, 15, 14, 3, tzinfo=timezone.utc)
    loop1_georgian = datetime(2026, 4, 15, 14, 6, tzinfo=timezone.utc)
    loop2_union = datetime(2026, 4, 15, 14, 14, tzinfo=timezone.utc)

    vid_running = "000000000000001"
    df_running = _loop_detections_df(
        vid_running, "NORTH",
        loop1_union, loop1_colonie, loop1_georgian,
        loop2_union_ts=loop2_union,
    )
    vid_idle = "000000000000099"
    df_idle = _idle_vehicle_df(
        vid_idle, north_stops, first_stop_coords,
        dwell_start=now - timedelta(seconds=1800),
        dwell_end=now,
    )
    df = pd.concat([df_running, df_idle], ignore_index=True)

    scheduled = {
        "NORTH": [
            now + timedelta(minutes=10),
            now + timedelta(minutes=25),
        ],
    }
    monkeypatch.setattr(trips_module, "_load_today_schedule", lambda _tz: scheduled)

    vehicle_stop_etas = {
        vid_running: {
            "route": "NORTH",
            "stops": [
                ("COLONIE", now + timedelta(minutes=1)),
                ("GEORGIAN", now + timedelta(minutes=4)),
            ],
        },
        vid_idle: {"route": "NORTH", "stops": []},
    }
    last_arrivals_by_vehicle = {
        vid_running: {
            "STUDENT_UNION": _iso(loop2_union),
            "COLONIE": _iso(loop1_colonie),
            "GEORGIAN": _iso(loop1_georgian),
        },
        vid_idle: {},
    }

    trips = compute_trips_from_vehicle_data(
        vehicle_stop_etas=vehicle_stop_etas,
        last_arrivals_by_vehicle=last_arrivals_by_vehicle,
        full_df=df,
        routes_data=Stops.routes_data,
        vehicle_ids=[vid_running, vid_idle],
        now_utc=now,
        campus_tz=timezone.utc,
    )

    # Single-row-per-slot invariant: no (route, departure_time) repeats.
    counts = Counter((t["route"], t["departure_time"]) for t in trips)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    assert not duplicates, (
        f"/api/trips has duplicate (route, departure_time) pairs — "
        f"the frontend will render 2+ rows per slot: {duplicates}"
    )

    # Regression guard: trip_id uniqueness is an even stronger check
    # (implied by slot uniqueness given the canonical shape). Verify it
    # directly so a future refactor that re-introduces a slot-agnostic
    # trip_id gets caught here too.
    tid_counts = Counter(t["trip_id"] for t in trips)
    duplicate_tids = {k: v for k, v in tid_counts.items() if v > 1}
    assert not duplicate_tids, (
        f"/api/trips has duplicate trip_id values: {duplicate_tids}"
    )
