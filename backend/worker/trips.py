"""Per-trip ETA tracking.

This module replaces the global per-stop ETA system with a per-trip model
similar to real transit apps. Each scheduled departure time is a "trip",
and each trip is assigned to a specific shuttle. ETAs are computed
independently per trip, so concurrent shuttles don't fight over displayed
data.

Key concepts:
- Trip: a scheduled or actual departure from the first stop of a route.
  Identified by (route, departure_time).
- Trip assignment: each active shuttle is matched to the nearest scheduled
  departure within a window; if none matches, a new injected trip is
  created with the shuttle's actual departure time.
- Per-trip ETAs: each trip has its own stop ETA list derived from its
  assigned shuttle's current position.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.time_utils import dev_now

logger = logging.getLogger(__name__)

# Path to the shared schedule file
SCHEDULE_PATH = Path(__file__).parent.parent.parent / "shared" / "aggregated_schedule.json"

# Matching window: a shuttle's actual departure must be within this
# many seconds of a scheduled time to be considered a match.
# Outside this window, a new injected trip is created.
# 5 minutes matches real-world shuttle tolerance (it's normal for a
# shuttle to leave a few minutes late).
MATCH_WINDOW_SEC = 300  # 5 minutes

# Hide a shuttle's trip when it hasn't made meaningful forward progress
# for this long. Must be longer than one full loop (~15 min) so a slow
# shuttle finishing its loop isn't accidentally hidden — only shuttles
# that have clearly stopped running.
IDLE_THRESHOLD_SEC = 1200  # 20 minutes

# When checking "no movement", compare the latest GPS position to the
# position from this many seconds ago. If the shuttle hasn't moved
# more than NO_MOVEMENT_DIST_M in that span, it's parked. The lookback
# must be longer than a typical mid-route stop dwell time (~30s) so
# brief stops don't accidentally hide trips, but short enough that a
# real break gets caught quickly.
NO_MOVEMENT_LOOKBACK_SEC = 600  # 10 minutes
NO_MOVEMENT_DIST_M = 100  # Less than 100m of movement = parked

# Hide a shuttle's trip when this many of its last N GPS pings are
# off-route (no route or polyline match). A single drift point won't
# trigger; sustained off-route presence (driver on break at a depot,
# parking lot, gas station, etc.) will.
OFF_ROUTE_WINDOW = 5
OFF_ROUTE_THRESHOLD = 4  # 4 of 5 = 80%


_SCHEDULE_CACHE: Dict[tuple, Dict[str, List[datetime]]] = {}


def _load_today_schedule(campus_tz) -> Dict[str, List[datetime]]:
    """Load today's schedule and parse departure times to datetimes.

    Returns:
        Dict mapping route_name to sorted list of departure datetimes (UTC).

    PERF: result is cached per (campus_date, campus_tz). The schedule JSON
    is static for the day, so re-reading and re-parsing it every worker
    cycle (every 5s) was pure waste.
    """
    now = dev_now(campus_tz)
    cache_key = (now.year, now.month, now.day, str(campus_tz))
    cached = _SCHEDULE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        with open(SCHEDULE_PATH) as f:
            schedule = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load schedule: {e}")
        return {}

    # aggregated_schedule indexed by JS getDay() (0=Sun)
    js_day = (now.weekday() + 1) % 7
    today = schedule[js_day] if js_day < len(schedule) else {}

    result: Dict[str, List[datetime]] = {}
    for route, times in today.items():
        parsed: List[datetime] = []
        for time_str in times:
            try:
                hm = datetime.strptime(time_str, "%I:%M %p")
                dt = now.replace(hour=hm.hour, minute=hm.minute, second=0, microsecond=0)
                if time_str.strip() == "12:00 AM":
                    dt += timedelta(days=1)
                parsed.append(dt.astimezone(timezone.utc))
            except ValueError:
                logger.warning(f"Could not parse schedule time: {time_str}")
        result[route] = sorted(parsed)

    # Drop prior-day entries so the cache doesn't grow unbounded
    _SCHEDULE_CACHE.clear()
    _SCHEDULE_CACHE[cache_key] = result
    return result


def _detect_vehicle_departures(
    vehicle_df: pd.DataFrame,
    first_stop: str,
    boundary_stops: Optional[List[str]] = None,
) -> List[datetime]:
    """Find all times this vehicle departed from the first stop of its route.

    Groups consecutive first-stop detections into clusters (separated by
    gaps > 120s). Each cluster represents a dwell at first_stop, and
    its LAST timestamp is the closest approximation of when the shuttle
    left to start its loop. Returns a sorted list of departure
    timestamps (UTC).

    Using the cluster's last detection (rather than its first) is
    important when a shuttle dwells at first_stop waiting for its
    scheduled departure: a 22-minute dwell from 9:53 to 10:15 should
    register as departing at 10:15 (matching the 10:15 schedule slot),
    NOT 9:53 (which would mismatch and snap to the prior 9:50 slot).

    Args:
        vehicle_df: the vehicle's sorted processed dataframe.
        first_stop: the route's first stop name (e.g. "STUDENT_UNION").
        boundary_stops: additional stop names that share first_stop's
            physical location. NORTH/WEST have STUDENT_UNION and
            STUDENT_UNION_RETURN at the SAME coordinate; the ML
            pipeline's resolve_duplicate_stops remaps back-half Union
            pings to STUDENT_UNION_RETURN. Without matching both names
            here, every loop-end crossing is invisible to the departure
            detector and `actual_departure` gets stuck at whatever the
            last raw STUDENT_UNION detection was (potentially hours or
            days ago). Pass the full list of co-located stop names.
    """
    if vehicle_df.empty or 'stop_name' not in vehicle_df.columns:
        return []

    # PERF: vectorized cluster detection. Caller provides timestamp-sorted
    # input (compute_trips_from_vehicle_data does a single upstream sort +
    # groupby), so we skip the per-call sort_values and the iterrows walk.
    match_names = {first_stop}
    if boundary_stops:
        match_names.update(boundary_stops)
    mask = vehicle_df['stop_name'].isin(match_names)
    if not mask.any():
        return []
    ts = pd.to_datetime(vehicle_df.loc[mask, 'timestamp']).reset_index(drop=True)
    if ts.empty:
        return []
    # cluster_id increments whenever the gap to the prior detection > 120s.
    # The last row of each cluster is the departure moment.
    gaps = ts.diff().dt.total_seconds().fillna(0) > 120
    cluster_ids = gaps.cumsum()
    cluster_ends = ts.groupby(cluster_ids).last().tolist()
    result: List[datetime] = []
    for raw in cluster_ends:
        dt = raw.to_pydatetime() if hasattr(raw, 'to_pydatetime') else raw
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        result.append(dt)
    return result


def build_trip_etas(
    trip: Dict[str, Any],
    vehicle_stops: List,  # List[Tuple[stop_key, eta_datetime]]
    last_arrivals: Dict[str, str],
    stops_in_route: List[str],
    now_utc: datetime,
    loop_cutoff: Optional[datetime] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build per-stop ETA dict for a single trip.

    Each stop on the route gets an entry. Stops behind the shuttle show
    `last_arrival`, stops ahead show future `eta`.

    Args:
        trip: Trip dict from compute_trips_from_vehicle_data
        vehicle_stops: List of (stop_key, eta_datetime) from the
            per-vehicle ETA computation
        last_arrivals: Dict of stop_key -> ISO timestamp. May contain
            detections from previous loops — this function filters
            by `loop_cutoff`.
        stops_in_route: Full STOPS list for the route
        now_utc: Current time
        loop_cutoff: Only count last_arrival detections at-or-after this
            moment. Pass `trip['actual_departure']` for active trips and
            `prior_departure` for completed trips. This is the SINGLE
            source of loop-boundary scoping — upstream
            `_compute_vehicle_etas_and_arrivals` returns raw per-vehicle
            detections without any time filter, and this function is the
            only place that knows which detections belong to THIS trip's
            loop. If `None`, no filtering is done (legacy behavior; avoid
            in production).

    Returns:
        Dict mapping stop_key to {eta, last_arrival, passed} entry.
    """
    # PER-TRIP LOOP SCOPING: drop any detection older than this trip's
    # departure so we never surface a prior-loop "Last:" on a stop the
    # shuttle hasn't reached yet in the current loop. See the detailed
    # rationale in `_compute_vehicle_etas_and_arrivals`.
    if loop_cutoff is not None and last_arrivals:
        filtered_las: Dict[str, str] = {}
        for k, v in last_arrivals.items():
            try:
                la_dt = datetime.fromisoformat(v)
                if la_dt.tzinfo is None:
                    la_dt = la_dt.replace(tzinfo=timezone.utc)
                if la_dt >= loop_cutoff:
                    filtered_las[k] = v
            except (ValueError, TypeError):
                continue
        last_arrivals = filtered_las

    stop_etas: Dict[str, Dict[str, Any]] = {}

    # Build a lookup for future ETAs from vehicle_stops
    eta_lookup = {}
    for stop_key, eta_dt in vehicle_stops:
        eta_lookup[stop_key] = eta_dt

    # SPURIOUS-DETECTION SCRUB: if the predictor says a stop still has a
    # significantly-future eta (minutes away) but the detection pipeline
    # also has a `last_arrival` for that same stop, the detection is
    # almost certainly a polyline-projection artifact at a loop-boundary
    # self-intersection. Example: the WEST route's STUDENT_UNION and
    # STUDENT_UNION_RETURN share a physical location, so GPS pings at
    # Union after `actual_departure` can get mis-attributed to
    # STUDENT_UNION_RETURN and pass the loop_cutoff filter. Left in
    # place, those stale detections become anchors in the monotonic
    # clamp below and copy the earlier-stop timestamps into every
    # downstream stop, producing a self-contradictory display
    # ("passed=True, last_arrival=15:56, eta=15:58") on the unreached
    # part of the route.
    #
    # TICK-BOUNDARY RACE CAVEAT: we only drop detections whose predictor
    # eta is MORE than SPURIOUS_UPCOMING_GRACE_SEC in the future. Within
    # a few seconds of "now", the predictor and the detection pipeline
    # can legitimately disagree: the GPS ping registered a stop arrival
    # (detection pipeline) but the polyline_idx hasn't advanced past
    # that stop yet (predictor still lists it as upcoming). In that
    # race, the detection IS real and dropping it causes the earlier-
    # stops backfill to stall — users see "Georgian still upcoming
    # even though Bryckwyck was just passed" for one worker cycle.
    # 60 s is large enough to catch the original WEST bug (which had
    # upcoming ETAs 2–8 minutes ahead) while small enough to let tick-
    # boundary races through.
    #
    # DUPLICATE-COORD GATING (260414-nq4): even within 60 s, the scrub
    # only fires for stops that share a coordinate with another stop on
    # the same route. Non-duplicate stops (e.g. HOUSTON_FIELD_HOUSE) get
    # legitimate close-approach detections that race the predictor's
    # polyline_idx — dropping them caused a "Last → live eta → Last"
    # flicker in the schedule UI for one worker cycle after each pass.
    # NOTE: import the module (not the names) — `_ROUTE_REMAP_CACHE` is
    # rebound by `_build_route_remap_cache()` (uses `global` and
    # reassigns to a new dict), so an `import _ROUTE_REMAP_CACHE` would
    # capture the empty initial value forever.
    from ml.data import stops as _ml_stops
    _ml_stops._build_route_remap_cache()
    duplicate_stop_names: set = set()
    for _route, entries in _ml_stops._ROUTE_REMAP_CACHE.items():
        for first_name, last_name, _threshold_idx in entries:
            duplicate_stop_names.add(first_name)
            duplicate_stop_names.add(last_name)

    SPURIOUS_UPCOMING_GRACE_SEC = 60
    if last_arrivals and eta_lookup:
        cleaned_las: Dict[str, str] = {}
        for k, v in last_arrivals.items():
            upcoming = eta_lookup.get(k)
            # Only drop as spurious when the stop is a duplicate-coord
            # stop (e.g. STUDENT_UNION ≡ STUDENT_UNION_RETURN). For
            # every other stop, a la-plus-future-eta is a normal
            # tick-boundary situation — detection wins. See the HFH
            # flicker bug investigation (quick task 260414-nq4).
            if (
                upcoming is not None
                and (upcoming - now_utc).total_seconds() > SPURIOUS_UPCOMING_GRACE_SEC
                and k in duplicate_stop_names
            ):
                continue  # predictor says not yet arrived — detection is noise
            cleaned_las[k] = v
        last_arrivals = cleaned_las

    # CROSS-LOOP la LEAK SCRUB (covers the "stale actual_departure" case):
    # The upstream loop_cutoff filter uses `trip.actual_departure` to drop
    # prior-loop detections. That works as long as `_detect_vehicle_departures`
    # correctly identifies every re-departure from the first stop. When the
    # shuttle's return dwell at Union produces no GPS pings tagged
    # STUDENT_UNION / STUDENT_UNION_RETURN (a real production case — GPS
    # gaps, or ML stop-remap ambiguity at the boundary), the detector
    # returns the PRIOR loop's departure timestamp and loop_cutoff becomes
    # the trip-N start instead of trip-N+1. In that world every prior-loop
    # la (STUDENT_UNION, COLONIE, HFH from trip N) trivially satisfies
    # `la >= loop_cutoff` and survives — the bug then surfaces as
    # COLONIE.passed=True on what the user sees as trip N+1.
    #
    # The existing spurious-upcoming scrub above only catches stops still
    # in `eta_lookup` (the WEST STUDENT_UNION_RETURN self-intersection case).
    # When the predictor has already advanced past a stop and dropped it
    # from vehicle_stops, the stop has no future eta to contradict its
    # stale la — the scrub can't fire and the backfill happily anchors on
    # the cross-loop timestamp.
    #
    # Two complementary rules extend the scrub:
    #   (a) Literal frontier rule: if a stop's index is at-or-past the
    #       first upcoming-stop frontier AND it has no corresponding eta
    #       in eta_lookup, the shuttle cannot physically have reached it
    #       (it'd be in vehicle_stops otherwise). Drop. This guards the
    #       rare case where vehicle_stops is truncated past the frontier.
    #   (b) Stale-before-frontier rule: iteratively inspect the la whose
    #       stop_idx is highest AND not in eta_lookup (the "most
    #       recently passed" stop per the last_arrivals chain). If the
    #       gap from la_ts to the first upcoming-stop's eta exceeds
    #       STALE_PRIOR_LOOP_GAP_SEC (10 min — comfortably larger than
    #       any adjacent-stop leg on the NORTH/WEST/EAST routes), the
    #       la can't belong to the current journey. Drop it and repeat —
    #       dropping the highest stale la may expose the next-highest
    #       as also stale, and we peel the whole prior-loop chain off
    #       in one pass.
    #
    # IMPORTANT: this rule only touches la entries for stops NOT in
    # eta_lookup. Stops still in eta_lookup are already covered by the
    # grace-window scrub above, and must preserve the tick-boundary race
    # behavior (STAC_1 just-arrived with eta a few seconds future).
    STALE_PRIOR_LOOP_GAP_SEC = 600  # 10 minutes
    if last_arrivals:
        stop_to_idx = {s: i for i, s in enumerate(stops_in_route)}
        # Find first upcoming eta: first stop in route order whose
        # eta_lookup entry is strictly in the future.
        first_upcoming_idx: Optional[int] = None
        first_upcoming_eta: Optional[datetime] = None
        for i, s in enumerate(stops_in_route):
            eta = eta_lookup.get(s)
            if eta is not None and eta > now_utc:
                first_upcoming_idx = i
                first_upcoming_eta = eta
                break

        # Rule (a): drop any la for stops AT-OR-PAST the frontier that
        # are not in eta_lookup. A stop past the frontier with no eta
        # cannot physically have been reached yet.
        if first_upcoming_idx is not None:
            frontier_cleaned: Dict[str, str] = {}
            for k, v in last_arrivals.items():
                if k in eta_lookup:
                    frontier_cleaned[k] = v
                    continue
                idx = stop_to_idx.get(k, -1)
                if idx >= first_upcoming_idx:
                    # At/past frontier but not in eta_lookup — drop.
                    continue
                frontier_cleaned[k] = v
            last_arrivals = frontier_cleaned

        # Rule (b): iteratively drop the highest-idx la (among stops
        # not in eta_lookup) whose gap to first_upcoming_eta exceeds
        # the implausible-journey threshold. Peels the whole stale
        # prior-loop chain off in one pass.
        if first_upcoming_eta is not None and last_arrivals:
            while True:
                # Find highest-idx la stop that is NOT in eta_lookup.
                candidate_stop: Optional[str] = None
                candidate_idx = -1
                for stop_key in last_arrivals:
                    if stop_key in eta_lookup:
                        continue  # tick-boundary cases handled above
                    idx = stop_to_idx.get(stop_key, -1)
                    if idx > candidate_idx:
                        candidate_idx = idx
                        candidate_stop = stop_key
                if candidate_stop is None:
                    break
                try:
                    la_dt = datetime.fromisoformat(last_arrivals[candidate_stop])
                    if la_dt.tzinfo is None:
                        la_dt = la_dt.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    break
                gap = (first_upcoming_eta - la_dt).total_seconds()
                if gap <= STALE_PRIOR_LOOP_GAP_SEC:
                    break  # highest plausible la reached — stop peeling
                # Implausibly long journey — drop and recheck.
                last_arrivals = {
                    k: v for k, v in last_arrivals.items() if k != candidate_stop
                }

    # Frontier: index of the first stop (in route order) with a strictly-
    # future predicted ETA. Stops whose index is BEFORE the frontier and
    # which have no real detection cannot still be "arriving" — the
    # predictor has advanced past them, even if a past-eta entry lingered
    # in eta_lookup from OFFSET-diff propagation. Surfacing those as
    # live countdowns produces negative timers on the UI.
    stop_to_idx_early = {s: i for i, s in enumerate(stops_in_route)}
    first_upcoming_idx_build: Optional[int] = None
    for _i, _s in enumerate(stops_in_route):
        _eta = eta_lookup.get(_s)
        if _eta is not None and _eta > now_utc:
            first_upcoming_idx_build = _i
            break

    for stop_key in stops_in_route:
        entry: Dict[str, Any] = {
            "eta": None,
            "last_arrival": None,
            "passed": False,
            # `passed_interpolated` is True when `passed=True` but the
            # `last_arrival` timestamp came from the gap-backfill
            # interpolation path, not from a real GPS detection. The
            # frontend should render "Passed" without a timestamp for
            # these, reserving "Last: HH:MM" for real-detection anchors.
            # See the backfill loop below for where this gets set.
            "passed_interpolated": False,
        }

        if stop_key in eta_lookup:
            eta_dt = eta_lookup[stop_key]
            # Only surface the predicted ETA if this stop is at-or-after
            # the shuttle's current frontier. Stops whose index is before
            # first_upcoming_idx_build have been passed by the predictor
            # (or never entered its ahead-of-shuttle window) and must not
            # show a ticking live countdown. Real passage is asserted by
            # the separate `last_arrivals` branch below.
            stop_idx_here = stop_to_idx_early.get(stop_key, -1)
            if (
                first_upcoming_idx_build is None
                or stop_idx_here >= first_upcoming_idx_build
            ):
                entry["eta"] = eta_dt.isoformat()
            # else: leave entry["eta"] = None (initialized above)
            # Do NOT infer passed=True from an expired ETA. The predictor
            # can easily overshoot (say "shuttle will be at ECAV at
            # 22:30" when the shuttle is still 2 minutes away), and
            # fabricating a "Last: 22:30" on an unpassed stop looks
            # like the shuttle visited when it didn't. Detection via
            # last_arrivals is the ONLY source of truth for passed.

        if stop_key in last_arrivals:
            la_iso = last_arrivals[stop_key]
            # Real GPS detection — authoritative. Clear any "future eta"
            # the predictor set in eta_lookup: a stop can't simultaneously
            # be "in the past" (la set) and "still upcoming" (eta in the
            # future). Happens at tick boundaries where the predictor's
            # polyline_idx hasn't advanced past a just-detected stop yet —
            # the detection wins, the stale predictor ETA must go.
            entry["last_arrival"] = la_iso
            entry["passed"] = True
            entry["eta"] = None

        stop_etas[stop_key] = entry

    # Defensive scrub: if a stop has a future ETA but no real detection-based
    # last_arrival, clear any la that might have been fabricated by a future
    # code path. This preserves the invariant "entries with a la are backed
    # by a real, loop-scoped detection."
    first_upcoming_idx: Optional[int] = None
    for i, stop_key in enumerate(stops_in_route):
        eta_dt = eta_lookup.get(stop_key)
        if eta_dt and eta_dt > now_utc:
            first_upcoming_idx = i
            break

    if first_upcoming_idx is not None:
        for i in range(first_upcoming_idx, len(stops_in_route)):
            stop_key = stops_in_route[i]
            entry = stop_etas[stop_key]
            eta_dt = eta_lookup.get(stop_key)
            if eta_dt and eta_dt > now_utc and stop_key not in last_arrivals:
                # Stop is upcoming and has no real detection-based la —
                # drop any stale la that might have been fabricated
                # elsewhere (defensive).
                entry["last_arrival"] = None
                entry["passed"] = False

    # Fix detection gaps + enforce monotonic last_arrivals. Passed stops
    # should have last_arrival timestamps in ascending order within a
    # single loop — a shuttle can't visit a later stop BEFORE an earlier
    # one. If the order is violated, it's because the detection matched
    # a stop from a PREVIOUS loop. Clamp the out-of-order stop's
    # last_arrival up to the preceding stop's value so the display
    # stays coherent.
    # Detection-gap backfill: a shuttle moves through stops sequentially,
    # so if any LATER stop has a real detection in last_arrivals, every
    # EARLIER stop in route order must also have been passed (even if
    # add_stops missed them). Find the farthest-reached real-detection
    # index, then mark all earlier stops as passed.
    farthest_detected_idx = -1
    for i, stop_key in enumerate(stops_in_route):
        if stop_key in last_arrivals:
            farthest_detected_idx = i

    if farthest_detected_idx >= 0:
        # Build a list of (idx, iso_la) anchor points and monotonic-clamp
        # them so any out-of-order real detections (cross-loop leaks
        # that sneaked past the freshness filter) get clamped up to
        # their preceding neighbor's la.
        anchors: List[Tuple[int, str]] = []
        for i in range(farthest_detected_idx + 1):
            la = last_arrivals.get(stops_in_route[i])
            if la:
                if anchors and la < anchors[-1][1]:
                    la = anchors[-1][1]  # clamp up
                anchors.append((i, la))

        def _interpolated_la(gap_idx: int) -> Optional[str]:
            """Linearly interpolate an la between the nearest anchor pair.

            If the gap is between two real detections, blend their
            timestamps proportional to how far the gap's index is
            between them — gives a realistic "Last: ~HH:MM" instead
            of defaulting to the preceding detection's exact time.
            If the gap is before any anchor, use the first anchor's
            la (best effort). If after all anchors, use the last.
            """
            if not anchors:
                return None
            # Find surrounding anchors
            prev_anchor = None
            next_anchor = None
            for idx, la in anchors:
                if idx <= gap_idx:
                    prev_anchor = (idx, la)
                elif idx > gap_idx and next_anchor is None:
                    next_anchor = (idx, la)
                    break
            if prev_anchor and next_anchor:
                # Interpolate
                p_idx, p_la = prev_anchor
                n_idx, n_la = next_anchor
                p_dt = datetime.fromisoformat(p_la)
                n_dt = datetime.fromisoformat(n_la)
                span = n_idx - p_idx
                if span <= 0:
                    return p_la
                ratio = (gap_idx - p_idx) / span
                interp_dt = p_dt + (n_dt - p_dt) * ratio
                return interp_dt.isoformat()
            if prev_anchor:
                return prev_anchor[1]
            if next_anchor:
                return next_anchor[1]
            return None

        for i in range(farthest_detected_idx + 1):
            stop_key = stops_in_route[i]
            entry = stop_etas[stop_key]
            real_la = last_arrivals.get(stop_key)
            if real_la:
                # Use the clamped value from anchors. Real detection —
                # passed_interpolated stays False, timestamp is honest.
                # Clear any residual future eta so the invariant
                # "passed => eta is None" holds even when the backfill
                # re-enters an entry whose initial pass set both.
                clamped = next((la for idx, la in anchors if idx == i), real_la)
                entry["last_arrival"] = clamped
                entry["passed"] = True
                entry["passed_interpolated"] = False
                entry["eta"] = None
            else:
                # Detection gap — interpolate between neighbors. The
                # shuttle physically passed this stop (a later stop has
                # a real detection), but we never caught a GPS ping at
                # this particular stop. Mark it as passed so the UI
                # shows it behind the shuttle, but flag
                # `passed_interpolated=True` so the frontend renders
                # "Passed" without a fabricated "Last: HH:MM" timestamp.
                # Students shouldn't see a synthesized time presented as
                # a real arrival.
                entry["passed"] = True
                entry["passed_interpolated"] = True
                entry["last_arrival"] = _interpolated_la(i)
                entry["eta"] = None

    return stop_etas


def compute_trips_from_vehicle_data(
    vehicle_stop_etas: Dict[str, Dict],
    last_arrivals_by_vehicle: Dict[str, Dict[str, str]],
    full_df: Optional[pd.DataFrame],
    routes_data: Dict[str, Any],
    vehicle_ids: List[str],
    now_utc: datetime,
    campus_tz,
) -> List[Dict[str, Any]]:
    """Compute per-trip ETAs from the per-vehicle computation.

    Args:
        vehicle_stop_etas: Output of per-vehicle ETA computation
            {vehicle_id: {route, stops: [(stop_key, eta_dt), ...], ...}}
        last_arrivals_by_vehicle: Per-vehicle last-arrival detections.
            Shape: {vehicle_id: {stop_key: ISO timestamp}}. Each trip
            only sees its own vehicle's history, preventing concurrent
            shuttles from marking each other's stops as passed.
        full_df: Today's processed dataframe (for detecting departures)
        routes_data: routes.json data
        vehicle_ids: List of active vehicle IDs
        now_utc: Current UTC time
        campus_tz: Campus timezone

    Returns:
        List of trip dicts, each with:
          trip_id, route, departure_time, scheduled, vehicle_id, status,
          stop_etas: {stop_key: {eta, last_arrival, passed}}
    """
    schedule = _load_today_schedule(campus_tz)
    trips: List[Dict[str, Any]] = []

    # Two-pass assignment so each scheduled time slot is claimed by at
    # most one vehicle. Pass 1: compute actual_departure per vehicle.
    # Pass 2: greedy match — for each vehicle sorted by proximity to
    # its best candidate, claim the closest un-claimed scheduled time.
    # Remaining vehicles (no match within window) become injected trips.
    assigned_scheduled_times: set = set()  # (route, iso) tuples of claimed schedule slots

    # PERF: precompute a per-vehicle sorted-group lookup once so the
    # per-vehicle loop below doesn't re-scan the full dataframe N times.
    vehicle_groups: Dict[str, pd.DataFrame] = {}
    if full_df is not None and not full_df.empty:
        target_ids = {str(v) for v in vehicle_stop_etas.keys()}
        filtered = full_df[full_df['vehicle_id'].astype(str).isin(target_ids)]
        if not filtered.empty:
            filtered = filtered.sort_values('timestamp', kind='mergesort')
            vehicle_groups = {
                str(vid): grp for vid, grp in filtered.groupby('vehicle_id', sort=False)
            }

    # Pass 1: collect per-vehicle preconditions
    vehicle_meta: List[Dict[str, Any]] = []
    for vid, vdata in vehicle_stop_etas.items():
        route = vdata.get("route")
        if not route or route not in routes_data:
            continue
        route_stops = routes_data[route].get("STOPS", [])
        if not route_stops:
            continue
        first_stop = route_stops[0]
        sched_deps = schedule.get(route, [])

        vehicle_df = vehicle_groups.get(str(vid), pd.DataFrame())

        # Find any stops co-located with first_stop (same physical coord).
        # On NORTH and WEST, STUDENT_UNION and STUDENT_UNION_RETURN share
        # one coordinate — the ML pipeline's resolve_duplicate_stops
        # remaps back-half Union pings to STUDENT_UNION_RETURN, so without
        # passing both names into the departure detector, loop returns
        # are invisible and `actual_departure` stays stuck at the first
        # raw STUDENT_UNION detection (which may be hours old).
        route_route_data = routes_data.get(route, {})
        first_stop_coord = route_route_data.get(first_stop, {}).get("COORDINATES")
        boundary_stops: List[str] = []
        if first_stop_coord is not None:
            target_key = (
                round(float(first_stop_coord[0]), 6),
                round(float(first_stop_coord[1]), 6),
            )
            for stop_name in route_stops:
                if stop_name == first_stop:
                    continue
                stop_data = route_route_data.get(stop_name, {})
                coord = stop_data.get("COORDINATES")
                if not coord:
                    continue
                key = (round(float(coord[0]), 6), round(float(coord[1]), 6))
                if key == target_key:
                    boundary_stops.append(stop_name)

        # Compute actual_departure without matching
        departures = _detect_vehicle_departures(
            vehicle_df, first_stop, boundary_stops=boundary_stops or None
        )
        prior_departure: Optional[datetime] = None
        if departures:
            actual_departure = departures[-1]
            # Penultimate departure = start of the loop that just finished
            if len(departures) >= 2:
                prior_departure = departures[-2]
        else:
            if vehicle_df.empty:
                continue
            earliest = pd.to_datetime(vehicle_df['timestamp'].min()).to_pydatetime()
            if earliest.tzinfo is None:
                earliest = earliest.replace(tzinfo=timezone.utc)
            actual_departure = earliest

        # --- Idle/off-route filter ----------------------------------
        # Don't emit a trip when the shuttle is parked. The forward-
        # projected ETAs would be pure speculation, and students would
        # see phantom arrival times that never materialize. The shuttle
        # still appears on the map via /api/locations — only the
        # schedule trip row is suppressed.
        if not vehicle_df.empty:
            last_point = vehicle_df.iloc[-1]

            # Filter 1: idle at the route's first stop for longer than
            # a full loop (driver on break at Union, end-of-day, etc.)
            #
            # We use a distance-based motion check rather than
            # speed_kmh. The stationary-override in ml/pipelines.py
            # zeros out speed when raw GPS movement is <5m, so this
            # check is now reliable — but the original `speed <= 1.0`
            # was also fooled by polyline-projection jitter that
            # produced phantom 30-60 km/h readings on parked shuttles.
            # Distance-from-actual_departure is the simpler signal.
            latest_stop = last_point.get('stop_name')
            latest_speed = last_point.get('speed_kmh')
            idle_seconds = (now_utc - actual_departure).total_seconds()
            at_first_stop = pd.notna(latest_stop) and str(latest_stop) == first_stop
            not_moving = pd.isna(latest_speed) or float(latest_speed) <= 1.0
            if at_first_stop and not_moving and idle_seconds > IDLE_THRESHOLD_SEC:
                logger.debug(
                    f"Skipping trip for vehicle {vid} on {route}: "
                    f"idle at {first_stop} for {idle_seconds:.0f}s"
                )
                continue

            # Filter 2: sustained off-route presence (driver on break
            # at a depot/lot/gas station, far from any polyline)
            if 'route' in vehicle_df.columns and 'polyline_idx' in vehicle_df.columns:
                recent = vehicle_df.tail(OFF_ROUTE_WINDOW)
                off_route_count = int(
                    (recent['route'].isna() & recent['polyline_idx'].isna()).sum()
                )
                if off_route_count >= OFF_ROUTE_THRESHOLD:
                    logger.debug(
                        f"Skipping trip for vehicle {vid} on {route}: "
                        f"{off_route_count}/{len(recent)} recent points off-route"
                    )
                    continue

            # Filter 3: no movement in the last NO_MOVEMENT_LOOKBACK_SEC.
            # Catches shuttles stuck mid-route (driver on break at a
            # mid-route stop, broken-down vehicle, etc.) — Filter 1
            # only catches the first-stop case.
            #
            # IMPORTANT: compare the MAX distance from the current
            # position across all points in the lookback window, not
            # just the endpoint-to-endpoint distance. A shuttle that
            # completes a full loop in 10 min returns to its starting
            # point, so endpoint-to-endpoint would read ~0m even
            # though the shuttle traveled the full route. Using max
            # distance catches this correctly: if the shuttle ever
            # got more than NO_MOVEMENT_DIST_M away from its current
            # position within the window, it's moving.
            if 'latitude' in vehicle_df.columns and 'longitude' in vehicle_df.columns:
                latest_ts = pd.to_datetime(last_point['timestamp']).to_pydatetime()
                if latest_ts.tzinfo is None:
                    latest_ts = latest_ts.replace(tzinfo=timezone.utc)
                lookback_cutoff = latest_ts - timedelta(seconds=NO_MOVEMENT_LOOKBACK_SEC)
                vehicle_df_ts = pd.to_datetime(vehicle_df['timestamp'], utc=True)
                window_points = vehicle_df[vehicle_df_ts >= lookback_cutoff]
                if len(window_points) >= 2:
                    try:
                        latest_lat = float(last_point['latitude'])
                        latest_lon = float(last_point['longitude'])
                        R = 6371000  # Earth radius, meters
                        # Compute haversine distance from current position to
                        # every point in the window. If the max is less than
                        # NO_MOVEMENT_DIST_M, the shuttle is parked (not just
                        # at a temporary wrap-around).
                        lats = window_points['latitude'].astype(float).values
                        lons = window_points['longitude'].astype(float).values
                        phi0 = np.radians(latest_lat)
                        lam0 = np.radians(latest_lon)
                        phis = np.radians(lats)
                        lams = np.radians(lons)
                        dphi = phis - phi0
                        dlam = lams - lam0
                        a = (np.sin(dphi / 2) ** 2
                             + np.cos(phi0) * np.cos(phis) * np.sin(dlam / 2) ** 2)
                        dists = 2 * R * np.arcsin(np.sqrt(a))
                        max_dist = float(np.nanmax(dists)) if len(dists) else 0.0
                        if max_dist < NO_MOVEMENT_DIST_M:
                            logger.debug(
                                f"Skipping trip for vehicle {vid} on {route}: "
                                f"max displacement only {max_dist:.0f}m in last "
                                f"{NO_MOVEMENT_LOOKBACK_SEC}s"
                            )
                            continue
                    except Exception:
                        pass  # On any error, fall through and emit the trip
        # ------------------------------------------------------------

        # Detect "dwelling at first_stop after a loop completed". When a
        # shuttle finishes its loop and is sitting at Union waiting for
        # its next departure, the most-recent first-stop detection is
        # the ARRIVAL of the loop that just finished — not a real
        # departure. Showing that arrival time on the timeline as
        # "leaving at HH:MM" is misleading: the user sees the current
        # time (~now) instead of the next scheduled departure.
        #
        # The signal: (a) latest point is at first_stop, (b) the
        # cluster started recently enough to be a CURRENT dwell (not
        # an idle shuttle from earlier in the day), (c) there's a
        # future scheduled departure we can promote to.
        #
        # We deliberately do NOT check speed here. When a shuttle is
        # parked at Union, GPS jitter projects onto the polyline in
        # ways that produce wildly inflated `speed_kmh` values (a
        # 1cm physical move can become 30+ km/h after polyline
        # projection at an intersection). Stop-based detection is
        # more reliable than speed at the dwell point.
        is_dwelling = False
        if not vehicle_df.empty:
            latest_stop = last_point.get('stop_name')
            at_first_stop = pd.notna(latest_stop) and str(latest_stop) == first_stop
            age_of_departure = (now_utc - actual_departure).total_seconds()
            # "Recent" cluster = the shuttle just rolled into Union.
            # Upper bound generous enough to cover typical inter-loop
            # dwells (~60-1200s) but short enough to stay out of the
            # idle-filter territory (>1200s).
            is_dwelling = (
                at_first_stop
                and 0 <= age_of_departure <= 1200
            )

        # Compute best candidate distance for sort ordering
        best_diff = float('inf')
        for sched in sched_deps:
            diff = abs((sched - actual_departure).total_seconds())
            if diff < best_diff:
                best_diff = diff
        vehicle_meta.append({
            "vid": vid,
            "vdata": vdata,
            "route": route,
            "route_stops": route_stops,
            "sched_deps": sched_deps,
            "actual_departure": actual_departure,
            "prior_departure": prior_departure,
            "best_diff": best_diff,
            "is_dwelling": is_dwelling,
        })

    # Pass 2: assign in order of best-fit — vehicles closest to a schedule
    # slot get first pick, preventing a later vehicle from stealing a slot
    # that would have been a better match for someone else.
    vehicle_meta.sort(key=lambda m: m["best_diff"])

    for meta in vehicle_meta:
        vid = meta["vid"]
        vdata = meta["vdata"]
        route = meta["route"]
        route_stops = meta["route_stops"]
        sched_deps = meta["sched_deps"]
        actual_departure = meta["actual_departure"]
        prior_departure = meta.get("prior_departure")
        is_dwelling = meta.get("is_dwelling", False)

        # When the shuttle is dwelling at Union after a loop just
        # finished, promote it to the NEXT future scheduled departure
        # rather than matching to the closest-in-either-direction slot.
        # This makes the timeline row honor the static schedule: a
        # shuttle that arrived at 14:27 and has a 14:30/14:45/15:00
        # schedule will show as the 14:30 row (next future slot), not
        # a phantom injected 14:27 row.
        matched_scheduled: Optional[datetime] = None
        if is_dwelling:
            future_slots = sorted(
                s for s in sched_deps
                if s > now_utc
                and (route, s.isoformat()) not in assigned_scheduled_times
            )
            if future_slots:
                matched_scheduled = future_slots[0]

        if matched_scheduled is None:
            # Find the closest UNCLAIMED scheduled time within the window
            min_diff = float('inf')
            for sched in sched_deps:
                if (route, sched.isoformat()) in assigned_scheduled_times:
                    continue
                diff = abs((sched - actual_departure).total_seconds())
                if diff < min_diff and diff <= MATCH_WINDOW_SEC:
                    min_diff = diff
                    matched_scheduled = sched

        if matched_scheduled is not None:
            trip_time = matched_scheduled
            scheduled = True
            assigned_scheduled_times.add((route, matched_scheduled.isoformat()))
        else:
            trip_time = actual_departure
            scheduled = False

        # When a shuttle is dwelling and we promoted it to a future
        # scheduled slot, the trip status is "scheduled" (upcoming),
        # not "active" — it hasn't actually started the next loop yet.
        if is_dwelling and scheduled and trip_time > now_utc:
            status = "scheduled"
        else:
            status = "scheduled" if actual_departure > now_utc else "active"

        # Trip completion: if the vehicle has arrived at the LAST stop of
        # the route recently, the current loop is done. Mark it completed
        # so the frontend stops showing ETAs for a phantom "next loop".
        # The vehicle will start a new trip once it physically departs
        # again and moves past the first stop.
        vehicle_las = last_arrivals_by_vehicle.get(str(vid), {})
        last_stop = route_stops[-1]
        vehicle_future_stops = vdata.get("stops", [])
        # Completion signal: the vehicle has no more future stops to
        # predict. compute_per_stop_etas returns entries for every
        # stop AHEAD of the shuttle's current position; when that's
        # empty the shuttle has either finished its loop or the
        # position is past the last polyline.
        #
        # We deliberately do NOT use `last_stop in vehicle_las`
        # anymore. Now that STUDENT_UNION_RETURN is detected
        # (resolve_duplicate_stops fix), that signal fires constantly
        # for looping shuttles — the return detection is fresh every
        # wrap-around. Relying on future-stop emptiness is robust
        # against detection noise.
        if status == "active" and len(vehicle_future_stops) == 0:
            status = "completed"

        trip = {
            "trip_id": f"{route}:{trip_time.isoformat()}:{vid}",
            "route": route,
            "departure_time": trip_time.isoformat(),
            "actual_departure": actual_departure.isoformat(),
            "scheduled": scheduled,
            "vehicle_id": vid,
            "status": status,
        }

        # Loop-scope the active trip's last_arrivals to detections at-or-after
        # the current loop's start (= actual_departure). For a dwelling shuttle
        # actual_departure is ~now, so the filter produces an empty dict — that
        # is correct because the "active" trip for a dwelling shuttle represents
        # the UPCOMING scheduled loop which hasn't started yet.
        stop_etas = build_trip_etas(
            trip=trip,
            vehicle_stops=vehicle_future_stops,
            last_arrivals=vehicle_las,
            stops_in_route=route_stops,
            now_utc=now_utc,
            loop_cutoff=actual_departure,
        )
        trip["stop_etas"] = stop_etas
        trips.append(trip)

        # If this vehicle just started a new loop (prior_departure recent),
        # also emit the just-completed loop as a separate "completed" trip.
        # This prevents the confusing moment when a shuttle reaches Union
        # and the UI instantly replaces the old trip's ETAs with a new loop.
        # The completed trip shows all stops as passed, giving the user a
        # clear visual "that loop is done".
        if prior_departure is not None:
            gap = (actual_departure - prior_departure).total_seconds()
            age_since_new_loop = (now_utc - actual_departure).total_seconds()
            # Only show the completed trip briefly (first 2 minutes of the
            # new loop) and only if the loops are separated by at least
            # 3 minutes (avoids false positives from GPS noise).
            if gap > 180 and age_since_new_loop < 120:
                # Match the prior_departure to a scheduled slot for the
                # display time, so the completed trip aligns with the
                # static schedule (e.g. shows "10:15 PM" instead of the
                # raw "10:15:23"). This mirrors the active-trip matching
                # done above.
                prior_matched: Optional[datetime] = None
                prior_min_diff = float('inf')
                for sched in sched_deps:
                    diff = abs((sched - prior_departure).total_seconds())
                    if diff < prior_min_diff and diff <= MATCH_WINDOW_SEC:
                        prior_min_diff = diff
                        prior_matched = sched
                prior_display = prior_matched if prior_matched else prior_departure
                completed_trip = {
                    "trip_id": f"{route}:{prior_display.isoformat()}:{vid}:done",
                    "route": route,
                    "departure_time": prior_display.isoformat(),
                    "actual_departure": prior_departure.isoformat(),
                    "scheduled": prior_matched is not None,
                    "vehicle_id": vid,
                    "status": "completed",
                }
                # Loop-scope to the just-finished loop via prior_departure.
                # Route through build_trip_etas so the completed trip goes
                # through the same assembly (monotonic clamp, anchor
                # interpolation) as the active trip — single source of truth.
                # Pass an empty vehicle_stops list because a completed trip
                # has no future ETAs.
                completed_stop_etas = build_trip_etas(
                    trip=completed_trip,
                    vehicle_stops=[],
                    last_arrivals=vehicle_las,
                    stops_in_route=route_stops,
                    now_utc=now_utc,
                    loop_cutoff=prior_departure,
                )
                # Mark every stop as passed — the loop is done. Any stop
                # missing a real detection (because add_stops missed it)
                # still gets passed=True by the backfill logic inside
                # build_trip_etas, which interpolates an la from
                # neighbors. Only stops that have NO neighbors with la
                # (e.g. the shuttle never fully completed the loop)
                # remain passed=False — in that case we force passed=True
                # here since the completed-trip emission only fires when
                # we know the loop finished.
                for stop_key in route_stops:
                    entry = completed_stop_etas.get(stop_key)
                    if entry is None:
                        # No real detection and no neighbor to interpolate
                        # from — mark as passed (loop is done) but flag
                        # passed_interpolated=True since there's no real
                        # arrival timestamp to show.
                        completed_stop_etas[stop_key] = {
                            "eta": None,
                            "last_arrival": None,
                            "passed": True,
                            "passed_interpolated": True,
                        }
                    else:
                        entry["eta"] = None  # completed trip has no future
                        # If the stop was force-marked passed here but had
                        # no real detection (passed was False coming out of
                        # build_trip_etas), the value is inferred. Preserve
                        # passed_interpolated state from build_trip_etas
                        # when it's already set; otherwise default to True
                        # if we're about to flip `passed` from False to True.
                        if not entry.get("passed"):
                            entry["passed_interpolated"] = True
                        entry["passed"] = True
                completed_trip["stop_etas"] = completed_stop_etas
                trips.append(completed_trip)

    # Add scheduled trips that don't have a vehicle assigned yet
    # (so frontend can show upcoming departures with static offsets)
    for route, sched_deps in schedule.items():
        if route not in routes_data:
            continue
        route_stops = routes_data[route].get("STOPS", [])
        if not route_stops:
            continue

        for dep in sched_deps:
            if (route, dep.isoformat()) in assigned_scheduled_times:
                continue
            trip_id = f"{route}:{dep.isoformat()}"
            # Only add future scheduled trips within a reasonable window
            if dep < now_utc - timedelta(minutes=5):
                continue
            if dep > now_utc + timedelta(hours=2):
                continue

            # Build static ETAs from the route's OFFSET values
            stop_etas: Dict[str, Dict[str, Any]] = {}
            for stop_key in route_stops:
                stop_data = routes_data[route].get(stop_key, {})
                offset = stop_data.get("OFFSET")
                if offset is None:
                    stop_etas[stop_key] = {"eta": None, "last_arrival": None, "passed": False}
                    continue
                eta_dt = dep + timedelta(minutes=offset)
                stop_etas[stop_key] = {
                    "eta": eta_dt.isoformat() if eta_dt > now_utc else None,
                    "last_arrival": None,
                    "passed": eta_dt <= now_utc,
                }

            trips.append({
                "trip_id": trip_id,
                "route": route,
                "departure_time": dep.isoformat(),
                "actual_departure": None,
                "scheduled": True,
                "vehicle_id": None,
                "status": "scheduled" if dep > now_utc else "unassigned",
                "stop_etas": stop_etas,
            })

    # Sort: active/past trips first, then by departure time
    trips.sort(key=lambda t: (t["vehicle_id"] is None, t["departure_time"]))
    return trips
