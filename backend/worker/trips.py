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


def _load_today_schedule(campus_tz) -> Dict[str, List[datetime]]:
    """Load today's schedule and parse departure times to datetimes.

    Returns:
        Dict mapping route_name to sorted list of departure datetimes (UTC).
    """
    try:
        with open(SCHEDULE_PATH) as f:
            schedule = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load schedule: {e}")
        return {}

    now = dev_now(campus_tz)
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
    return result


def _detect_vehicle_departures(
    vehicle_df: pd.DataFrame,
    first_stop: str,
) -> List[datetime]:
    """Find all times this vehicle was detected at the first stop of its route.

    Each distinct detection (separated by a gap) represents a loop start.
    Returns a sorted list of departure timestamps (UTC).
    """
    if vehicle_df.empty or 'stop_name' not in vehicle_df.columns:
        return []

    first_stop_points = vehicle_df[vehicle_df['stop_name'] == first_stop].copy()
    if first_stop_points.empty:
        return []

    first_stop_points = first_stop_points.sort_values('timestamp')
    timestamps = []
    last_ts = None
    # Group consecutive first-stop detections; each cluster = one departure
    for _, row in first_stop_points.iterrows():
        ts = pd.to_datetime(row['timestamp']).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if last_ts is None or (ts - last_ts).total_seconds() > 120:
            timestamps.append(ts)
        last_ts = ts
    return timestamps


def assign_trip_to_vehicle(
    vehicle_id: str,
    vehicle_df: pd.DataFrame,
    route: str,
    first_stop: str,
    scheduled_departures: List[datetime],
    now_utc: datetime,
) -> Optional[Dict[str, Any]]:
    """Assign this vehicle to a trip based on its departure history.

    Logic:
    1. Find the vehicle's most recent loop start (last detection at first stop)
    2. Match to the nearest scheduled departure within MATCH_WINDOW_SEC
    3. If no match, inject a new trip with the actual departure time

    Returns:
        Trip dict with fields:
          trip_id, route, departure_time (ISO), scheduled (bool),
          vehicle_id, status
        Returns None if the vehicle hasn't started a loop yet.
    """
    departures = _detect_vehicle_departures(vehicle_df, first_stop)
    if departures:
        # Most recent departure is the current active trip start
        actual_departure = departures[-1]
    else:
        # Fallback: stop detection hasn't fired yet. Use the vehicle's
        # earliest timestamp as an approximate departure time so the trip
        # still appears in the UI.
        if vehicle_df.empty:
            return None
        earliest = pd.to_datetime(vehicle_df['timestamp'].min()).to_pydatetime()
        if earliest.tzinfo is None:
            earliest = earliest.replace(tzinfo=timezone.utc)
        actual_departure = earliest

    # Find the closest scheduled departure within the matching window
    matched_scheduled: Optional[datetime] = None
    min_diff = float('inf')
    for sched in scheduled_departures:
        diff = abs((sched - actual_departure).total_seconds())
        if diff < min_diff and diff <= MATCH_WINDOW_SEC:
            min_diff = diff
            matched_scheduled = sched

    if matched_scheduled is not None:
        trip_time = matched_scheduled
        scheduled = True
    else:
        # Injected trip — use the actual departure time
        trip_time = actual_departure
        scheduled = False

    # Status derivation
    if actual_departure > now_utc:
        status = "scheduled"
    else:
        status = "active"

    return {
        # Include vehicle_id so two vehicles matched to the same scheduled
        # departure each get a unique trip (no collision in frontend keys).
        "trip_id": f"{route}:{trip_time.isoformat()}:{vehicle_id}",
        "route": route,
        "departure_time": trip_time.isoformat(),
        "actual_departure": actual_departure.isoformat(),
        "scheduled": scheduled,
        "vehicle_id": vehicle_id,
        "status": status,
    }


def build_trip_etas(
    trip: Dict[str, Any],
    vehicle_stops: List,  # List[Tuple[stop_key, eta_datetime]]
    last_arrivals: Dict[str, str],
    stops_in_route: List[str],
    now_utc: datetime,
) -> Dict[str, Dict[str, Any]]:
    """Build per-stop ETA dict for a single trip.

    Each stop on the route gets an entry. Stops behind the shuttle show
    `last_arrival`, stops ahead show future `eta`.

    Args:
        trip: Trip dict from assign_trip_to_vehicle
        vehicle_stops: List of (stop_key, eta_datetime) from
            compute_per_stop_etas per-vehicle computation
        last_arrivals: Dict of stop_key -> ISO timestamp
        stops_in_route: Full STOPS list for the route
        now_utc: Current time

    Returns:
        Dict mapping stop_key to {eta, last_arrival, passed} entry.
    """
    stop_etas: Dict[str, Dict[str, Any]] = {}

    # Build a lookup for future ETAs from vehicle_stops
    eta_lookup = {}
    for stop_key, eta_dt in vehicle_stops:
        eta_lookup[stop_key] = eta_dt

    for stop_key in stops_in_route:
        entry: Dict[str, Any] = {
            "eta": None,
            "last_arrival": None,
            "passed": False,
        }

        if stop_key in eta_lookup:
            eta_dt = eta_lookup[stop_key]
            if eta_dt > now_utc:
                entry["eta"] = eta_dt.isoformat()
            else:
                # ETA is in the past — treat as passed, use expired ETA
                # as implied arrival time if no better data exists
                entry["passed"] = True
                entry["last_arrival"] = eta_dt.isoformat()

        if stop_key in last_arrivals:
            la_iso = last_arrivals[stop_key]
            # Prefer the detection-based last_arrival if it's newer
            current = entry.get("last_arrival")
            if current is None or la_iso > current:
                entry["last_arrival"] = la_iso
                entry["passed"] = True

        stop_etas[stop_key] = entry

    # Fix detection gaps + enforce monotonic last_arrivals. Passed stops
    # should have last_arrival timestamps in ascending order within a
    # single loop — a shuttle can't visit a later stop BEFORE an earlier
    # one. If the order is violated, it's because the detection matched
    # a stop from a PREVIOUS loop. Clamp the out-of-order stop's
    # last_arrival up to the preceding stop's value so the display
    # stays coherent.
    last_passed_idx = -1
    last_passed_la = None
    for i, stop_key in enumerate(stops_in_route):
        entry = stop_etas[stop_key]
        if entry["passed"]:
            # Enforce monotonic ordering: if this stop's la is earlier
            # than the previous passed stop's la, use the previous.
            if last_passed_la and entry["last_arrival"] and entry["last_arrival"] < last_passed_la:
                entry["last_arrival"] = last_passed_la
            last_passed_idx = i
            last_passed_la = entry["last_arrival"]
            continue
        # Not passed — check if we should backfill (detection gap)
        if entry["eta"] is None and last_passed_idx >= 0:
            # Gap: no ETA, no last_arrival, but earlier stops are passed
            entry["passed"] = True
            entry["last_arrival"] = last_passed_la

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

        if full_df is not None and not full_df.empty:
            vehicle_df = full_df[full_df['vehicle_id'] == str(vid)].sort_values('timestamp')
        else:
            vehicle_df = pd.DataFrame()

        # Compute actual_departure without matching
        departures = _detect_vehicle_departures(vehicle_df, first_stop)
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
            if 'latitude' in vehicle_df.columns and 'longitude' in vehicle_df.columns:
                latest_ts = pd.to_datetime(last_point['timestamp']).to_pydatetime()
                if latest_ts.tzinfo is None:
                    latest_ts = latest_ts.replace(tzinfo=timezone.utc)
                lookback_cutoff = latest_ts - timedelta(seconds=NO_MOVEMENT_LOOKBACK_SEC)
                vehicle_df_ts = pd.to_datetime(vehicle_df['timestamp'], utc=True)
                old_points = vehicle_df[vehicle_df_ts <= lookback_cutoff]
                if not old_points.empty:
                    old_point = old_points.iloc[-1]  # most recent point >= lookback ago
                    try:
                        from shared.stops import Stops
                        latest_lat = float(last_point['latitude'])
                        latest_lon = float(last_point['longitude'])
                        old_lat = float(old_point['latitude'])
                        old_lon = float(old_point['longitude'])
                        # Reuse haversine via Stops if available, else inline.
                        import math
                        R = 6371000
                        phi1 = math.radians(old_lat)
                        phi2 = math.radians(latest_lat)
                        dphi = math.radians(latest_lat - old_lat)
                        dlam = math.radians(latest_lon - old_lon)
                        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
                        dist_moved = 2 * R * math.asin(math.sqrt(a))
                        if dist_moved < NO_MOVEMENT_DIST_M:
                            logger.debug(
                                f"Skipping trip for vehicle {vid} on {route}: "
                                f"moved only {dist_moved:.0f}m in last "
                                f"{NO_MOVEMENT_LOOKBACK_SEC}s"
                            )
                            continue
                    except Exception:
                        pass  # On any error, fall through and emit the trip
        # ------------------------------------------------------------

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

        # Find the closest UNCLAIMED scheduled time within the window
        matched_scheduled: Optional[datetime] = None
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

        status = "scheduled" if actual_departure > now_utc else "active"

        # Trip completion: if the vehicle has arrived at the LAST stop of
        # the route recently, the current loop is done. Mark it completed
        # so the frontend stops showing ETAs for a phantom "next loop".
        # The vehicle will start a new trip once it physically departs
        # again and moves past the first stop.
        vehicle_las = last_arrivals_by_vehicle.get(str(vid), {})
        last_stop = route_stops[-1]
        vehicle_future_stops = vdata.get("stops", [])
        if status == "active" and last_stop in vehicle_las:
            last_stop_arrival = datetime.fromisoformat(vehicle_las[last_stop])
            # If the vehicle reached the last stop within 10 minutes,
            # it just finished a loop. Mark completed.
            if (now_utc - last_stop_arrival).total_seconds() < 600:
                status = "completed"
        # Also mark completed if the vehicle has NO future stops at all
        # (it's at or past the last stop in the ETA computation).
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

        stop_etas = build_trip_etas(
            trip=trip,
            vehicle_stops=vehicle_future_stops,
            last_arrivals=vehicle_las,
            stops_in_route=route_stops,
            now_utc=now_utc,
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
                completed_trip = {
                    "trip_id": f"{route}:{prior_departure.isoformat()}:{vid}:done",
                    "route": route,
                    "departure_time": prior_departure.isoformat(),
                    "actual_departure": prior_departure.isoformat(),
                    "scheduled": False,
                    "vehicle_id": vid,
                    "status": "completed",
                }
                # All stops passed — use arrival timestamps or synthetic ones
                completed_stop_etas: Dict[str, Dict[str, Any]] = {}
                for stop_key in route_stops:
                    la = vehicle_las.get(stop_key)
                    completed_stop_etas[stop_key] = {
                        "eta": None,
                        "last_arrival": la,
                        "passed": True,
                    }
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
