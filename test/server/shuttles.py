"""Shuttle-related API routes and state management for the test server."""
import asyncio
import json
import logging
import os
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, and_

from backend.models import Vehicle, GeofenceEvent
from backend.time_utils import get_campus_start_of_day
from .shuttle import Shuttle, ShuttleAction
from .dev_time import dev_now, CAMPUS_TZ
from shared.stops import Stops

logger = logging.getLogger(__name__)

# CAMPUS_TZ imported from dev_time
SCHEDULE_PATH = Path(__file__).parent.parent.parent / "shared" / "aggregated_schedule.json"

# Global shuttle management
shuttles: dict[str, Shuttle] = {}
shuttle_counter = 1
shuttle_lock = asyncio.Lock()
route_names = Stops.active_routes
_scheduler_threads: list[threading.Thread] = []

# --- Break-simulation constants ---
# Lunch rotation window (campus time). Inside this window, Sun has 2 shuttles
# per route (1 on break, 1 covering) and Weekday has 3 (1 on break, 2 covering);
# outside this window the "extra" shuttles simply stop picking up new scheduled
# departures and idle out after finishing their current loop. Saturday always
# has 1 shuttle so the schedule gap IS the break — no ON_BREAK is queued.
# Chosen to span typical RPI dining hours and to cover the 12:00 PM -> 1:30 PM
# aggregated-schedule gap visible on day 6 (Sat) WEST.
LUNCH_WINDOW_START = (11, 30)  # (hour, minute), 11:30 AM campus time
LUNCH_WINDOW_END = (14, 30)    # 2:30 PM campus time

# How long a single break lasts in seconds (~45 min, matches D-02).
# Override via DEV_BREAK_DURATION_SEC env var for faster iteration.
BREAK_DURATION_SEC = int(os.environ.get("DEV_BREAK_DURATION_SEC", "2700"))

router = APIRouter(prefix="/api", tags=["shuttles"])


def stop_all_shuttles():
    """Stop all running shuttles."""
    for shuttle in shuttles.values():
        shuttle.stop()
    shuttles.clear()


def reset_shuttle_counter():
    """Reset the shuttle counter to 1."""
    global shuttle_counter
    shuttle_counter = 1


async def setup_shuttles(session_factory):
    """Populate the shuttles dict from database and start them."""
    async with session_factory() as db:
        start_of_today = get_campus_start_of_day()

        # Subquery: latest geofence event today per vehicle
        latest_geofence_events = (
            select(
                GeofenceEvent.vehicle_id,
                func.max(GeofenceEvent.event_time).label("latest_time"),
            )
            .where(GeofenceEvent.event_time >= start_of_today)
            .group_by(GeofenceEvent.vehicle_id)
            .subquery()
        )

        # Join to get full geofence event rows where event is geofenceEntry
        geofence_entries = (
            select(GeofenceEvent.vehicle_id)
            .join(
                latest_geofence_events,
                and_(
                    GeofenceEvent.vehicle_id == latest_geofence_events.c.vehicle_id,
                    GeofenceEvent.event_time == latest_geofence_events.c.latest_time,
                ),
            )
            .where(GeofenceEvent.event_type == "geofenceEntry")
            .subquery()
        )

        # Get vehicles that are currently in geofence
        query = select(Vehicle).where(Vehicle.id.in_(select(geofence_entries.c.vehicle_id)))

        result = await db.execute(query)
        vehicles = result.scalars().all()

        # Create and start shuttles
        async with shuttle_lock:
            for vehicle in vehicles:
                shuttle_id = str(vehicle.id)
                shuttle = Shuttle(shuttle_id)
                shuttle.start()
                shuttles[shuttle_id] = shuttle


@router.get("/shuttles")
async def list_shuttles():
    """List all active shuttles as a dictionary keyed by shuttle ID."""
    async with shuttle_lock:
        return {shuttle_id: s.to_dict() for shuttle_id, s in shuttles.items()}


@router.post("/shuttles")
async def create_shuttle():
    """Create a new test shuttle."""
    global shuttle_counter
    async with shuttle_lock:
        shuttle_id = str(shuttle_counter).zfill(15)
        shuttle = Shuttle(shuttle_id)
        shuttle.start()
        shuttles[shuttle_id] = shuttle
        logger.info(f"Created shuttle {shuttle_counter}")
        shuttle_counter += 1
        return JSONResponse(shuttle.to_dict(), status_code=201)


@router.post("/shuttles/{shuttle_id}/queue")
async def add_to_queue(shuttle_id: str, request: Request):
    """Add one or more actions to a shuttle's queue."""
    data = await request.json()

    # Accept either a single action or a list of actions
    actions = data.get("actions", [])
    if not actions:
        # Legacy single action format
        action_str = data.get("action")
        if action_str:
            actions = [{"action": action_str, "route": data.get("route"), "duration": data.get("duration")}]

    if not actions:
        raise HTTPException(status_code=400, detail="No actions provided")

    async with shuttle_lock:
        shuttle = shuttles.get(shuttle_id)
        if not shuttle:
            raise HTTPException(status_code=404, detail="Shuttle not found")

        for item in actions:
            action_str = item.get("action")
            route = item.get("route")
            duration = item.get("duration")

            try:
                action = ShuttleAction(action_str)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid action: {action_str}")

            try:
                shuttle.push_action(action, route=route, duration=duration)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            logger.info(f"Queued action {action_str} for shuttle {shuttle_id}")

        return shuttle.to_dict()


@router.get("/shuttles/{shuttle_id}/queue")
async def get_shuttle_queue(shuttle_id: str):
    """Get a shuttle's action queue."""
    async with shuttle_lock:
        shuttle = shuttles.get(shuttle_id)
        if not shuttle:
            raise HTTPException(status_code=404, detail="Shuttle not found")
        return {
            "shuttle_id": shuttle_id,
            "action_index": shuttle.action_index,
            "queue": shuttle.get_queue(),
        }


@router.delete("/shuttles/{shuttle_id}/queue")
async def clear_shuttle_queue(shuttle_id: str):
    """Clear a shuttle's pending actions."""
    async with shuttle_lock:
        shuttle = shuttles.get(shuttle_id)
        if not shuttle:
            raise HTTPException(status_code=404, detail="Shuttle not found")
        shuttle.clear_queue()
        return {"message": "Queue cleared"}


@router.post("/shuttles/schedule")
async def start_schedule_shuttles(strict: bool = False):
    """Start schedule-based shuttles (2 per route, alternating departure times).

    Query params:
        strict: when true, each shuttle runs exactly ONE loop per
            scheduled departure time (waiting idle at Union between
            slots). When false (default), shuttles start looping
            immediately and repeat continuously — useful for iterative
            UX testing but doesn't mirror real schedule behavior.
    """
    count = await setup_schedule_shuttles(strict=strict)
    async with shuttle_lock:
        return {
            "shuttles": count,
            "ids": list(shuttles.keys()),
            "strict": strict,
            "message": f"Created {count} schedule-based shuttles"
                + (" (strict)" if strict else ""),
        }


# --- Schedule-based shuttle management ---


def _load_today_schedule() -> dict[str, list[str]]:
    """Load today's route schedule from aggregated_schedule.json."""
    with open(SCHEDULE_PATH) as f:
        schedule = json.load(f)
    now = dev_now(CAMPUS_TZ)
    # aggregated_schedule indexed by JS getDay() (0=Sun)
    # Python weekday(): Mon=0..Sun=6  →  JS: (py+1)%7
    js_day = (now.weekday() + 1) % 7
    return schedule[js_day]


def _parse_schedule_time(time_str: str) -> datetime:
    """Parse '9:00 AM' into today's datetime in campus timezone."""
    now = dev_now(CAMPUS_TZ)
    parsed = datetime.strptime(time_str, "%I:%M %p")
    result = now.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
    # "12:00 AM" means midnight → next day
    if time_str.strip() == "12:00 AM":
        result += timedelta(days=1)
    return result


def _fleet_size_for_today() -> tuple[int, int]:
    """Return (peak_fleet_size, base_fleet_size) for today per D-03.

    peak_fleet_size is the count during the lunch rotation window.
    base_fleet_size is the count outside that window (extras idle out after
    their current loop finishes, since they won't match any more future
    scheduled departures once we trim the split below).

    Mapping (JS getDay semantics, matching _load_today_schedule):
      Sat (6)      -> 1, 1      -- single shuttle all day, schedule gap = break
      Sun (0)      -> 2, 1      -- 2 during lunch, 1 otherwise
      Weekday 1-5  -> 3, 1      -- 3 during lunch, 1 otherwise
    """
    now = dev_now(CAMPUS_TZ)
    js_day = (now.weekday() + 1) % 7
    if js_day == 6:       # Saturday
        return 1, 1
    if js_day == 0:       # Sunday
        return 2, 1
    return 3, 1           # Mon-Fri


def _lunch_window_today() -> tuple[datetime, datetime]:
    """Return (window_start, window_end) as campus-tz datetimes for today."""
    now = dev_now(CAMPUS_TZ)
    start = now.replace(
        hour=LUNCH_WINDOW_START[0], minute=LUNCH_WINDOW_START[1],
        second=0, microsecond=0,
    )
    end = now.replace(
        hour=LUNCH_WINDOW_END[0], minute=LUNCH_WINDOW_END[1],
        second=0, microsecond=0,
    )
    return start, end


def _schedule_worker(
    shuttle: Shuttle,
    route: str,
    departure_times: list[datetime],
    strict: bool = False,
    fleet_size: int = 1,
    shuttle_slot: int = 0,
):
    """Background thread: queues LOOPING at each scheduled departure and, on
    multi-shuttle days, queues ON_BREAK round-robin during the lunch window
    so one shuttle per route is on break at a time.

    Args:
        fleet_size: total shuttles on this route today during the lunch
            rotation window. fleet_size<=1 skips BREAK queuing entirely
            (Saturday + non-lunch windows rely on schedule gaps).
        shuttle_slot: this shuttle's 0-indexed slot within the rotation.
            Determines when during the lunch window this shuttle breaks.

    When strict=True, each queued LOOPING is single_loop=True so the
    shuttle runs exactly one loop and then returns to idle at Union,
    waiting for the next scheduled departure. When strict=False, the
    queued actions are redundant (the shuttle is already looping
    continuously) but kept for logging symmetry.
    """
    # --- Queue break(s) for this shuttle's rotation slot ---
    if fleet_size >= 2:
        window_start, window_end = _lunch_window_today()
        total_window = (window_end - window_start).total_seconds()
        if total_window > 0:
            # Round-robin: divide the window into fleet_size slots; this
            # shuttle breaks in its slot. Slot length must be >= break
            # duration; if not, shorten the break to the slot length so
            # we don't extend past the window.
            slot_len = total_window / fleet_size
            break_start = window_start + timedelta(seconds=slot_len * shuttle_slot)
            if break_start > dev_now(CAMPUS_TZ):
                break_duration = min(BREAK_DURATION_SEC, int(slot_len))
                # Schedule a BREAK action to fire at break_start. We sleep
                # in a helper thread until break_start then push so the
                # BREAK lands in the queue at the right time (otherwise
                # it would execute immediately after any already-queued
                # LOOPING, not at the intended rotation slot).
                def _break_pusher(sh=shuttle, rt=route,
                                  dur=break_duration, when=break_start):
                    wait = (when - dev_now(CAMPUS_TZ)).total_seconds()
                    if wait > 0:
                        time.sleep(wait)
                    if sh._running:
                        sh.push_action(
                            ShuttleAction.ON_BREAK,
                            route=rt,
                            duration=dur,
                        )
                        logger.info(
                            f"Shuttle {sh.id}: rotation break queued on "
                            f"{rt} at {when.strftime('%I:%M %p')} "
                            f"(slot {shuttle_slot + 1}/{fleet_size}, "
                            f"{dur:.0f}s)"
                        )
                break_thread = threading.Thread(
                    target=_break_pusher, daemon=True,
                    name=f"break-{shuttle.id}-{route}",
                )
                break_thread.start()
                _scheduler_threads.append(break_thread)

    # --- Existing behaviour: queue LOOPING at each scheduled departure ---
    for dep_time in departure_times:
        now = dev_now(CAMPUS_TZ)
        wait = (dep_time - now).total_seconds()
        if wait > 0:
            time.sleep(wait)
        if not shuttle._running:
            break
        shuttle.push_action(
            ShuttleAction.LOOPING,
            route=route,
            single_loop=strict,
        )
        logger.info(
            f"Shuttle {shuttle.id}: scheduled departure on {route} at "
            f"{dep_time.strftime('%I:%M %p')}"
            + (" (strict single-loop)" if strict else "")
        )


async def setup_schedule_shuttles(strict: bool = False) -> int:
    """Create 2 shuttles per route with alternating departure times from the static schedule.

    Args:
        strict: when True, shuttles wait idle at Union between scheduled
            departures and run exactly one loop per slot. When False
            (legacy), shuttles start looping continuously regardless
            of schedule times — still useful for fast UX iteration.
    """
    global shuttle_counter

    schedule = _load_today_schedule()
    now = dev_now(CAMPUS_TZ)

    async with shuttle_lock:
        # Tear down any existing shuttles
        stop_all_shuttles()
        for t in _scheduler_threads:
            t.join(timeout=0.1)
        _scheduler_threads.clear()
        shuttle_counter = 1

        for route_name in sorted(schedule.keys()):
            times = schedule[route_name]
            dep_times = [_parse_schedule_time(t) for t in times]
            future_deps = [t for t in dep_times if t > now]

            # End-of-day fallback: if the static schedule has no
            # future departures today (e.g. testing after the last
            # scheduled run), synthesize a pair of departures at
            # now+30s and now+5min so iteration testing still works.
            # Only triggers when fewer than 2 future slots remain.
            if len(future_deps) < 2:
                logger.warning(
                    f"Route {route_name}: only {len(future_deps)} future "
                    f"departures in schedule, synthesizing two immediate slots "
                    f"for testing"
                )
                synthetic_base = now + timedelta(seconds=30)
                future_deps = [
                    synthetic_base,
                    synthetic_base + timedelta(minutes=5),
                ]

            # Per-day fleet sizing (D-03): Sat=1, Sun=2-lunch/1-other,
            # Weekday=3-lunch/1-other. During the lunch window we spawn
            # peak_fleet shuttles (one of which breaks at a time, round-robin);
            # outside the window we spawn only base_fleet, letting extras idle
            # out naturally by not spawning them at all.
            peak_fleet, base_fleet = _fleet_size_for_today()
            window_start, window_end = _lunch_window_today()
            now_campus = dev_now(CAMPUS_TZ)
            in_lunch = window_start <= now_campus <= window_end
            fleet_size = peak_fleet if in_lunch else base_fleet

            # Stagger offset: first gap between scheduled departures / fleet_size
            # so shuttles spread through the loop rather than clustering.
            su_coords = (42.730711, -73.676737)
            if len(future_deps) >= 2 and fleet_size >= 2:
                first_gap_sec = (future_deps[1] - future_deps[0]).total_seconds()
                stagger_sec = max(60, int(first_gap_sec / fleet_size))
            else:
                stagger_sec = 300

            # Split departures N-ways by interleaving: shuttle i gets
            # future_deps[i::N]. peak_fleet is used for the workers' break
            # scheduling (round-robin slot math assumes this many shuttles
            # share the lunch window), but only fleet_size shuttles are
            # spawned right now — if in_lunch is False and peak > base, the
            # extras simply don't spawn and naturally don't break.
            for slot in range(fleet_size):
                shuttle_deps = future_deps[slot::fleet_size]
                if not shuttle_deps:
                    continue

                shuttle_id = str(shuttle_counter).zfill(15)
                shuttle = Shuttle(shuttle_id)
                shuttle._location = su_coords
                shuttle._send_webhook(entry=True)

                if not strict:
                    if slot == 0:
                        shuttle.push_action(ShuttleAction.LOOPING, route=route_name)
                    else:
                        def delayed_start(sh=shuttle, rn=route_name, offset=stagger_sec * slot):
                            sh.push_action(ShuttleAction.LOOPING, route=rn)
                        t = threading.Timer(stagger_sec * slot, delayed_start)
                        t.daemon = True
                        t.start()

                shuttle.start()
                shuttles[shuttle_id] = shuttle
                shuttle_counter += 1

                thread = threading.Thread(
                    target=_schedule_worker,
                    args=(shuttle, route_name, shuttle_deps, strict),
                    kwargs={"fleet_size": peak_fleet, "shuttle_slot": slot},
                    daemon=True,
                    name=f"sched-{shuttle_id}-{route_name}",
                )
                thread.start()
                _scheduler_threads.append(thread)

                logger.info(
                    f"Shuttle {shuttle_id}: {route_name}, "
                    f"slot {slot + 1}/{fleet_size} "
                    f"(peak_fleet={peak_fleet}), "
                    f"{len(shuttle_deps)} departures, "
                    f"first at {shuttle_deps[0].strftime('%I:%M %p')}"
                    + (" [strict]" if strict else "")
                )

    count = len(shuttles)
    logger.info(f"Created {count} schedule-based shuttles" + (" (strict)" if strict else ""))
    return count


@router.get("/routes")
async def get_routes():
    """Get list of available routes."""
    return sorted(list(route_names))
