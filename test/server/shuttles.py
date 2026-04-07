"""Shuttle-related API routes and state management for the test server."""
import asyncio
import json
import logging
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
async def start_schedule_shuttles():
    """Start schedule-based shuttles (2 per route, alternating departure times)."""
    count = await setup_schedule_shuttles()
    async with shuttle_lock:
        return {
            "shuttles": count,
            "ids": list(shuttles.keys()),
            "message": f"Created {count} schedule-based shuttles",
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


def _schedule_worker(shuttle: Shuttle, route: str, departure_times: list[datetime]):
    """Background thread: queues a LOOPING action at each scheduled departure."""
    for dep_time in departure_times:
        now = dev_now(CAMPUS_TZ)
        wait = (dep_time - now).total_seconds()
        if wait > 0:
            time.sleep(wait)
        if not shuttle._running:
            break
        shuttle.push_action(ShuttleAction.LOOPING, route=route)
        logger.info(f"Shuttle {shuttle.id}: scheduled departure on {route} at {dep_time.strftime('%I:%M %p')}")


async def setup_schedule_shuttles() -> int:
    """Create 2 shuttles per route with alternating departure times from the static schedule."""
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

            if len(future_deps) < 2:
                logger.warning(f"Route {route_name}: fewer than 2 future departures, skipping")
                continue

            # Shuttle A gets even-indexed departures, Shuttle B gets odd-indexed
            for shuttle_deps in [future_deps[0::2], future_deps[1::2]]:
                if not shuttle_deps:
                    continue

                shuttle_id = str(shuttle_counter).zfill(15)
                shuttle = Shuttle(shuttle_id)
                shuttle.push_action(ShuttleAction.ENTERING)
                shuttle.start()
                shuttles[shuttle_id] = shuttle
                shuttle_counter += 1

                thread = threading.Thread(
                    target=_schedule_worker,
                    args=(shuttle, route_name, shuttle_deps),
                    daemon=True,
                    name=f"sched-{shuttle_id}-{route_name}",
                )
                thread.start()
                _scheduler_threads.append(thread)

                logger.info(
                    f"Shuttle {shuttle_id}: {route_name}, "
                    f"{len(shuttle_deps)} departures, "
                    f"first at {shuttle_deps[0].strftime('%I:%M %p')}"
                )

    count = len(shuttles)
    logger.info(f"Created {count} schedule-based shuttles")
    return count


@router.get("/routes")
async def get_routes():
    """Get list of available routes."""
    return sorted(list(route_names))
