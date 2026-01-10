"""FastAPI test server - Mock Samsara API for development/testing."""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, and_, select

from server.config import settings
from server.database import create_async_db_engine, create_session_factory
from server.models import Vehicle, GeofenceEvent, VehicleLocation
from server.time_utils import get_campus_start_of_day
from .shuttle import Shuttle, ShuttleState
from data.stops import Stops

# Global shuttle management
shuttles = {}
shuttle_counter = 1
shuttle_lock = asyncio.Lock()
route_names = Stops.active_routes

logger = logging.getLogger(__name__)


async def setup_shuttles(session_factory):
    """Populate the shuttles dict from database."""
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

        # Subquery: latest vehicle location per vehicle
        latest_locations = (
            select(
                VehicleLocation.vehicle_id,
                func.max(VehicleLocation.timestamp).label("latest_time"),
            )
            .where(VehicleLocation.vehicle_id.in_(select(geofence_entries.c.vehicle_id)))
            .group_by(VehicleLocation.vehicle_id)
            .subquery()
        )

        # Join to get full location and vehicle info for vehicles in geofence
        query = (
            select(VehicleLocation, Vehicle)
            .join(
                latest_locations,
                and_(
                    VehicleLocation.vehicle_id == latest_locations.c.vehicle_id,
                    VehicleLocation.timestamp == latest_locations.c.latest_time,
                ),
            )
            .join(Vehicle, VehicleLocation.vehicle_id == Vehicle.id)
        )

        result = await db.execute(query)
        results = result.all()

        # Extract vehicle information
        async with shuttle_lock:
            for loc, vehicle in results:
                shuttles[vehicle.id] = Shuttle(vehicle.id, loc.latitude, loc.longitude)


async def update_loop():
    """Background task to update shuttle states."""
    while True:
        await asyncio.sleep(0.1)
        async with shuttle_lock:
            for shuttle in shuttles.values():
                shuttle.update_state()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting test server...")

    # Initialize database
    app.state.db_engine = create_async_db_engine(
        settings.DATABASE_URL, echo=settings.DEBUG
    )
    app.state.session_factory = create_session_factory(app.state.db_engine)
    logger.info("Database initialized")

    # Setup shuttles from database
    await setup_shuttles(app.state.session_factory)
    logger.info(f"Initialized {len(shuttles)} shuttles from database")

    # Start background updater task
    app.state.update_task = asyncio.create_task(update_loop())
    logger.info("Background updater task started")

    yield

    # Shutdown
    logger.info("Shutting down test server...")
    app.state.update_task.cancel()
    try:
        await app.state.update_task
    except asyncio.CancelledError:
        pass
    await app.state.db_engine.dispose()
    logger.info("Test server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Mock Samsara API",
    description="Test server for Shubble development",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for test-client
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.TEST_FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API Routes ---
@app.get("/api/shuttles")
async def list_shuttles():
    """List all active shuttles."""
    async with shuttle_lock:
        return [s.to_dict() for s in shuttles.values()]


@app.post("/api/shuttles")
async def create_shuttle():
    """Create a new test shuttle."""
    global shuttle_counter
    async with shuttle_lock:
        shuttle_id = str(shuttle_counter).zfill(15)
        shuttle = Shuttle(shuttle_id)
        shuttles[shuttle_id] = shuttle
        logger.info(f"Created shuttle {shuttle_counter}")
        shuttle_counter += 1
        return JSONResponse(shuttle.to_dict(), status_code=201)


@app.post("/api/shuttles/{shuttle_id}/set-next-state")
async def trigger_action(shuttle_id: str, request: Request):
    """Set the next state for a shuttle."""
    data = await request.json()
    next_state = data.get("state")

    async with shuttle_lock:
        shuttle = shuttles.get(shuttle_id)
        if not shuttle:
            raise HTTPException(status_code=404, detail="Shuttle not found")

        try:
            desired_state = ShuttleState(next_state)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid action")

        shuttle.set_next_state(desired_state)
        if desired_state == ShuttleState.LOOPING:
            route = data.get("route")
            shuttle.set_next_route(route)

        logger.info(f"Set shuttle {shuttle_id} next state to {next_state}")
        return shuttle.to_dict()


@app.get("/api/routes")
async def get_routes():
    """Get list of available routes."""
    return sorted(list(route_names))


@app.get("/api/events/today")
async def get_events_today(request: Request):
    """Get count of events from today."""
    async with request.app.state.session_factory() as db:
        start_of_today = get_campus_start_of_day()

        loc_query = select(func.count()).select_from(VehicleLocation).where(
            VehicleLocation.timestamp >= start_of_today
        )
        geo_query = select(func.count()).select_from(GeofenceEvent).where(
            GeofenceEvent.event_time >= start_of_today
        )

        loc_result = await db.execute(loc_query)
        geo_result = await db.execute(geo_query)

        return {
            "locationCount": loc_result.scalar(),
            "geofenceCount": geo_result.scalar(),
        }


@app.delete("/api/events/today")
async def clear_events_today(request: Request, keepShuttles: bool = False):
    """Clear events from today."""
    global shuttle_counter
    async with request.app.state.session_factory() as db:
        start_of_today = get_campus_start_of_day()

        # Delete vehicle locations
        await db.execute(
            VehicleLocation.__table__.delete().where(
                VehicleLocation.timestamp >= start_of_today
            )
        )
        logger.info(f"Deleted vehicle location events past {start_of_today}")

        if not keepShuttles:
            # Delete all geofence events
            await db.execute(
                GeofenceEvent.__table__.delete().where(
                    GeofenceEvent.event_time >= start_of_today
                )
            )
            logger.info(f"Deleted geofence events past {start_of_today}")

            # Clear all shuttles
            async with shuttle_lock:
                shuttles.clear()
                shuttle_counter = 1
            logger.info("Deleted all shuttles")
        else:
            # Keep latest geofenceEntry for each vehicle
            today_events = (
                select(GeofenceEvent)
                .where(GeofenceEvent.event_time >= start_of_today)
                .subquery()
            )

            latest_times = (
                select(
                    today_events.c.vehicle_id,
                    func.max(today_events.c.event_time).label("latest_time"),
                )
                .group_by(today_events.c.vehicle_id)
                .subquery()
            )

            latest_entries = (
                select(today_events.c.id)
                .join(
                    latest_times,
                    and_(
                        today_events.c.vehicle_id == latest_times.c.vehicle_id,
                        today_events.c.event_time == latest_times.c.latest_time,
                    ),
                )
                .where(today_events.c.event_type == "geofenceEntry")
                .scalar_subquery()
            )

            # Delete events not in latest_entries
            await db.execute(
                GeofenceEvent.__table__.delete().where(
                    GeofenceEvent.event_time >= start_of_today,
                    ~GeofenceEvent.id.in_(latest_entries),
                )
            )
            logger.info(
                f"Deleted geofence events past {start_of_today} except for currently running shuttles"
            )

        await db.commit()
        return JSONResponse(content="", status_code=204)


# --- Mock Samsara API Endpoints ---
@app.get("/fleet/vehicles/stats")
async def mock_stats(vehicleIds: str = "", after: str = None):
    """Mock Samsara vehicle stats endpoint."""
    vehicle_ids = vehicleIds.split(",") if vehicleIds else []

    logger.info(
        f"[MOCK API] Received stats snapshot request for vehicles {vehicle_ids} after={after}"
    )

    async with shuttle_lock:
        data = []
        for shuttle_id in vehicle_ids:
            if shuttle_id in shuttles:
                # Add error to location
                lat, lon = shuttles[shuttle_id].location
                lat += np.random.normal(0, 0.00008)
                lon += np.random.normal(0, 0.00008)
                data.append(
                    {
                        "id": shuttle_id,
                        "name": shuttle_id[-3:],
                        "gps": {
                            "latitude": lat,
                            "longitude": lon,
                            "time": datetime.fromtimestamp(
                                shuttles[shuttle_id].last_updated
                            )
                            .isoformat(timespec="seconds")
                            .replace("+00:00", "Z"),
                            "speedMilesPerHour": shuttles[shuttle_id].speed,
                            "headingDegrees": 90,
                            "reverseGeo": {"formattedLocation": "Test Location"},
                        },
                    }
                )

        return {
            "data": data,
            "pagination": {"hasNextPage": False, "endCursor": "fake-token-next"},
        }


@app.get("/fleet/vehicles/stats/feed")
async def mock_feed(vehicleIds: str = "", after: str = None):
    """Mock Samsara vehicle stats feed endpoint."""
    vehicle_ids = vehicleIds.split(",") if vehicleIds else []

    logger.info(
        f"[MOCK API] Received stats feed request for vehicles {vehicle_ids} after={after}"
    )

    async with shuttle_lock:
        data = []
        for shuttle_id in vehicle_ids:
            if shuttle_id in shuttles:
                # Add error to location
                lat, lon = shuttles[shuttle_id].location
                lat += np.random.normal(0, 0.00008)
                lon += np.random.normal(0, 0.00008)
                data.append(
                    {
                        "id": shuttle_id,
                        "name": shuttle_id[-3:],
                        "gps": [
                            {
                                "latitude": lat,
                                "longitude": lon,
                                "time": datetime.fromtimestamp(
                                    shuttles[shuttle_id].last_updated
                                )
                                .isoformat(timespec="seconds")
                                .replace("+00:00", "Z"),
                                "speedMilesPerHour": shuttles[shuttle_id].speed,
                                "headingDegrees": 90,
                                "reverseGeo": {"formattedLocation": "Test Location"},
                            }
                        ],
                    }
                )

        return {
            "data": data,
            "pagination": {"hasNextPage": False, "endCursor": "fake-token-next"},
        }


@app.get("/fleet/driver-vehicle-assignments")
async def mock_driver_assignments(vehicleIds: str = ""):
    """Mock endpoint for driver-vehicle assignments."""
    vehicle_ids = vehicleIds.split(",") if vehicleIds else []

    logger.info(
        f"[MOCK API] Received driver-vehicle assignments request for vehicles {vehicle_ids}"
    )

    async with shuttle_lock:
        data = []
        for shuttle_id in vehicle_ids:
            if shuttle_id in shuttles:
                # Generate a mock driver for each active shuttle
                driver_id = f"driver-{shuttle_id[-3:]}"
                driver_name = f"Driver {shuttle_id[-3:]}"
                data.append(
                    {
                        "assignedAtTime": datetime.now()
                        .isoformat(timespec="seconds")
                        .replace("+00:00", "Z"),
                        "driver": {
                            "id": driver_id,
                            "name": driver_name,
                        },
                        "vehicle": {
                            "id": shuttle_id,
                        },
                    }
                )

        return {
            "data": data,
            "pagination": {"hasNextPage": False, "endCursor": "fake-token-next"},
        }


# --- Frontend Serving ---
@app.get("/")
@app.get("/{path:path}")
async def serve_frontend(path: str = ""):
    """Serve the test client frontend."""
    static_folder = Path(__file__).parent.parent / "test-client" / "dist"

    if path and (static_folder / path).exists():
        return FileResponse(static_folder / path)
    else:
        index_path = static_folder / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Frontend not built")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=4000)
