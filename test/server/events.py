"""Event-related API routes for the test server."""
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, and_

from backend.models import GeofenceEvent, VehicleLocation
from backend.time_utils import get_campus_start_of_day
from .shuttles import shuttle_lock, shuttles, stop_all_shuttles, reset_shuttle_counter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("/today")
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


@router.delete("/today")
async def clear_events_today(request: Request, keepShuttles: bool = False):
    """Clear events from today."""
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

            # Stop and clear all shuttles
            async with shuttle_lock:
                stop_all_shuttles()
                reset_shuttle_counter()
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
