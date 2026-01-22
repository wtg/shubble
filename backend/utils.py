"""Utility functions for database queries."""
from sqlalchemy import func, and_, select

from backend.cache import cache
from backend.models import GeofenceEvent
from backend.time_utils import get_campus_start_of_day


def get_vehicles_in_geofence_query():
    """
    Returns a query for vehicle_ids where the latest geofence event from today
    is a geofenceEntry.

    Returns:
        SQLAlchemy select query that returns vehicle IDs currently in the geofence
    """
    start_of_today = get_campus_start_of_day()

    # Subquery to get latest event per vehicle from today's events
    subquery = (
        select(
            GeofenceEvent.vehicle_id,
            func.max(GeofenceEvent.event_time).label("latest_time"),
        )
        .where(GeofenceEvent.event_time >= start_of_today)
        .group_by(GeofenceEvent.vehicle_id)
        .subquery()
    )

    # Join back to get the latest event row where type is entry
    query = (
        select(GeofenceEvent.vehicle_id)
        .join(
            subquery,
            and_(
                GeofenceEvent.vehicle_id == subquery.c.vehicle_id,
                GeofenceEvent.event_time == subquery.c.latest_time,
            ),
        )
        .where(GeofenceEvent.event_type == "geofenceEntry")
    )

    return query


@cache(expire=900, namespace="vehicles_in_geofence")
async def get_vehicles_in_geofence(session_factory):
    """
    Returns a cached set of vehicle_ids where the latest geofence event from today
    is a geofenceEntry.

    This function executes the query and caches the result for 5 seconds.

    Args:
        session_factory: Async session factory for creating database sessions

    Returns:
        Set of vehicle IDs currently in the geofence
    """
    async with session_factory() as session:
        query = get_vehicles_in_geofence_query()
        result = await session.execute(query)
        rows = result.all()
        return {row.vehicle_id for row in rows}