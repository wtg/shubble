"""Utility functions for database queries."""
from sqlalchemy import func, and_, select
from fastapi_cache.decorator import cache
from typing import Dict, Tuple, Optional, List
import pandas as pd

from backend.models import GeofenceEvent
from backend.time_utils import get_campus_start_of_day
from backend.cache_dataframe import get_today_dataframe


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


@cache(expire=60, namespace="smart_closest_point")
async def smart_closest_point(
    vehicle_ids: List[str]
) -> Dict[str, Tuple[Optional[float], Optional[Tuple[float, float]], Optional[str], Optional[int], Optional[int], Optional[str]]]:
    """
    Get the closest point data for each vehicle from the cached dataframe (cached for 60 seconds).

    The cached dataframe (from get_today_dataframe) already contains preprocessed
    route matching data from the ML pipeline. This function simply retrieves the
    latest row for each vehicle and extracts the relevant columns.

    Args:
        vehicle_ids: List of vehicle IDs to get closest point data for

    Returns:
        Dictionary mapping vehicle_id to (distance, closest_point, route_name, polyline_idx, segment_idx, stop_name)
        Returns (None, None, None, None, None, None) for vehicles not found in the cache.
    """
    results = {}

    try:
        # Load cached dataframe with preprocessed route information
        df = await get_today_dataframe()

        if df.empty:
            # No cached data, return None for all vehicles
            for vehicle_id in vehicle_ids:
                results[vehicle_id] = (None, None, None, None, None, None)
            return results

        # Ensure vehicle_id is string type for comparison
        df['vehicle_id'] = df['vehicle_id'].astype(str)

        # Get the latest row for each vehicle
        for vehicle_id in vehicle_ids:
            vehicle_data = df[df['vehicle_id'] == str(vehicle_id)]

            if vehicle_data.empty:
                # No data for this vehicle
                results[vehicle_id] = (None, None, None, None, None, None)
                continue

            # Get the latest row (dataframe is sorted by timestamp)
            latest = vehicle_data.iloc[-1]

            # Extract closest point data from the preprocessed columns
            # These columns are added by ml/data/preprocess.py pipeline
            distance = latest.get('dist_to_route')
            route_name = latest.get('route')
            polyline_idx = latest.get('polyline_idx')
            segment_idx = latest.get('segment_idx')
            closest_lat = latest.get('closest_lat')
            closest_lon = latest.get('closest_lon')
            stop_name = latest.get('stop_name')  # From stops pipeline

            # Build closest_point tuple if coordinates are available
            if closest_lat is not None and closest_lon is not None:
                closest_point = (float(closest_lat), float(closest_lon))
            else:
                closest_point = None

            # Convert polyline_idx and segment_idx to int if they exist
            if polyline_idx is not None:
                try:
                    polyline_idx = int(polyline_idx)
                except (ValueError, TypeError):
                    polyline_idx = None

            if segment_idx is not None:
                try:
                    segment_idx = int(segment_idx)
                except (ValueError, TypeError):
                    segment_idx = None

            # Convert stop_name to string or None
            if stop_name is not None and not pd.isna(stop_name):
                stop_name = str(stop_name)
            else:
                stop_name = None

            results[vehicle_id] = (distance, closest_point, route_name, polyline_idx, segment_idx, stop_name)

    except Exception as e:
        # If anything goes wrong, return None for all vehicles
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in smart_closest_point: {e}")

        for vehicle_id in vehicle_ids:
            results[vehicle_id] = (None, None, None, None, None, None)

    return results
