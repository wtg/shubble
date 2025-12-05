import pandas as pd
import numpy as np
import json
import logging
from scipy.optimize import linear_sum_assignment
from server.models import VehicleLocation
from data.stops import Stops
from datetime import datetime, timezone
from server.time_utils import get_campus_start_of_day
from server import db, cache

logger = logging.getLogger(__name__)

class Schedule:

    @classmethod
    def get_stop_info(cls, row):
        """
        Calculate whether Vehicle is at Stop. Return (route_name, stop_name)
        or (None, None) if not at stop using Redis to cache repeated coordinate lookups.
        """
        coords = (float(row.latitude), float(row.longitude))
        key = f"coords:{coords[0]}:{coords[1]}"

        # Try Redis cache first
        cached = cache.get(key)
        if cached:
            return cached

        try:
            route_name, stop_name = Stops.is_at_stop(coords)
        except Exception:
            route_name, stop_name = None, None

        # Store in Redis
        cache.set(key, (route_name, stop_name), timeout=60 * 60 * 24)
        return route_name, stop_name

    @classmethod
    def load_and_label_stops(cls):
        """
        Load DB rows and add new columns route_name and stop_name using is_at_stop().
        Uses Redis to cache labeled stop results
        """
        start = get_campus_start_of_day()
        now = datetime.now(timezone.utc)

        cache_key = f"labeled_stops:{start.date()}"
        cached_df = cache.get(cache_key)
        if cached_df is not None:
            return cached_df

        rows = (
            db.session.query(VehicleLocation)
            .filter(
                VehicleLocation.timestamp >= start,
                VehicleLocation.timestamp <= now
            )
            .order_by(VehicleLocation.timestamp.asc())
            .all()
        )

        if not rows:
            empty_df = pd.DataFrame(columns=['vehicle_id','timestamp','route_name','stop_name'])
            cache.set(cache_key, empty_df, timeout=60 * 10)
            return empty_df

        df = pd.DataFrame([
            {
                'vehicle_id': r.vehicle_id,
                'timestamp': r.timestamp,
                'latitude': float(r.latitude) if r.latitude is not None else None,
                'longitude': float(r.longitude) if r.longitude is not None else None
            }
            for r in rows
        ])

        # Add route_name and stop_name columns
        df[['route_name','stop_name']] = df.apply(
            lambda r: pd.Series(cls.get_stop_info(r)), axis=1
        )

        labeled = df.dropna(subset=['route_name','stop_name'])[
            ['vehicle_id','timestamp','route_name','stop_name']
        ].copy()

        # Cache for the duration of the day
        cache.set(cache_key, labeled, timeout=60 * 10)
        return labeled

    @classmethod
    def match_shuttles_to_schedules(cls):
        """
        Match shuttle vehicle data to the most likely schedule route
        based on timestamp and stop location using the Hungarian algorithm.

        Redis caching prevents recomputation for one hour.
        """
        cached = cache.get("schedule_entries")
        if cached is not None:
            return cached

        at_stops = cls.load_and_label_stops()

        sched = Stops.schedule_data

        # Determine day from first timestamp
        required_cols = {"vehicle_id", "timestamp", "route_name"}

        if at_stops.empty or not required_cols.issubset(at_stops.columns):
            logger.warning("at_stops is missing required data returning empty match.")
            return {}

        first_ts = pd.to_datetime(at_stops['timestamp'].iloc[0])
        day_name = first_ts.day_name().upper()
        day_key = sched[day_name]

        #Get routes for the day
        routes = sched[day_key]

        #Get Unique shuttles that are at stops
        shuttles = at_stops['vehicle_id'].unique().tolist()

        # Expand schedule with actual date
        date_str = first_ts.strftime("%Y-%m-%d")
        sched_flat = [
            (
                name,
                [(pd.to_datetime(f"{date_str} {t}"), s) for t, s in times]
            )
            for name, times in routes.items()
        ]

        W = np.zeros((len(shuttles), len(sched_flat)))

        # Precompute minute-aligned timestamps
        at_stops['minute'] = at_stops['timestamp'].dt.floor('min')

        # Group logs
        shuttle_groups = {k: v for k, v in at_stops.groupby('vehicle_id')}

        # Build cost matrix
        for i, shuttle in enumerate(shuttles):
            logs = shuttle_groups[shuttle]
            log_pairs = set(zip(logs['route_name'], logs['minute']))

            for j, (_, stops) in enumerate(sched_flat):
                # Build schedule pairs
                sched_pairs = {(stop_name, time_stamp) for time_stamp, stop_name in stops}

                # Compute matches
                matches = len(log_pairs & sched_pairs)

                # Cost = 0 means perfect match, 1 means no match
                cost = 1 - (matches / len(stops))
                W[i, j] = cost

        # Hungarian algorithm
        row, col = linear_sum_assignment(W)

        # Generate result dictionary
        result = {shuttles[r]: sched_flat[c][0] for r, c in zip(row, col)}

        # Cache results for 1 hour
        cache.set("schedule_entries", result, timeout=3600)

        return result
    
if __name__ == "__main__":
    result = Schedule.match_shuttles_to_schedules()