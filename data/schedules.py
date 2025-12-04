import pandas as pd
import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from server.models import VehicleLocation
from stops import Stops
from datetime import datetime, timezone
from server.time_utils import get_campus_start_of_day
from server import db


class Schedule:
    cache = {}

    @classmethod
    def get_stop_info(cls, row):
        """
        Calculate whether Vehicle is at Stop. Return (route_name, stop_name)
        or (None, None) if not at stop.
        """
        coords = (float(row.latitude), float(row.longitude))

        if coords in cls.cache:
            return cls.cache[coords]

        try:
            rn, sn = Stops.is_at_stop(coords)
        except Exception:
            rn, sn = None, None

        cls.cache[coords] = (rn, sn)
        return rn, sn

    @classmethod
    def load_and_label_stops(cls):
        """
        Load CSV and add new columns route_name and stop_name using is_at_stop().
        """
        start = get_campus_start_of_day()
        now = datetime.now(timezone.utc)

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
            return pd.DataFrame(columns=['vehicle_id','timestamp','route_name','stop_name'])

        df = pd.DataFrame([
            {
                'vehicle_id': r.vehicle_id,
                'timestamp': r.timestamp,
                'latitude': float(r.latitude) if r.latitude is not None else None,
                'longitude': float(r.longitude) if r.longitude is not None else None
            }
            for r in rows
        ])

        df[['route_name','stop_name']] = df.apply(
            lambda r: pd.Series(cls.get_stop_info(r)), axis=1
        )

        labeled = df.dropna(subset=['route_name','stop_name'])[
            ['vehicle_id','timestamp','route_name','stop_name']
        ].copy()

        return labeled

    @classmethod
    def match_shuttles_to_schedules(cls):
        """
        Match shuttle vehicle data to the most likely schedule route based on timestamp 
        and stop location using Hungarian algorithm.
        """
        at_stops = cls.load_and_label_stops()

        with open('data/schedule.json') as f:
            sched = json.load(f)

        # Determine the day name from the first timestamp
        first_ts = pd.to_datetime(at_stops['timestamp'].iloc[0])
        day_name = first_ts.day_name().upper()

        # Step 1: "MONDAY" → "weekday"
        day_key = sched[day_name]

        # Step 2: "weekday" → actual bus route dictionary
        routes = sched[day_key]

        # Unique shuttles
        shuttles = at_stops['vehicle_id'].unique().tolist()

        # Expand schedule with actual date
        date_str = first_ts.strftime("%Y-%m-%d")
        sched_flat = [
            (name, [(pd.to_datetime(f"{date_str} {t}"), s) for t, s in times])
            for name, times in routes.items()
        ]

        W = np.zeros((len(shuttles), len(sched_flat)))

        # Add minute column for faster matching
        at_stops['minute'] = at_stops['timestamp'].dt.floor('min')

        # Group logs by shuttle
        shuttle_groups = {k: v for k, v in at_stops.groupby('vehicle_id')}

        # Build cost matrix
        for i, shuttle in enumerate(shuttles):
            logs = shuttle_groups[shuttle]
            log_pairs = set(zip(logs['route_name'], logs['minute']))

            for j, (_, stops) in enumerate(sched_flat):
                sched_pairs = {(sn, ts) for ts, sn in stops}
                matches = len(log_pairs & sched_pairs)
                
                cost = 1 - (matches / len(stops))
                W[i, j] = cost

        # Run Hungarian algorithm
        row, col = linear_sum_assignment(W)

        # Final dict
        result = {shuttles[r]: sched_flat[c][0] for r, c in zip(row, col)}

        # Print assignments
        for r_idx, c_idx in zip(row, col):
            shuttle = shuttles[r_idx]
            route_label = sched_flat[c_idx][0]
            match_cost = W[r_idx, c_idx] * 100
            print(f"{route_label} = Shuttle {shuttle} with matching accuracy: {match_cost:.3f}%")

        return result


if __name__ == "__main__":
    result = Schedule.match_shuttles_to_schedules()