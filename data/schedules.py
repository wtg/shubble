import pandas as pd
import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from stops import Stops


def load_and_label_stops():
    df = pd.read_csv('data/shuttle.csv')

    route_names = []
    stop_names = []

    for _, row in df.iterrows():
        lat, lon = float(row['latitude']), float(row['longitude'])

        try:
            origin = np.array([[lat, lon]], dtype=float)
            route_name, stop_name = Stops.is_at_stop(origin)
        except:
            route_name, stop_name = None, None

        route_names.append(route_name)
        stop_names.append(stop_name)

    df['route_name'] = route_names
    df['stop_name'] = stop_names
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Only rows where we detected being at a stop
    return df.dropna(subset=['route_name', 'stop_name']).copy()


def resolve_day(date: pd.Timestamp, schedules: dict):
    """
    Based on timestamp weekday name, map to schedule group
    using the top mapping in schedule JSON.
    Example: "MONDAY" -> "weekday"
    """
    weekday_name = date.day_name().upper()
    if weekday_name in schedules:
        return schedules[weekday_name]
    return None


def build_weight_matrix(at_stops, schedule_json):
    # Determine which schedule block to use (weekday/weekend)
    first_ts = at_stops['timestamp'].iloc[0]
    schedule_key = resolve_day(first_ts, schedule_json)

    schedule_block = schedule_json[schedule_key]

    shuttles = at_stops['vehicle_id'].unique().tolist()

    # Flatten schedule shapes
    schedule_list = []
    for route_name, times in schedule_block.items():
        parsed = [(pd.to_datetime(t), stop) for t, stop in times]
        schedule_list.append((route_name, parsed))

    # cost matrix: rows=shuttles, cols=schedules
    W = np.zeros((len(shuttles), len(schedule_list)))

    for i, shuttle in enumerate(shuttles):
        logs = at_stops[at_stops['vehicle_id'] == shuttle]

        for j, (route_name, sched_times) in enumerate(schedule_list):
            matches = 0

            for sched_time, sched_stop in sched_times:
                mask = (
                    (logs['stop_name'] == sched_stop) &
                    (logs['timestamp'] == sched_time)  # exact match
                )
                if mask.any():
                    matches += 1

            similarity = matches / len(sched_times)
            W[i, j] = 1 - similarity  # Hungarian minimizes

    return shuttles, [name for name, _ in schedule_list], W


def match_shuttles_to_schedules():
    at_stops = load_and_label_stops()

    with open('data/schedule.json') as f:
        schedule_json = json.load(f)

    shuttles, routes, W = build_weight_matrix(at_stops, schedule_json)

    row_idx, col_idx = linear_sum_assignment(W)
    matches = {shuttles[r]: routes[c] for r, c in zip(row_idx, col_idx)}

    return matches


if __name__ == "__main__":
    mapping = match_shuttles_to_schedules()

    print("Shuttle → Schedule Assignment\n")
    for shuttle, route in mapping.items():
        print(f"{shuttle} → {route}")
