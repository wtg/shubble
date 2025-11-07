import pandas as pd
import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from stops import Stops


def load_and_label_stops():
    """
    Load data from shuttle.csv file
    and add 2 new columns route_name and stop_name
    with data by calling is_at_stop from stops.py 
    """
    df = pd.read_csv('data/shubble_2025-10-29.csv')

    #Check if vehicle is at stop and add its route and stop name to dataset or NULL
    route, stop = [], []
    for idx, r in df.iterrows():
        try:
            rn, sn = Stops.is_at_stop((float(r['latitude']), float(r['longitude'])))
        except Exception as e: #If Vehicle isn't at Stop add None to route_name and stop_name columns
            print(f"[ERROR] Stop detection failed at row {idx}: {e}")
            rn, sn = None, None
        route.append(rn)
        stop.append(sn)

    df['route_name'] = route
    df['stop_name'] = stop
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')

    # Filter only rows where at a stop
    labeled = df.dropna(subset=['route_name', 'stop_name'])[['vehicle_id','timestamp','route_name','stop_name']].copy()
    return labeled


def match_shuttles_to_schedules():
    """
    Match shuttle vehicle data to the most likely schedule route based on timestamp and stop location.
    Uses a cost matrix to measure how well each shuttle’s actual stop times align with scheduled times.
    """

    # Load labeled shuttle stop data
    at_stops = load_and_label_stops()

    with open('data/schedule.json') as f:
        sched = json.load(f)

    # Get first timestamp to determine current day
    first_ts = at_stops['timestamp'].iloc[0]
    day_name = first_ts.day_name().upper()

    # Select correct day schedule (weekday, saturday, sunday)
    day_key = sched[day_name]
    routes = sched[day_key]

    # Get all unique shuttle IDs
    shuttles = at_stops['vehicle_id'].unique().tolist()
    
    # Combine date from logs with times from schedule
    date_str = first_ts.strftime("%Y-%m-%d")
    sched_flat = [
        (name, [(pd.to_datetime(f"{date_str} {t}"), s) for t, s in times])
        for name, times in routes.items()
    ]

    W = np.zeros((len(shuttles), len(sched_flat)))

    # Compare each shuttle's stop log to each schedule route
    for i, shuttle in enumerate(shuttles):
        logs = at_stops[at_stops['vehicle_id'] == shuttle]
        
        for j, (schedule_label, stops) in enumerate(sched_flat):
            # Count matches where route name and timestamps by the minute align
            matches = sum(
                (
                    (
                        (logs['route_name'] == sn)
                        & (logs['timestamp'].dt.floor('min') == pd.to_datetime(ts).floor('min'))
                    ).any()
                )
                for ts, sn in stops
            )
            #Calculate cost : cost = times shuttle i is at union at scheduled timestamp from schedule j/number of scheduled loops in schedule j
            cost = 1 - (matches / len(stops))
            W[i, j] = cost

    # Use Hungarian algorithm to assign each shuttle to one route
    row, col = linear_sum_assignment(W)

    # Store best matches in dictionary
    result = {shuttles[r]: sched_flat[c][0] for r, c in zip(row, col)}

    # Print results with matching weights
    for shuttle_idx, route_idx in zip(row, col):
        shuttle = shuttles[shuttle_idx]
        route_label = sched_flat[route_idx][0]
        match_cost = W[shuttle_idx, route_idx]
        print(f"{route_label} = Shuttle {shuttle} with matching weight {match_cost:.3f}")

    return result

if __name__ == "__main__":
    result = match_shuttles_to_schedules()
   
