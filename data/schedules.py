import pandas as pd
import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from stops import Stops


def load_and_label_stops():
    df = pd.read_csv('data/shuttle.csv')

    route, stop = [], []
    for _, r in df.iterrows():
        try:
            rn, sn = Stops.is_at_stop(np.array([[float(r['latitude']), float(r['longitude'])]]))
        except:
            rn, sn = None, None
        route.append(rn)
        stop.append(sn)

    df['route_name'] = route
    df['stop_name'] = stop
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df.dropna(subset=['route_name', 'stop_name'])[['vehicle_id','timestamp','stop_name']].copy()


def match_shuttles_to_schedules():
    at_stops = load_and_label_stops()

    with open('data/schedule.json') as f:
        sched = json.load(f)

    day_key = sched[pd.Timestamp(at_stops['timestamp'].iloc[0]).day_name().upper()]
    routes = sched[day_key]

    shuttles = at_stops['vehicle_id'].unique().tolist()
    sched_flat = [(name, [(pd.to_datetime(t), s) for t, s in times]) for name, times in routes.items()]

    W = np.zeros((len(shuttles), len(sched_flat)))

    for i, shuttle in enumerate(shuttles):
        logs = at_stops[at_stops['vehicle_id'] == shuttle]
        for j, (_, stops) in enumerate(sched_flat):
            matches = sum(
                ((logs['stop_name'] == sn) & (logs['timestamp'] == ts)).any()
                for ts, sn in stops
            )
            W[i, j] = 1 - (matches / len(stops))

    row, col = linear_sum_assignment(W)
    return {shuttles[r]: sched_flat[c][0] for r, c in zip(row, col)}


if __name__ == "__main__":
    result = match_shuttles_to_schedules()
    for shuttle, route in result.items():
        print(f"{shuttle} -> {route}")
