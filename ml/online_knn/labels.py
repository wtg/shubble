"""
Compute time-to-stop labels from raw trajectories.

For each (vehicle_id, lat, lon, timestamp) we compute: seconds from this point
until the vehicle first reaches stop1 and stop2 (looking forward in the same
trajectory). If segment_column is provided, lookahead stays within the same
segment (so large time gaps = "left and came back" do not link runs).
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from shared.stops import haversine_vectorized

# Default radius (km) to consider "at stop"
DEFAULT_STOP_RADIUS_KM = 0.005  # ~5 m


def _trajectory_etas_to_stops(
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    stop_lat: float,
    stop_lon: float,
    radius_km: float,
) -> np.ndarray:
    """
    For each index i, seconds from time[i] until first time within radius of stop.
    Returns NaN where the trajectory never reaches the stop. O(n) per vehicle.
    """
    n = len(lats)
    out = np.full(n, np.nan, dtype=float)
    coords = np.column_stack([lats, lons])
    stop = np.array([[stop_lat, stop_lon]])
    dists = haversine_vectorized(coords, np.broadcast_to(stop, (n, 2)))
    at_stop = dists <= radius_km
    # For each hit index j, backfill eta for all i in (prev_j, j]
    prev_j = -1
    for j in range(n):
        if not at_stop[j]:
            continue
        t_j = times[j] / 1e9  # seconds
        for i in range(prev_j + 1, j + 1):
            out[i] = t_j - (times[i] / 1e9)
        prev_j = j
    return out


def compute_trajectory_etas(
    df: pd.DataFrame,
    stops: List[Tuple[float, float]],
    radius_km: float = DEFAULT_STOP_RADIUS_KM,
    time_column: str = "timestamp",
    segment_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add ETA columns (seconds to each stop) for every row, from trajectory lookahead.

    Groups by vehicle_id (and segment_column if given), sorts by time, then for each
    point computes seconds until the vehicle first enters radius_km of each stop
    (forward in time, within the same segment). Rows that never reach a stop get NaN.

    Args:
        df: Must have vehicle_id, latitude, longitude, and time_column.
        stops: List of (lat, lon) for each stop (e.g. 2 stops).
        radius_km: Consider "at stop" when within this distance (km).
        time_column: Datetime column name.
        segment_column: If set, group by (vehicle_id, segment_column) so ETAs do
            not look across segment boundaries (e.g. after a large time gap).

    Returns:
        New DataFrame with added columns eta_seconds_stop_0, eta_seconds_stop_1, ...
        and original columns.
    """
    if time_column not in df.columns:
        raise ValueError(f"Missing time column: {time_column}")
    group_cols = ["vehicle_id"]
    if segment_column and segment_column in df.columns:
        group_cols.append(segment_column)
    df = df.sort_values(group_cols + [time_column]).reset_index(drop=True)
    times_ns = df[time_column].astype("int64").values
    lats = df["latitude"].values
    lons = df["longitude"].values

    out = df.copy()
    for idx, (stop_lat, stop_lon) in enumerate(stops):
        eta = np.full(len(df), np.nan, dtype=float)
        for _, group in df.groupby(group_cols):
            if group.empty:
                continue
            inds = group.index.values
            eta[inds] = _trajectory_etas_to_stops(
                lats[inds], lons[inds], times_ns[inds],
                stop_lat, stop_lon, radius_km,
            )
        out[f"eta_seconds_stop_{idx}"] = eta

    return out
