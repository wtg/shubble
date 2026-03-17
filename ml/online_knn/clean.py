"""
Data cleaning for online KNN using the same ideas as LSTM pipelines.

Uses preprocessed data (route, dist_to_route from polyline matching). Drops points
not on the target route and points too far from the route polyline.

Also supports segmenting by time gap: a large gap between two points (same vehicle)
means the shuttle left and came back; we segment so ETAs are not computed across that gap.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

from shared.stops import haversine_vectorized


# Columns we need from preprocessed data for ETA computation
REQUIRED_AFTER_CLEAN = ["vehicle_id", "latitude", "longitude", "timestamp"]


def add_segment_by_time_gap(
    df: pd.DataFrame,
    max_gap_seconds: float = 3600,
    time_column: str = "timestamp",
    segment_column: str = "segment_id",
) -> pd.DataFrame:
    """
    Add segment_id so that a new segment starts when the time gap to the previous
    point (same vehicle) exceeds max_gap_seconds. Use this so ETAs are not computed
    across "left for the day and came back" gaps.

    Does not delete points; callers can pass segment_column to compute_trajectory_etas
    so lookahead stays within the same segment.
    """
    out = df.sort_values(["vehicle_id", time_column]).reset_index(drop=True).copy()
    ts = pd.to_datetime(out[time_column])
    out["_gap"] = ts.diff()
    out.loc[out["vehicle_id"] != out["vehicle_id"].shift(1), "_gap"] = pd.NaT
    out["_new_seg"] = (out["_gap"].isna()) | (out["_gap"].dt.total_seconds() > max_gap_seconds)
    out[segment_column] = out["_new_seg"].cumsum()
    out = out.drop(columns=["_gap", "_new_seg"])
    return out


def add_minutes_from_segment_start(
    df: pd.DataFrame,
    segment_column: str = "segment_id",
    time_column: str = "timestamp",
    out_column: str = "minutes_from_segment_start",
) -> pd.DataFrame:
    """
    Add minutes_from_segment_start: for each row, minutes since the first
    timestamp in that segment. Lets the model use "where we are in the run."
    """
    out = df.copy()
    ts = pd.to_datetime(out[time_column])
    seg_start = ts.groupby(out[segment_column]).transform("min")
    out[out_column] = (ts - seg_start).dt.total_seconds() / 60.0
    return out


def add_minutes_since_last_stop_passage(
    df: pd.DataFrame,
    stops: List[Tuple[float, float]],
    radius_km: float,
    segment_column: str = "segment_id",
    time_column: str = "timestamp",
    out_column: str = "minutes_since_last_stop",
) -> pd.DataFrame:
    """
    Add minutes_since_last_stop: minutes since the last time the vehicle passed
    either stop (or segment start). Counter resets when the vehicle first enters
    within radius_km of any stop. Use for position-in-run and time-weighted KNN.
    """
    out = df.sort_values([segment_column, time_column]).reset_index(drop=True).copy()
    ts = pd.to_datetime(out[time_column])
    lats = out["latitude"].values
    lons = out["longitude"].values
    coords = np.column_stack([lats, lons])

    # Per-row distance to nearest stop (km)
    min_dist = np.full(len(out), np.inf, dtype=float)
    for stop_lat, stop_lon in stops:
        stop_arr = np.array([[stop_lat, stop_lon]])
        dists = haversine_vectorized(coords, np.broadcast_to(stop_arr, (len(out), 2)))
        min_dist = np.minimum(min_dist, dists)
    at_stop = min_dist <= radius_km

    minutes = np.full(len(out), np.nan, dtype=float)
    for seg_id, group in out.groupby(segment_column):
        idx = group.index.values
        seg_ts = ts.loc[idx].values
        seg_at = at_stop[idx]
        # Passage times: segment start + each first entry into any stop
        t0 = pd.Timestamp(seg_ts[0])
        passage_times = [t0]
        for i in range(1, len(seg_at)):
            if seg_at[i] and not seg_at[i - 1]:
                passage_times.append(pd.Timestamp(seg_ts[i]))
        for i in range(len(idx)):
            t = pd.Timestamp(seg_ts[i])
            last = max(p for p in passage_times if p <= t)
            minutes[idx[i]] = (t - last).total_seconds() / 60.0

    out[out_column] = minutes
    return out


def clean_preprocessed_for_route(
    df: pd.DataFrame,
    route_name: str,
    max_dist_to_route_km: float = 0.08,  # 80 m
) -> pd.DataFrame:
    """
    Keep only rows that are on the given route and within max_dist_to_route_km of it.

    Matches LSTM segment pipeline logic: points with dist_to_route > threshold are
    considered "off route" and are dropped. Also drops rows with NaN route.

    Expects df to have columns: route, dist_to_route (from ml preprocess pipeline).

    Returns:
        DataFrame with only REQUIRED_AFTER_CLEAN columns, sorted by timestamp.
    """
    if "route" not in df.columns:
        raise ValueError("Preprocessed DataFrame must have 'route' column")
    if "dist_to_route" not in df.columns:
        raise ValueError("Preprocessed DataFrame must have 'dist_to_route' column")

    out = df.copy()
    # Drop shuttles/points not on the route
    out = out[out["route"] == route_name]
    # Drop points too far from route (same idea as LSTM max_distance)
    out = out[out["dist_to_route"].notna() & (out["dist_to_route"] <= max_dist_to_route_km)]
    out = out[REQUIRED_AFTER_CLEAN].sort_values("timestamp").reset_index(drop=True)
    return out
