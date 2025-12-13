# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 22:08:09 2025

@author: willi
"""

import os
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------- CONFIG ----------------
INPUT_PATH = "shubble_october.csv"
VEHICLE_COL = "vehicle_id"
MAX_GAP_INCLUDE_S = 15.0

RESAMPLE = True
DELTAS = [5.0]

# AR (ridge) config
ORDERS = range(1,31)
LAMBDAS = [None, 1e-4, 1e-3, 1e-2]

ARIMA_P = [4]
ARIMA_D = [0]
ARIMA_Q = [2]

DEMEAN_PER_SEGMENT = True
INCLUDE_DT_LAGS = False

OUT_PATH_AR = os.path.join(os.path.dirname(INPUT_PATH), "ar_resample_sweep.csv")
OUT_PATH_ARIMA = os.path.join(os.path.dirname(INPUT_PATH), "arima_resample_sweep.csv")

PLOT_RESID_VS_FITTED = True

# ---------------- GEO / TIME UTILITIES ----------------
def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def to_epoch_seconds(series):
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    secs = ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).astype("Int64")
    return secs.astype("float64").to_numpy()


def build_resampled_speed_segments(
    df_raw,
    vehicle_col,
    max_gap_include_s,
    delta,
    demean_per_segment=True,
):
    """
    Returns a list of 1D numpy arrays: each is a resampled speed series (m/s).
    """
    if vehicle_col and vehicle_col in df_raw.columns:
        groups = list(df_raw.groupby(vehicle_col))
    else:
        groups = [(None, df_raw)]

    seg_series = []

    for _, df_g in groups:
        df_speed = compute_speed_mps(df_g)
        raw_segs = split_segments_on_gaps(
            df_speed["t"],
            df_speed["speed_mps"],
            df_speed["dt_s"],
            max_gap_include_s,
        )
        for (t_seg, s_seg) in raw_segs:
            if len(s_seg) == 0:
                continue
            if demean_per_segment:
                s_seg = s_seg - np.nanmean(s_seg)
            t_grid, s_grid = resample_linear(t_seg, s_seg, float(delta))
            if len(s_grid) >= 5:
                seg_series.append(s_grid.astype("float64"))

    return seg_series



def compute_speed_mps(df):
    """
    Compute ground speed in m/s from either:
    - a 'speed_mph' column, or
    - latitude/longitude + time via haversine distance / Δt.

    Returns a DataFrame with columns:
      - 't'        : integer epoch seconds
      - 'speed_mps': speed in meters per second
      - 'dt_s'     : time difference to previous sample in seconds
    """
    time_col = None
    for c in ["timestamp", "created_at", "time", "datetime"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("No timestamp-like column (timestamp/created_at/time/datetime).")

    out = df.copy()
    t_secs = to_epoch_seconds(out[time_col])

    # Drop rows where time couldn't be parsed
    mask = ~np.isnan(t_secs)
    out = out.loc[mask].copy()
    out["t"] = t_secs[mask].astype("int64")

    # Sort by time and compute dt
    out = out.sort_values("t").reset_index(drop=True)
    out["dt_s"] = out["t"].diff()
    out.loc[out.index[0], "dt_s"] = np.nan

    # First choice: directly use speed_mph if present
    if "speed_mph" in out.columns:
        out["speed_mps"] = pd.to_numeric(out["speed_mph"], errors="coerce") * 0.44704
        need_geo = out["speed_mps"].isna()
    else:
        out["speed_mps"] = np.nan
        need_geo = pd.Series(True, index=out.index)

    # Fill missing speeds using GPS distance / Δt if possible
    if need_geo.any():
        if {"latitude", "longitude"}.issubset(out.columns):
            lat = pd.to_numeric(out["latitude"], errors="coerce")
            lon = pd.to_numeric(out["longitude"], errors="coerce")

            dist = np.full(len(out), np.nan, dtype=float)
            for i in range(1, len(out)):
                if (
                    pd.notna(lat.iloc[i - 1])
                    and pd.notna(lat.iloc[i])
                    and pd.notna(lon.iloc[i - 1])
                    and pd.notna(lon.iloc[i])
                ):
                    dist[i] = haversine_meters(
                        lat.iloc[i - 1],
                        lon.iloc[i - 1],
                        lat.iloc[i],
                        lon.iloc[i],
                    )

            speed_geo = np.where(out["dt_s"] > 0, dist / out["dt_s"], np.nan)
            out.loc[need_geo, "speed_mps"] = speed_geo[need_geo.values]
        else:
            raise ValueError("Missing speed_mph and latitude/longitude; cannot compute speeds.")

    # Keep only rows with positive dt and a valid speed
    out = out.loc[(out["dt_s"] > 0) & out["speed_mps"].notna()].reset_index(drop=True)

    return out[["t", "speed_mps", "dt_s"]]


def split_segments_on_gaps(t_series, speed_series, dt_series, max_gap_include_s=15.0):
    t = np.asarray(t_series, dtype="int64")
    s = np.asarray(speed_series, dtype="float64")
    dt = np.asarray(dt_series, dtype="float64")
    segments = []
    start = 0
    for i in range(1, len(t)):
        if (not np.isfinite(dt[i])) or (dt[i] > max_gap_include_s):
            if i - start >= 2:
                segments.append((t[start:i], s[start:i]))
            start = i
    if len(t) - start >= 2:
        segments.append((t[start:], s[start:]))
    return segments


def resample_linear(t_seg, s_seg, delta_s):
    if len(t_seg) < 2:
        return np.array([], dtype="int64"), np.array([], dtype="float64")
    t0, tN = int(t_seg[0]), int(t_seg[-1])
    if (tN - t0) < delta_s:
        return np.array([], dtype="int64"), np.array([], dtype="float64")
    step = int(round(delta_s))
    kmax = (tN - t0) // step
    t_grid = t0 + np.arange(0, kmax + 1) * step
    s_grid = np.interp(t_grid, t_seg, s_seg)
    return t_grid.astype("int64"), s_grid.astype("float64")

# ---------------- AR (RIDGE) DESIGN BUILDERS ----------------
def build_ar_design_from_resampled(s_grid, p):
    T = len(s_grid)
    if T <= p:
        return np.zeros((0, p), dtype="float64"), np.zeros((0,), dtype="float64")
    X = np.zeros((T - p, p), dtype="float64")
    y = np.zeros((T - p,), dtype="float64")
    for i in range(p, T):
        X[i - p, :] = s_grid[i - p : i][::-1]
        y[i - p] = s_grid[i]
    return X, y


def build_arx_design_irregular(t_seg, s_seg, p, include_dt_lags=True):
    T = len(s_seg)
    if T <= p:
        q = p + (p if include_dt_lags else 0)
        return np.zeros((0, q), dtype="float64"), np.zeros((0,), dtype="float64")
    dt_seg = np.diff(t_seg, prepend=np.nan)
    rows = []
    targets = []
    for i in range(p, T):
        s_lags = s_seg[i - p : i][::-1]
        if include_dt_lags:
            dt_lags = dt_seg[i - p : i][::-1]
            if np.isnan(dt_lags).any():
                med = np.nanmedian(dt_seg[1:i]) if i > 1 else 0.0
                dt_lags = np.where(
                    np.isnan(dt_lags),
                    (med if np.isfinite(med) else 0.0),
                    dt_lags,
                )
            feat = np.concatenate([s_lags, dt_lags])
        else:
            feat = s_lags
        rows.append(feat)
        targets.append(s_seg[i])
    X = np.asarray(rows, dtype="float64")
    y = np.asarray(targets, dtype="float64")
    return X, y


