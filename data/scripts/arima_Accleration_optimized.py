# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 22:27:17 2025

@author: willi
"""

import os
import math
import warnings
import random

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
DELTAS = [5.0]  # resampling step (seconds)

# Acceleration horizon: a_k = (v_{k+h} - v_k) / (h * DELTAS[0])
ACCEL_H = 3  

# AR (ridge) config
ORDERS = range(1, 31)
LAMBDAS = [None, 1e-4, 1e-3, 1e-2]

# ARIMA search grid
ARIMA_P_LIST = range(1, 31)
ARIMA_D_LIST = [0]
ARIMA_Q_LIST = [0, 1, 2]

# Quick ARIMA "train" sweep points
ARIMA_TARGET_POINTS = 15000
ARIMA_MIN_LEN = 10
ARIMA_RANDOM_SEED = 41

# Test-suite config (shared idea: ~10k points)
TEST_TARGET_POINTS = 5000
TEST_RANDOM_SEED = 39

DEMEAN_PER_SEGMENT = False   # we now work on raw acceleration
INCLUDE_DT_LAGS = False

NOISE_WINDOW_SAMPLES = 4

OUT_PATH_AR = os.path.join(os.path.dirname(INPUT_PATH), "MINI_ar_resample_sweep.csv")

PLOT_RESID_VS_FITTED = True

CONSEC_POINTS = 200

RECURSIVE_FORECAST = True


# ---------------- GEO / TIME UTILITIES ----------------
def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def to_epoch_seconds(series):
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    secs = ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).astype("Int64")
    return secs.astype("float64").to_numpy()


def pick_long_segment(seg_series, min_len, random_seed=999):
    """
    Pick a segment of at least min_len points.
    Prefers longest available; breaks ties randomly.
    """
    candidates = [(i, s) for i, s in enumerate(seg_series) if len(s) >= min_len]
    if not candidates:
        raise ValueError(
            f"No segment has length >= {min_len}. "
            "Decrease CONSEC_POINTS or min_len requirement."
        )

    max_len = max(len(s) for _, s in candidates)
    top = [(i, s) for (i, s) in candidates if len(s) >= max_len * 0.9]

    rng = random.Random(random_seed)
    idx, seg = rng.choice(top)
    print(f"[PLOT] Using segment {idx} with length={len(seg)} for consecutive forecast.")
    return seg


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


def build_resampled_accel_segments(
    df_raw,
    vehicle_col,
    max_gap_include_s,
    delta,
    demean_per_segment=True,
):
    """
    Returns a list of 1D numpy arrays: each is a resampled acceleration series (m/s^2).

    Acceleration is computed from resampled speeds as:
        a[k] = (v[k + ACCEL_H] - v[k]) / (ACCEL_H * delta)
    where ACCEL_H is a global config (number of resampled steps).
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
            # Need enough points for an ACCEL_H-step difference plus some slack
            if len(s_grid) >= ACCEL_H + 5:
                a_grid = (s_grid[ACCEL_H:] - s_grid[:-ACCEL_H]) / (ACCEL_H * float(delta))
                if len(a_grid) >= 5:
                    seg_series.append(a_grid.astype("float64"))

    return seg_series


# ---------------- NOISE ESTIMATION ----------------
def estimate_speed_noise_from_smoothing(seg_series, window=20):
    """
    seg_series: list of 1D numpy arrays (resampled *acceleration* in m/s^2)
    window: moving average window length in samples
    """
    all_resid = []

    for s in seg_series:
        if len(s) < window + 2:
            continue
        kernel = np.ones(window) / window
        smooth = np.convolve(s, kernel, mode="same")
        resid = s - smooth
        resid = resid[window:-window] if len(resid) > 2 * window else resid
        all_resid.append(resid)

    if not all_resid:
        return np.nan, np.nan

    all_resid = np.concatenate(all_resid)
    sigma = float(np.std(all_resid))
    # convert to "mph/s" just for a rough sense: m/s^2 * 2.23694 ≈ mph/s
    sigma_mph_per_s = sigma * 2.23694
    return sigma, sigma_mph_per_s


# ---------------- AR (RIDGE) DESIGN BUILDERS ----------------
def build_ar_design_from_resampled(s_grid, p):
    # Here s_grid is acceleration
    T = len(s_grid)
    if T <= p:
        return np.zeros((0, p), dtype="float64"), np.zeros((0,), dtype="float64")
    X = np.zeros((T - p, p), dtype="float64")
    y = np.zeros((T - p,), dtype="float64")
    for i in range(p, T):
        X[i - p, :] = s_grid[i - p: i][::-1]
        y[i - p] = s_grid[i]
    return X, y


def build_arx_design_irregular(t_seg, s_seg, p, include_dt_lags=True):
    # Not used in our resampled setup, but left for completeness
    T = len(s_seg)
    if T <= p:
        q = p + (p if include_dt_lags else 0)
        return np.zeros((0, q), dtype="float64"), np.zeros((0,), dtype="float64")
    dt_seg = np.diff(t_seg, prepend=np.nan)
    rows = []
    targets = []
    for i in range(p, T):
        s_lags = s_seg[i - p: i][::-1]
        if include_dt_lags:
            dt_lags = dt_seg[i - p: i][::-1]
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


def ridge_fit(X, y, lam):
    if X.shape[0] == 0:
        return (
            (np.zeros((X.shape[1],), dtype="float64") if X.shape[1] > 0 else np.array([])),
            np.array([]),
            np.array([]),
            float("nan"),
        )

    XT = X.T
    A = XT @ X + (0 if lam in (None, 0) else lam) * np.eye(X.shape[1])
    b = XT @ y
    beta = np.linalg.solve(A, b)
    y_hat = X @ beta
    eps = y - y_hat
    mse = float(np.mean(eps ** 2))
    return beta, y_hat, eps, mse


def evaluate_ar_pipeline(
    df_raw,
    vehicle_col,
    max_gap_include_s,
    resample,
    deltas,
    orders,
    lambdas,
    demean_per_segment=True,
    include_dt_lags=True,
):
    """
    Now evaluates AR on *acceleration* segments.
    """
    results = []
    models = {}
    debug_pairs = []

    delta_list = deltas if resample else [None]

    for delta in delta_list:
        segs = []

        if resample:
            # Use acceleration segments directly
            accel_segs = build_resampled_accel_segments(
                df_raw,
                vehicle_col,
                max_gap_include_s,
                delta,
                demean_per_segment=demean_per_segment,
            )
            for a_grid in accel_segs:
                if len(a_grid) >= 5:
                    segs.append(("resampled", None, a_grid))
        else:
            # Fallback: irregular acceleration from raw speeds (rarely used here)
            if vehicle_col and vehicle_col in df_raw.columns:
                groups = list(df_raw.groupby(vehicle_col))
            else:
                groups = [(None, df_raw)]
            for _, df_g in groups:
                df_speed = compute_speed_mps(df_g)
                raw_segs = split_segments_on_gaps(
                    df_speed["t"],
                    df_speed["speed_mps"],
                    df_speed["dt_s"],
                    max_gap_include_s,
                )
                for (t_seg, s_seg) in raw_segs:
                    if len(s_seg) < 3:
                        continue
                    if demean_per_segment:
                        s_seg = s_seg - np.nanmean(s_seg)
                    a_seg = np.diff(s_seg) / np.diff(t_seg.astype("float64"))
                    if len(a_seg) >= 3:
                        segs.append(("raw", t_seg[1:], a_seg))

