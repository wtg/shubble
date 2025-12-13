# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 19:18:37 2025

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

# AR (ridge) config
ORDERS = range(1, 20)
LAMBDAS = [None, 1e-4, 1e-3, 1e-2]

# ARIMA search grid
ARIMA_P_LIST = range(1, 3)
ARIMA_D_LIST = [0]
ARIMA_Q_LIST = [0, 1, 2]

# Quick ARIMA "train" sweep points

AR_TRAIN_TARGET_POINTS = 15000
AR_TRAIN_RANDOM_SEED = 123

ARIMA_TARGET_POINTS = 15000
ARIMA_MIN_LEN = 10
ARIMA_RANDOM_SEED = 41

# Test-suite config (shared idea: ~10k points)
TEST_TARGET_POINTS = 5000
TEST_RANDOM_SEED = 39

DEMEAN_PER_SEGMENT = False
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
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
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
        raise ValueError(f"No segment has length >= {min_len}. "
                         "Decrease CONSEC_POINTS or min_len requirement.")

    # Prefer the longest few, then random among them for variety
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


# ---------------- NOISE ESTIMATION ----------------
def estimate_speed_noise_from_smoothing(seg_series, window=20):
    """
    seg_series: list of 1D numpy arrays (resampled speeds in m/s)
    window: moving average window length in samples
    """
    all_resid = []

    for s in seg_series:
        if len(s) < window + 2:
            continue
        # moving average via convolution
        kernel = np.ones(window) / window
        smooth = np.convolve(s, kernel, mode="same")
        resid = s - smooth
        # drop edges to avoid convolution edge effects
        resid = resid[window:-window] if len(resid) > 2 * window else resid
        all_resid.append(resid)

    if not all_resid:
        return np.nan, np.nan

    all_resid = np.concatenate(all_resid)
    sigma_mps = float(np.std(all_resid))
    sigma_mph = sigma_mps * 2.23694
    return sigma_mps, sigma_mph


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
    results = []
    models = {}
    debug_pairs = []

    # Group by vehicle if column present
    if vehicle_col and vehicle_col in df_raw.columns:
        groups = list(df_raw.groupby(vehicle_col))
    else:
        groups = [(None, df_raw)]

    delta_list = deltas if resample else [None]

    for delta in delta_list:
        segs = []

        # Build segments (either resampled or raw)
        for _, df_g in groups:
            df_speed = compute_speed_mps(df_g)
            raw_segs = split_segments_on_gaps(
                df_speed["t"],
                df_speed["speed_mps"],
                df_speed["dt_s"],
                max_gap_include_s,
            )
            for (t_seg, s_seg) in raw_segs:
                if demean_per_segment and len(s_seg) > 0:
                    s_seg = s_seg - np.nanmean(s_seg)
                if resample:
                    t_grid, s_grid = resample_linear(t_seg, s_seg, float(delta))
                    if len(s_grid) >= 5:
                        segs.append(("resampled", t_grid, s_grid))
                        debug_pairs.append(
                            {
                                "delta": float(delta),
                                "t_raw": t_seg,
                                "speed_raw": s_seg,
                                "t_grid": t_grid,
                                "speed_grid": s_grid,
                            }
                        )
                else:
                    if len(s_seg) >= 3:
                        segs.append(("raw", t_seg, s_seg))

        # Build lagged design matrices for each order p
        lagged_by_p = {}
        for p in orders:
            X_list, y_list = [], []
            for mode, t_arr, s_arr in segs:
                if mode == "resampled":
                    Xs, ys = build_ar_design_from_resampled(s_arr, p)
                else:
                    Xs, ys = build_arx_design_irregular(
                        t_arr, s_arr, p, include_dt_lags=include_dt_lags
                    )
                if Xs.shape[0] > 0:
                    X_list.append(Xs)
                    y_list.append(ys)
        
            if X_list:
                X_all = np.vstack(X_list)
                y_all = np.concatenate(y_list)
        
                # ---- NEW: cap AR training rows for fairness ----
                n_rows = X_all.shape[0]
                if n_rows > AR_TRAIN_TARGET_POINTS:
                    rng = np.random.default_rng(AR_TRAIN_RANDOM_SEED + p)
                    idx = rng.choice(n_rows, size=AR_TRAIN_TARGET_POINTS, replace=False)
                    X_all = X_all[idx]
                    y_all = y_all[idx]
            else:
                feat_dim = p if resample else (p + (p if include_dt_lags else 0))
                X_all = np.zeros((0, feat_dim))
                y_all = np.zeros((0,))
        
            lagged_by_p[p] = (X_all, y_all)


        # Fit models for each p, lambda
        for p in orders:
            X_all, y_all = lagged_by_p[p]
            for lam in lambdas:
                lam_key = 0.0 if lam in (None, 0) else float(lam)
                beta, y_hat, eps, mse = ridge_fit(X_all, y_all, lam)
                rmse = float(np.sqrt(mse)) if mse == mse else float("nan")
                results.append(
                    {
                        "mode": "RESAMPLED"
                        if resample
                        else ("RAW_ARX_dt" if include_dt_lags else "RAW_AR"),
                        "delta_s": (float(delta) if resample else None),
                        "order_p": int(p),
                        "lambda": lam_key,
                        "n_rows": int(X_all.shape[0]),
                        "MSE": mse,
                        "RMSE": rmse,
                    }
                )
                models[((float(delta) if resample else None), int(p), lam_key)] = {
                    "beta": beta,
                    "mse": mse,
                    "rmse": rmse,
                    "n_rows": int(X_all.shape[0]),
                    "feat_dim": int(X_all.shape[1]) if X_all.ndim == 2 else 0,
                    "eps": eps,
                    "y_hat": y_hat,
                }

    res_df = pd.DataFrame(results).sort_values(
        ["mode", "delta_s", "order_p", "lambda"]
    ).reset_index(drop=True)
    return res_df, models, debug_pairs


# ---------------- QUICK ARIMA EVAL (SAMPLED) ----------------
def quick_arima_eval(
    seg_series,
    p, d, q,
    target_points=5000,
    min_len=10,
    random_seed=42,
):
    """
    Quickly evaluate one ARIMA(p,d,q) on a randomly chosen set of segments,
    selected until we accumulate ~target_points total time points.

    Returns (rmse, aic_mean, bic_mean, n_obs, n_segments_used).
    """
    if not seg_series:
        print(f"[quick_arima_eval] seg_series is empty for (p,d,q)=({p},{d},{q})")
        return float("nan"), float("nan"), float("nan"), 0, 0

    # minimum length needed for ARIMA(p,d,q)
    min_len_eff = max(min_len, p + d + q + 3)

    # Randomize segment order
    rng = random.Random(random_seed)
    indices = list(range(len(seg_series)))
    rng.shuffle(indices)

    segs_used = []
    cum_len = 0

    for idx in indices:
        s = seg_series[idx]
        if len(s) < min_len_eff:
            continue
        segs_used.append(s)
        cum_len += len(s)
        if cum_len >= target_points:
            break

    if not segs_used:
        print(
            f"[quick_arima_eval] No segments long enough for (p,d,q)=({p},{d},{q}). "
            f"min_len_eff={min_len_eff}"
        )
        return float("nan"), float("nan"), float("nan"), 0, 0

    print(
        f"[quick_arima_eval] ARIMA({p},{d},{q}): "
        f"using {len(segs_used)} segments, total_len≈{cum_len} points "
        f"(target={target_points})"
    )

    all_resid = []
    aics = []
    bics = []
    n_obs = 0
    n_seg_fit = 0
    fail_count = 0

    for local_idx, s in enumerate(segs_used):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model = ARIMA(
                    s,
                    order=(p, d, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit()
        except Exception as e:
            fail_count += 1
            if fail_count <= 5:
                print(
                    f"[quick_arima_eval] ARIMA failed for segment idx={local_idx}, "
                    f"len={len(s)}, (p,d,q)=({p},{d},{q}): {repr(e)}"
                )
            continue

        resid = np.asarray(res.resid, dtype="float64")
        mask = np.isfinite(resid)
        resid = resid[mask]
        if resid.size == 0:
            continue

        all_resid.append(resid)
        n_obs += resid.size
        n_seg_fit += 1

        if hasattr(res, "aic"):
            aics.append(res.aic)
        if hasattr(res, "bic"):
            bics.append(res.bic)

    if n_obs == 0:
        print(
            f"[quick_arima_eval] No successful fits for (p,d,q)=({p},{d},{q}). "
            f"fail_count={fail_count}, attempted_segments={len(segs_used)}"
        )
        return float("nan"), float("nan"), float("nan"), 0, 0

    resid_concat = np.concatenate(all_resid)
    mse = float(np.mean(resid_concat ** 2))
    rmse = float(np.sqrt(mse))
    aic_mean = float(np.mean(aics)) if aics else float("nan")
    bic_mean = float(np.mean(bics)) if bics else float("nan")

    return rmse, aic_mean, bic_mean, n_obs, n_seg_fit


# ---------- AR TEST DESIGN (10k points) ----------
def build_ar_test_design(seg_series, p, target_points=10000, random_seed=123):
    """
    Build a test design matrix (X_test, y_test) for AR of order p,
    from random segments until we accumulate ~target_points rows.
    """
    rng = random.Random(random_seed)
    indices = list(range(len(seg_series)))
    rng.shuffle(indices)

    X_list = []
    y_list = []
    total_rows = 0

    for idx in indices:
        s = seg_series[idx]
        if len(s) <= p:
            continue
        X_s, y_s = build_ar_design_from_resampled(s, p)
        if X_s.shape[0] == 0:
            continue
        X_list.append(X_s)
        y_list.append(y_s)
        total_rows += X_s.shape[0]
        if total_rows >= target_points:
            break

    if not X_list:
        return np.zeros((0, p)), np.zeros((0,))

    X_test = np.vstack(X_list)
    y_test = np.concatenate(y_list)

    # Optionally trim to exactly target_points rows
    if X_test.shape[0] > target_points:
        X_test = X_test[:target_points]
        y_test = y_test[:target_points]

    return X_test, y_test

def build_arima_global_train_test_series(
    seg_series,
    train_target_points=15000,
    test_target_points=10000,
    random_seed=123,
):
    """
    Build a single global train and test 1D series for ARIMA by concatenating segments.

    - Randomly shuffles segments.
    - Fills `train` up to ~train_target_points.
    - Then fills `test` up to ~test_target_points from the remaining data.
    - Returns (train_series, test_series) as 1D float64 numpy arrays.
    """
    if not seg_series:
        return np.zeros((0,), dtype="float64"), np.zeros((0,), dtype="float64")

    rng = random.Random(random_seed)
    indices = list(range(len(seg_series)))
    rng.shuffle(indices)

    train_list = []
    test_list = []
    n_train = 0
    n_test = 0

