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
    Quickly evaluate one ARIMA(p,d,q) on a randomly chosen set of *acceleration* segments,
    selected until we accumulate ~target_points total time points.
    """
    if not seg_series:
        print(f"[quick_arima_eval] seg_series is empty for (p,d,q)=({p},{d},{q})")
        return float("nan"), float("nan"), float("nan"), 0, 0

    min_len_eff = max(min_len, p + d + q + 3)

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
    from random *acceleration* segments until we accumulate ~target_points rows.
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

    if X_test.shape[0] > target_points:
        X_test = X_test[:target_points]
        y_test = y_test[:target_points]

    return X_test, y_test


# ---------------- MAIN ----------------
if __name__ == "__main__":
    df_raw = pd.read_csv(INPUT_PATH)

    # Build resampled acceleration segments just once (for noise + ARIMA + AR test)
    seg_series = build_resampled_accel_segments(
        df_raw,
        VEHICLE_COL,
        MAX_GAP_INCLUDE_S,
        DELTAS[0],
        DEMEAN_PER_SEGMENT,
    )

    print(f"Total resampled acceleration segments: {len(seg_series)}")
    if seg_series:
        lengths = [len(s) for s in seg_series]
        print(
            f"Segment length stats: "
            f"min={min(lengths)}, max={max(lengths)}, median={np.median(lengths):.1f}"
        )
        all_s = np.concatenate(seg_series)
        global_std = float(np.std(all_s))
        print(
            f"Global accel std: {global_std:.3f} m/s^2 "
            f"({global_std * 2.23694:.3f} mph/s)"
        )
        print(
            f"Acceleration horizon: ACCEL_H={ACCEL_H} steps, "
            f"span={ACCEL_H * DELTAS[0]:.1f} seconds"
        )

    # ---------- Noise estimate ----------
    sigma_acc, sigma_mph_per_s = estimate_speed_noise_from_smoothing(
        seg_series,
        window=NOISE_WINDOW_SAMPLES,
    )
    print(
        f"\n[NOISE] Estimated accel σ ≈ {sigma_acc:.3f} m/s^2 "
        f"≈ {sigma_mph_per_s:.3f} mph/s (window={NOISE_WINDOW_SAMPLES} samples)"
    )

    # ---------- AR (ridge) pipeline ----------
    res_df_ar, models_ar, debug_pairs = evaluate_ar_pipeline(
        df_raw,
        VEHICLE_COL,
        MAX_GAP_INCLUDE_S,
        RESAMPLE,
        DELTAS,
        ORDERS,
        LAMBDAS,
        DEMEAN_PER_SEGMENT,
        INCLUDE_DT_LAGS,
    )

    res_df_ar.to_csv(OUT_PATH_AR, index=False)
    print(f"\n[AR] Saved sweep to: {OUT_PATH_AR}")

    if not res_df_ar.empty:
        leaderboard_ar = res_df_ar.sort_values("RMSE").head(20)
        print("\n[AR] Top 20 (lowest RMSE):")
        cols = ["mode", "delta_s", "order_p", "lambda", "n_rows", "MSE", "RMSE"]
        print(leaderboard_ar[cols].to_string(index=False))

        # Best AR config (train RMSE)
        best_ar_row = res_df_ar.sort_values("RMSE").iloc[0]
        best_ar_rmse = float(best_ar_row["RMSE"])
        key_ar = (
            (float(best_ar_row["delta_s"]) if pd.notna(best_ar_row["delta_s"]) else None),
            int(best_ar_row["order_p"]),
            float(best_ar_row["lambda"]),
        )
        print("\n[AR] Best config (train RMSE):")
        print(best_ar_row[cols].to_string())
    else:
        print("[AR] No rows produced (check input columns and parameters).")
        best_ar_rmse = float("nan")
        key_ar = None

    # ---------- Quick ARIMA "train" eval (5k points) ----------
    print("\n[ARIMA] Quick eval on sampled accel segments (train, ~15000 points)...")
    arima_results = []

    for p in ARIMA_P_LIST:
        for d in ARIMA_D_LIST:
            for q in ARIMA_Q_LIST:
                print(f"  Evaluating ARIMA({p},{d},{q})...")
                rmse, aic_mean, bic_mean, n_obs, n_seg = quick_arima_eval(
                    seg_series,
                    p,
                    d,
                    q,
                    target_points=ARIMA_TARGET_POINTS,
                    min_len=ARIMA_MIN_LEN,
                    random_seed=ARIMA_RANDOM_SEED,
                )
                arima_results.append(
                    {
                        "p": p,
                        "d": d,
                        "q": q,
                        "RMSE": rmse,
                        "AIC_mean": aic_mean,
                        "BIC_mean": bic_mean,
                        "n_obs": n_obs,
                        "n_segments": n_seg,
                    }
                )

    arima_df = pd.DataFrame(arima_results).sort_values("RMSE").reset_index(drop=True)
    print("\n[ARIMA] Quick-eval results (train, sorted by RMSE):")
    print(arima_df.to_string(index=False))

    if not arima_df.empty and np.isfinite(arima_df["RMSE"]).any():
        best_arima_row_train = arima_df.loc[arima_df["RMSE"].idxmin()]
        best_arima_rmse_train = float(best_arima_row_train["RMSE"])
        print("\n[ARIMA] Best quick-eval config (train):")
        print(best_arima_row_train.to_string())
    else:
        best_arima_rmse_train = float("nan")
        best_arima_row_train = None
        print("[ARIMA] No successful ARIMA fits in quick eval.")

    # ---------- AR TEST SUITE (~10k points) ----------
    print("\n[AR TEST] Evaluating all AR configs on ~10,000 points...")
    ar_test_rows = []

    for p in ORDERS:
        X_test, y_test = build_ar_test_design(
            seg_series,
            p,
            target_points=TEST_TARGET_POINTS,
            random_seed=TEST_RANDOM_SEED,
        )
        n_rows_test = X_test.shape[0]
        if n_rows_test == 0:
            continue

        for lam in LAMBDAS:
            lam_key = 0.0 if lam in (None, 0) else float(lam)
            model_key = (float(DELTAS[0]) if RESAMPLE else None, int(p), lam_key)
            if model_key not in models_ar:
                continue
            beta = models_ar[model_key]["beta"]
            if beta.size == 0:
                rmse_test = float("nan")
            else:
                y_hat_test = X_test @ beta
                eps_test = y_test - y_hat_test
                mse_test = float(np.mean(eps_test ** 2))
                rmse_test = float(math.sqrt(mse_test))

            ar_test_rows.append(
                {
                    "order_p": int(p),
                    "lambda": lam_key,
                    "n_rows_test": int(n_rows_test),
                    "RMSE_test": rmse_test,
                }
            )

    if ar_test_rows:
        ar_test_df = pd.DataFrame(ar_test_rows).sort_values("RMSE_test").reset_index(drop=True)
        print("\n[AR TEST] Results on ~10,000 points (sorted by RMSE_test):")
        print(ar_test_df.head(20).to_string(index=False))
        best_ar_test_row = ar_test_df.iloc[0]
        best_ar_rmse_test = float(best_ar_test_row["RMSE_test"])
    else:
        print("[AR TEST] No AR test rows produced.")
        best_ar_rmse_test = float("nan")

    # ---------- ARIMA TEST SUITE (~10k points) ----------
    print("\n[ARIMA TEST] Evaluating all ARIMA configs on ~10,000 points...")
    arima_test_results = []

    for p in ARIMA_P_LIST:
        for d in ARIMA_D_LIST:
            for q in ARIMA_Q_LIST:
                print(f"  Testing ARIMA({p},{d},{q})...")
                rmse_t, aic_t, bic_t, n_obs_t, n_seg_t = quick_arima_eval(
                    seg_series,
                    p,
                    d,
                    q,
                    target_points=TEST_TARGET_POINTS,
                    min_len=ARIMA_MIN_LEN,
                    random_seed=TEST_RANDOM_SEED,
                )
                arima_test_results.append(
                    {
                        "p": p,
                        "d": d,
                        "q": q,
                        "RMSE_test": rmse_t,
                        "AIC_mean_test": aic_t,
                        "BIC_mean_test": bic_t,
                        "n_obs_test": n_obs_t,
                        "n_segments_test": n_seg_t,
                    }
                )

    arima_test_df = pd.DataFrame(arima_test_results).sort_values("RMSE_test").reset_index(drop=True)
    print("\n[ARIMA TEST] Results on ~10,000 points (sorted by RMSE_test):")
    print(arima_test_df.to_string(index=False))

    if not arima_test_df.empty and np.isfinite(arima_test_df["RMSE_test"]).any():
        best_arima_row_test = arima_test_df.loc[arima_test_df["RMSE_test"].idxmin()]
        best_arima_rmse_test = float(best_arima_row_test["RMSE_test"])
        print("\n[ARIMA TEST] Best config on ~10,000 points:")
        print(best_arima_row_test.to_string())
    else:
        best_arima_rmse_test = float("nan")
        best_arima_row_test = None
        print("[ARIMA TEST] No successful ARIMA test fits.")

    # ---------- Summary comparison ----------
    print("\n===== SUMMARY =====")
    print(
        f"Global accel std              : {global_std:.3f} m/s^2 "
        f"({global_std * 2.23694:.3f} mph/s)"
    )
    print(
        f"Noise σ (smoothing, accel)    : {sigma_acc:.3f} m/s^2 "
        f"({sigma_mph_per_s:.3f} mph/s)"
    )
    print(f"Best AR RMSE (train, accel)   : {best_ar_rmse:.3f} m/s^2")
    print(f"Best AR RMSE (test~10k)       : {best_ar_rmse_test:.3f} m/s^2")
    print(f"Best ARIMA RMSE (train, accel): {best_arima_rmse_train:.3f} m/s^2")
    print(f"Best ARIMA RMSE (test~10k)    : {best_arima_rmse_test:.3f} m/s^2")

    if np.isfinite(best_ar_rmse_test) and np.isfinite(sigma_acc):
        print(f"AR test RMSE / noise σ        : {best_ar_rmse_test / sigma_acc:.3f}")
    if np.isfinite(best_arima_rmse_test) and np.isfinite(sigma_acc):
        print(f"ARIMA test RMSE / noise σ     : {best_arima_rmse_test / sigma_acc:.3f}")

    # ---------- Per-step consecutive predictions for best AR & ARIMA ----------
    print("\n[CONSEC] Per-step predictions for best AR and ARIMA on a single long accel segment...")

    if (
        (not np.isnan(best_ar_rmse_test))
        and ("best_ar_test_row" in locals())
        and (best_arima_row_test is not None)
        and np.isfinite(best_arima_rmse_test)
    ):
        # ---- Best AR hyperparams from test suite ----
        best_p_ar = int(best_ar_test_row["order_p"])
        best_lam_ar = float(best_ar_test_row["lambda"])
        delta_used = float(DELTAS[0]) if RESAMPLE else None
        model_key_ar = (delta_used, best_p_ar, best_lam_ar)
        if model_key_ar not in models_ar:
            raise KeyError(f"Best AR model key {model_key_ar} not found in models_ar.")
        beta_ar_global = models_ar[model_key_ar]["beta"]

        # ---- Best ARIMA hyperparams from test suite ----
        best_p_arima = int(best_arima_row_test["p"])
        best_d_arima = int(best_arima_row_test["d"])
        best_q_arima = int(best_arima_row_test["q"])

        # Need enough history for lags + window
        min_needed = best_p_ar + CONSEC_POINTS + 5
        seg = pick_long_segment(seg_series, min_len=min_needed, random_seed=TEST_RANDOM_SEED)
        T = len(seg)
        print(f"[CONSEC] Using accel segment of length {T} for per-step evaluation")

        # ================= AR (ridge) one-step-ahead =================
        print(f"[CONSEC][AR] p={best_p_ar}, lambda={best_lam_ar}")

        X_full_ar, y_full_ar = build_ar_design_from_resampled(seg, best_p_ar)
        if X_full_ar.shape[0] < CONSEC_POINTS:
            raise ValueError(
                f"Not enough rows in X_full_ar ({X_full_ar.shape[0]}) "
                f"for CONSEC_POINTS={CONSEC_POINTS}."
            )

        X_sub_ar = X_full_ar[-CONSEC_POINTS:]
        y_true_ar = y_full_ar[-CONSEC_POINTS:]
        y_hat_ar = X_sub_ar @ beta_ar_global

        # ================= ARIMA one-step-ahead (fittedvalues) =================
        print(f"[CONSEC][ARIMA] order=({best_p_arima},{best_d_arima},{best_q_arima})")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model_arima = ARIMA(
                    seg,
                    order=(best_p_arima, best_d_arima, best_q_arima),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res_arima = model_arima.fit()
        except Exception as e:
            print(f"[CONSEC][ARIMA] Fit failed: {repr(e)}")
            res_arima = None

        if res_arima is not None:
            fitted = np.asarray(res_arima.fittedvalues, dtype="float64")
            n = min(CONSEC_POINTS, len(fitted))
            y_hat_arima = fitted[-n:]
            y_true_arima = seg[-n:]
        else:
            n = CONSEC_POINTS
            y_hat_arima = np.full(n, np.nan, dtype="float64")
            y_true_arima = seg[-n:]

        # ================= RMSE helpers =================
        def rmse(a, b):
            a = np.asarray(a, dtype="float64")
            b = np.asarray(b, dtype="float64")
            mask = np.isfinite(a) & np.isfinite(b)
            if not mask.any():
                return float("nan")
            diff = a[mask] - b[mask]
            return float(np.sqrt(np.mean(diff ** 2)))

        rmse_ar_window = rmse(y_true_ar, y_hat_ar)
        rmse_arima_window = rmse(y_true_arima, y_hat_arima)

        print(f"[CONSEC] Window RMSE (AR, per-step, accel)     : {rmse_ar_window:.3f} m/s^2")
        print(f"[CONSEC] Window RMSE (ARIMA, per-step, accel) : {rmse_arima_window:.3f} m/s^2")

        # ================= PLOTS =================
        x_ar = np.arange(len(y_true_ar))
        plt.figure(figsize=(10, 4))
        plt.plot(x_ar, y_true_ar, label="True accel (m/s^2)")
        plt.plot(x_ar, y_hat_ar, label=f"AR pred (p={best_p_ar}, λ={best_lam_ar})")
        plt.title(
            f"Best AR per-step accel predictions (last {len(y_true_ar)} usable points)\n"
            f"Window RMSE={rmse_ar_window:.3f} m/s^2"
        )
        plt.xlabel("Time index within window")
        plt.ylabel("Acceleration (m/s^2)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        x_arima = np.arange(len(y_true_arima))
        plt.figure(figsize=(10, 4))
        plt.plot(x_arima, y_true_arima, label="True accel (m/s^2)")
        plt.plot(
            x_arima,
            y_hat_arima,
            label=f"ARIMA fitted (p,d,q)=({best_p_arima},{best_d_arima},{best_q_arima})",
        )
        plt.title(
            f"Best ARIMA per-step accel predictions (last {len(y_true_arima)} usable points)\n"
            f"Window RMSE={rmse_arima_window:.3f} m/s^2"
        )
        plt.xlabel("Time index within window")
        plt.ylabel("Acceleration (m/s^2)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        print("[CONSEC] Skipping per-step plots; missing best AR/ARIMA results.")
