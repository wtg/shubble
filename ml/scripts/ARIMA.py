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


def ridge_fit(X, y, lam):
    if X.shape[0] == 0:
        return (
            (np.zeros((X.shape[1],), dtype="float64") if X.shape[1] > 0 else np.array([])),
            np.array([]),
            np.array([]),
            float("nan"),
        )

    print("dimension:", X.shape)
    XT = X.T
    A = XT @ X + (0 if lam in (None, 0) else lam) * np.eye(X.shape[1])
    b = XT @ y
    beta = np.linalg.solve(A, b)
    y_hat = X @ beta
    eps = y - y_hat
    mse = float(np.mean(eps ** 2))
    print("dim_phi: ", beta.shape)
    return beta, y_hat, eps, mse

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
        # drop edges if you want to avoid convolution edge effects
        resid = resid[window:-window] if len(resid) > 2 * window else resid
        all_resid.append(resid)

    if not all_resid:
        return np.nan, np.nan

    all_resid = np.concatenate(all_resid)
    sigma_mps = float(np.std(all_resid))
    sigma_mph = sigma_mps * 2.23694
    return sigma_mps, sigma_mph

# ---------------- AR (RIDGE) PIPELINE ----------------
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

# ---------------- ARIMA (INTEGRATED MOVING AVG) PIPELINE ----------------
def evaluate_arima_resampled(
    df_raw,
    vehicle_col,
    max_gap_include_s,
    deltas,
    p_list,
    d_list,
    q_list,
    demean_per_segment=True,
):
    """
    For each delta and ARIMA(p, d, q) combination:
      - build resampled segments across all vehicles
      - fit ARIMA(p, d, q) on each segment (if long enough)
      - aggregate residuals to compute global RMSE
      - aggregate AIC/BIC (mean over segments)
    """
    results = []
    models = {}

    # Group by vehicle if column present
    if vehicle_col and vehicle_col in df_raw.columns:
        groups = list(df_raw.groupby(vehicle_col))
    else:
        groups = [(None, df_raw)]

    for delta in deltas:
        seg_series = []  # store only s_grid per segment; time not needed for ARIMA

        # Build resampled segments across all vehicles
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

        if not seg_series:
            # No usable segments for this delta
            for p in p_list:
                for d in d_list:
                    for q in q_list:
                        results.append(
                            {
                                "mode": "ARIMA_RESAMPLED",
                                "delta_s": float(delta),
                                "p": int(p),
                                "d": int(d),
                                "q": int(q),
                                "n_obs": 0,
                                "n_segments": 0,
                                "AIC_mean": float("nan"),
                                "BIC_mean": float("nan"),
                                "MSE": float("nan"),
                                "RMSE": float("nan"),
                            }
                        )
                        models[(float(delta), int(p), int(d), int(q))] = {
                            "params": None,
                            "mse": float("nan"),
                            "rmse": float("nan"),
                            "aic_mean": float("nan"),
                            "bic_mean": float("nan"),
                            "n_obs": 0,
                            "n_segments": 0,
                        }
            continue

        # For each ARIMA(p, d, q) combination, fit on all segments and aggregate
        for p in p_list:
            for d in d_list:
                for q in q_list:
                    all_resid = []
                    aics = []
                    bics = []
                    n_obs = 0
                    n_seg_fit = 0
                    example_params = None

                    min_len = max(p + d + q + 3, 10)

                    for s_seg in seg_series:
                        if len(s_seg) < min_len:
                            continue
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    "ignore", category=ConvergenceWarning
                                )
                                model = ARIMA(s_seg, order=(p, d, q))
                                res = model.fit()
                        except Exception:
                            # Skip segments where ARIMA fails to converge or throws
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
                        example_params = np.asarray(res.params, dtype="float64")

                    if n_obs == 0:
                        mse = float("nan")
                        rmse = float("nan")
                    else:
                        all_resid_concat = np.concatenate(all_resid)
                        mse = float(np.mean(all_resid_concat ** 2))
                        rmse = float(np.sqrt(mse))

                    aic_mean = float(np.mean(aics)) if aics else float("nan")
                    bic_mean = float(np.mean(bics)) if bics else float("nan")

                    results.append(
                        {
                            "mode": "ARIMA_RESAMPLED",
                            "delta_s": float(delta),
                            "p": int(p),
                            "d": int(d),
                            "q": int(q),
                            "n_obs": int(n_obs),
                            "n_segments": int(n_seg_fit),
                            "AIC_mean": aic_mean,
                            "BIC_mean": bic_mean,
                            "MSE": mse,
                            "RMSE": rmse,
                        }
                    )

                    models[(float(delta), int(p), int(d), int(q))] = {
                        "params": example_params,
                        "mse": mse,
                        "rmse": rmse,
                        "aic_mean": aic_mean,
                        "bic_mean": bic_mean,
                        "n_obs": int(n_obs),
                        "n_segments": int(n_seg_fit),
                    }

    res_df = pd.DataFrame(results).sort_values(
        ["mode", "delta_s", "p", "d", "q"]
    ).reset_index(drop=True)
    return res_df, models
#---------------------------ARIMA APROX--------------------------------------
def build_resampled_segments(df_raw, vehicle_col, max_gap_include_s, delta_s, demean_per_segment=True):
    """
    Returns a list of 1D numpy arrays (speed series), one per resampled segment.
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
            t_grid, s_grid = resample_linear(t_seg, s_seg, float(delta_s))
            if len(s_grid) >= 5:
                seg_series.append(s_grid.astype("float64"))

    print(f"Built {len(seg_series)} resampled segments for Δ={delta_s}.")
    total_len = sum(len(s) for s in seg_series)
    print(f"Total resampled length = {total_len} points.")
    return seg_series

import random
def quick_arima_eval(seg_series,
                     p, d, q,
                     max_segments=800,
                     maxiter=30,      # kept in signature but NOT used (for compatibility)
                     min_len=10,
                     use_longest=True,
                     random_seed=42):
    """
    Quickly evaluate one ARIMA(p,d,q) on a subset of segments.

    Returns (rmse, aic_mean, bic_mean, n_obs, n_segments_used).

    This version:
      - does NOT pass maxiter to ARIMA.fit() (your statsmodels doesn't support it),
      - prints debug info if no segments are successfully fit.
    """
    if not seg_series:
        print(f"[quick_arima_eval] seg_series is empty for (p,d,q)=({p},{d},{q})")
        return float("nan"), float("nan"), float("nan"), 0, 0

    # Choose which segments to use
    if use_longest:
        segs_sorted = sorted(seg_series, key=len, reverse=True)
        segs_used = segs_sorted[:max_segments]
    else:
        random.seed(random_seed)
        segs_used = random.sample(seg_series, min(max_segments, len(seg_series)))

    all_resid = []
    aics = []
    bics = []
    n_obs = 0
    n_seg_fit = 0

    # minimum length needed for ARIMA(p,d,q)
    min_len_eff = max(min_len, p + d + q + 3)

    fail_count = 0

    for idx, s in enumerate(segs_used):
        if len(s) < min_len_eff:
            continue

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
                # NOTE: no maxiter/disp here; your ARIMA.fit() doesn't accept them
                res = model.fit()
        except Exception as e:
            fail_count += 1
            if fail_count <= 5:
                print(
                    f"[quick_arima_eval] ARIMA failed for segment idx={idx}, "
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
            f"fail_count={fail_count}, used={len(segs_used)} segments, "
            f"min_len_eff={min_len_eff}"
        )
        return float("nan"), float("nan"), float("nan"), 0, 0

    resid_concat = np.concatenate(all_resid)
    mse = float(np.mean(resid_concat ** 2))
    rmse = float(np.sqrt(mse))
    aic_mean = float(np.mean(aics)) if aics else float("nan")
    bic_mean = float(np.mean(bics)) if bics else float("nan")

    return rmse, aic_mean, bic_mean, n_obs, n_seg_fit



# ---------------- MAIN ----------------
if __name__ == "__main__" or True:
    df_raw = pd.read_csv(INPUT_PATH)

    seg_series = build_resampled_speed_segments(
        df_raw,
        VEHICLE_COL,
        MAX_GAP_INCLUDE_S,
        DELTAS[0],
        DEMEAN_PER_SEGMENT,
    )
    
    all_s = np.concatenate(seg_series)
    print("Global speed std:", np.std(all_s), "m/s")

    print(f"Total resampled segments: {len(seg_series)}")
    if seg_series:
        lengths = [len(s) for s in seg_series]
        print(f"Segment length stats: min={min(lengths)}, max={max(lengths)}, "
              f"median={np.median(lengths):.1f}")
    
    

    sigma_mps, sigma_mph = estimate_speed_noise_from_smoothing(
        seg_series,
        window=4,  # 10 samples * 5s = 50s smoothing if delta=5
    )
    print(f"\n[NOISE] Estimated σ ≈ {sigma_mps:.3f} m/s ≈ {sigma_mph:.3f} mph")
    
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
        leaderboard = res_df_ar.sort_values("RMSE").head(20)
        print("\n[AR] Top 20 (lowest RMSE):")
        cols = ["mode", "delta_s", "order_p", "lambda", "n_rows", "MSE", "RMSE"]
        print(leaderboard[cols].to_string(index=False))
    else:
        print("[AR] No rows produced (check input columns and parameters).")

    # Debug plots for a few segments (same as your original)
    if RESAMPLE and debug_pairs:
        show_n = min(3, len(debug_pairs))
        for i in range(show_n):
            dp = debug_pairs[i]
            df_raw_seg = pd.DataFrame(
                {"t_raw": dp["t_raw"], "speed_raw": dp["speed_raw"]}
            )
            df_grid_seg = pd.DataFrame(
                {"T_grid": dp["t_grid"], "speed_grid": dp["speed_grid"]}
            )
            print(f"\nSegment {i+1} (delta={dp['delta']}) raw head:")
            print(df_raw_seg.head().to_string(index=False))
            print(f"\nSegment {i+1} (delta={dp['delta']}) resampled head:")
            print(df_grid_seg.head().to_string(index=False))

            plt.figure()
            plt.plot(
                dp["t_raw"],
                dp["speed_raw"],
                marker="o",
                linestyle="",
                label="raw (t, speed)",
            )
            plt.plot(
                dp["t_grid"],
                dp["speed_grid"],
                marker="x",
                label="resampled (T, speed)",
            )
            plt.xlabel("time (s)")
            plt.ylabel("speed (m/s)")
            plt.title(f"Segment {i+1} resampling overlay (Δ={dp['delta']})")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # Residual diagnostics for best AR config
    if not res_df_ar.empty:
        best_ar = res_df_ar.sort_values("RMSE").iloc[0]
        key_ar = (
            (float(best_ar["delta_s"]) if pd.notna(best_ar["delta_s"]) else None),
            int(best_ar["order_p"]),
            float(best_ar["lambda"]),
        )
        if key_ar in models_ar and models_ar[key_ar]["n_rows"] > 0:
            eps = models_ar[key_ar]["eps"]
            yhat = models_ar[key_ar]["y_hat"]

            plt.figure()
            plt.hist(eps, bins=30)
            plt.xlabel("residual ε = y - ŷ")
            plt.ylabel("count")
            plt.title(
                "Residuals histogram — best AR config "
                f"(mode={best_ar['mode']}, Δ={best_ar['delta_s']}, "
                f"p={best_ar['order_p']}, λ={best_ar['lambda']})"
            )
            plt.tight_layout()
            plt.show()

            if PLOT_RESID_VS_FITTED:
                plt.figure()
                plt.scatter(yhat, eps, s=12)
                plt.axhline(0.0)
                plt.xlabel("ŷ (fitted)")
                plt.ylabel("ε (residual)")
                plt.title(
                    f"Residuals vs Fitted — best AR config (mode={best_ar['mode']})"
                )
                plt.tight_layout()
                plt.show()

        # Print best AR weights
        if key_ar in models_ar:
            beta = models_ar[key_ar]["beta"]
            print("\n[AR] Best config weights (phi):")
            print(
                f"mode={best_ar['mode']}, Δ={best_ar['delta_s']}, "
                f"p={best_ar['order_p']}, λ={best_ar['lambda']}"
            )
            print(beta)
            
    # ---------- ARIMA (integrated moving avg) pipeline ----------
    res_df_arima, models_arima = evaluate_arima_resampled(
        df_raw,
        VEHICLE_COL,
        MAX_GAP_INCLUDE_S,
        DELTAS,
        ARIMA_P,
        ARIMA_D,
        ARIMA_Q,
        DEMEAN_PER_SEGMENT,
    )

    res_df_arima.to_csv(OUT_PATH_ARIMA, index=False)
    print(f"\n[ARIMA] Saved sweep to: {OUT_PATH_ARIMA}")

    if not res_df_arima.empty:
        leaderboard_arima = res_df_arima.sort_values("RMSE").head(20)
        print("\n[ARIMA] Top 20 (lowest RMSE):")
        cols = [
            "mode",
            "delta_s",
            "p",
            "d",
            "q",
            "n_obs",
            "n_segments",
            "AIC_mean",
            "BIC_mean",
            "MSE",
            "RMSE",
        ]
        print(leaderboard_arima[cols].to_string(index=False))
        

