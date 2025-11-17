import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "shubble_october.csv"
VEHICLE_COL = "vehicle_id"

MAX_GAP_INCLUDE_S = 15
RESAMPLE = True
DELTAS = [5.0]
ORDERS = [4,6]
LAMBDAS = [None, 1e-4, 1e-3, 1e-2]

DEMEAN_PER_SEGMENT = True
INCLUDE_DT_LAGS = False   

OUT_PATH = os.path.join(os.path.dirname(INPUT_PATH), "ar_resample_sweep.csv")
PLOT_RESID_VS_FITTED = True


HOLDOUT_N = 150000           
HOLDOUT_SEED = 42         


def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def to_epoch_seconds(series):
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    secs = ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).astype("Int64")
    return secs.astype("float64").to_numpy()

def compute_velocity_mps(df):
    """
    Compute velocity (m/s) + dt from either speed_mph or GPS and time.
    Returns columns: ['t', 'v_mps', 'dt_s']
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

    # Prefer ECU speed_mph if present
    if "speed_mph" in out.columns:
        out["v_mps"] = pd.to_numeric(out["speed_mph"], errors="coerce") * 0.44704
        need_geo = out["v_mps"].isna()
    else:
        out["v_mps"] = np.nan
        need_geo = pd.Series(True, index=out.index)

    # Fallback to GPS distance / dt if needed
    if need_geo.any():
        if {"latitude", "longitude"}.issubset(out.columns):
            lat = pd.to_numeric(out["latitude"], errors="coerce")
            lon = pd.to_numeric(out["longitude"], errors="coerce")
            dist = np.full(len(out), np.nan, dtype=float)
            for i in range(1, len(out)):
                if (pd.notna(lat.iloc[i-1]) and pd.notna(lat.iloc[i]) and
                    pd.notna(lon.iloc[i-1]) and pd.notna(lon.iloc[i])):
                    dist[i] = haversine_meters(
                        lat.iloc[i-1], lon.iloc[i-1],
                        lat.iloc[i],   lon.iloc[i]
                    )
            v_geo = np.where(out["dt_s"] > 0, dist / out["dt_s"], np.nan)
            out.loc[need_geo, "v_mps"] = v_geo[need_geo.values]
        else:
            raise ValueError("Missing speed_mph and latitude/longitude; cannot compute velocities.")

    out = out.loc[(out["dt_s"] > 0) & out["v_mps"].notna()].reset_index(drop=True)
    return out[["t", "v_mps", "dt_s"]]

def split_segments_on_gaps(t_series, v_series, dt_series, max_gap_include_s=15.0):
    """
    Split into segments when dt > max_gap_include_s or dt is non-finite.
    """
    t = np.asarray(t_series, dtype="int64")
    v = np.asarray(v_series, dtype="float64")
    dt = np.asarray(dt_series, dtype="float64")

    segments = []
    start = 0
    for i in range(1, len(t)):
        if (not np.isfinite(dt[i])) or (dt[i] > max_gap_include_s):
            if i - start >= 2:
                segments.append((t[start:i], v[start:i]))
            start = i
    if len(t) - start >= 2:
        segments.append((t[start:], v[start:]))

    return segments

def resample_linear(t_seg, v_seg, delta_s):
    """
    Linearly resample segment onto a regular time grid with spacing delta_s.
    """
    if len(t_seg) < 2:
        return np.array([], dtype="int64"), np.array([], dtype="float64")

    t0, tN = int(t_seg[0]), int(t_seg[-1])
    if (tN - t0) < delta_s:
        return np.array([], dtype="int64"), np.array([], dtype="float64")

    step = int(round(delta_s))
    kmax = (tN - t0) // step
    t_grid = t0 + np.arange(0, kmax + 1) * step
    v_grid = np.interp(t_grid, t_seg, v_seg)

    return t_grid.astype("int64"), v_grid.astype("float64")

def build_ar_design_from_resampled(v_grid, p):
    """
    Build AR(p) design matrix from a regularly sampled series v_grid.
    X rows: [v(t-Δ), v(t-2Δ), ..., v(t-pΔ)]
    y: v(t)
    """
    T = len(v_grid)
    if T <= p:
        return np.zeros((0, p), dtype="float64"), np.zeros((0,), dtype="float64")

    X = np.zeros((T - p, p), dtype="float64")
    y = np.zeros((T - p,), dtype="float64")
    for i in range(p, T):
        X[i - p, :] = v_grid[i - p : i][::-1]
        y[i - p] = v_grid[i]

    return X, y

def build_arx_design_irregular(t_seg, v_seg, p, include_dt_lags=True):
    """
    Build ARX design matrix for irregularly sampled data:
      features = [v_lags, (optionally) dt_lags]
      target   = v(t)
    """
    T = len(v_seg)
    if T <= p:
        q = p + (p if include_dt_lags else 0)
        return np.zeros((0, q), dtype="float64"), np.zeros((0,), dtype="float64")

    dt_seg = np.diff(t_seg, prepend=np.nan)
    rows = []
    targets = []

    for i in range(p, T):
        v_lags = v_seg[i - p : i][::-1]
        if include_dt_lags:
            dt_lags = dt_seg[i - p : i][::-1]
            if np.isnan(dt_lags).any():
                med = np.nanmedian(dt_seg[1:i]) if i > 1 else 0.0
                dt_lags = np.where(
                    np.isnan(dt_lags),
                    (med if np.isfinite(med) else 0.0),
                    dt_lags
                )
            feat = np.concatenate([v_lags, dt_lags])
        else:
            feat = v_lags

        rows.append(feat)
        targets.append(v_seg[i])

    X = np.asarray(rows, dtype="float64")
    y = np.asarray(targets, dtype="float64")
    return X, y
def ridge_fit(X, y, lam):
    """
    Solve (X^T X + λI) β = X^T y.
    Returns beta, y_hat, eps, mse on the provided (X, y).
    """
    if X.shape[0] == 0:
        beta = np.zeros((X.shape[1],), dtype="float64") if X.shape[1] > 0 else np.array([])
        return beta, np.array([]), np.array([]), float("nan")

    XT = X.T
    A = XT @ X + (0 if lam in (None, 0) else lam) * np.eye(X.shape[1])
    b = XT @ y
    beta = np.linalg.solve(A, b)
    y_hat = X @ beta
    eps = y - y_hat
    mse = float(np.mean(eps**2))
    return beta, y_hat, eps, mse



def evaluate_ar_pipeline(df_raw,
                         vehicle_col,
                         max_gap_include_s,
                         resample,
                         deltas,
                         orders,
                         lambdas,
                         demean_per_segment=True,
                         include_dt_lags=True,
                         holdout_n=0,
                         holdout_seed=42):
    """
    Build AR windows, optionally hold out `holdout_n` random rows per (delta, p)
    for test, and evaluate MSE/RMSE on both train and test.
    """

    results = []
    models = {}
    debug_pairs = []

    # For reproducible splits
    rng = np.random.default_rng(holdout_seed)

    # Group by vehicle, if present
    if vehicle_col and vehicle_col in df_raw.columns:
        groups = list(df_raw.groupby(vehicle_col))
    else:
        groups = [(None, df_raw)]

    delta_list = deltas if resample else [None]

    for delta in delta_list:
        segs = []

        for _, df_g in groups:
            dfv = compute_velocity_mps(df_g)
            raw_segs = split_segments_on_gaps(
                dfv["t"], dfv["v_mps"], dfv["dt_s"], max_gap_include_s
            )
            for (t_seg, v_seg) in raw_segs:
                if demean_per_segment and len(v_seg) > 0:
                    v_seg = v_seg - np.nanmean(v_seg)
                if resample:
                    t_grid, v_grid = resample_linear(t_seg, v_seg, float(delta))
                    if len(v_grid) >= 5:
                        segs.append(("resampled", t_grid, v_grid))
                        debug_pairs.append({
                            "delta": float(delta),
                            "t_raw": t_seg,
                            "v_raw": v_seg,
                            "t_grid": t_grid,
                            "v_grid": v_grid
                        })
                else:
                    if len(v_seg) >= 3:
                        segs.append(("raw", t_seg, v_seg))

        lagged_by_p = {}
        for p in orders:
            X_list, y_list = [], []
            for mode, t_arr, v_arr in segs:
                if mode == "resampled":
                    Xs, ys = build_ar_design_from_resampled(v_arr, p)
                else:
                    Xs, ys = build_arx_design_irregular(
                        t_arr, v_arr, p, include_dt_lags=include_dt_lags
                    )
                if Xs.shape[0] > 0:
                    X_list.append(Xs)
                    y_list.append(ys)

            if X_list:
                X_all = np.vstack(X_list)
                y_all = np.concatenate(y_list)
            else:
                feat_dim = (p if resample else (p + (p if include_dt_lags else 0)))
                X_all = np.zeros((0, feat_dim), dtype="float64")
                y_all = np.zeros((0,), dtype="float64")

            # --- NEW: train/test split --- #
            n_total = X_all.shape[0]
            if holdout_n > 0 and n_total > holdout_n:
                idx = np.arange(n_total)
                rng.shuffle(idx)
                test_idx = idx[:holdout_n]
                train_idx = idx[holdout_n:]
                X_train, y_train = X_all[train_idx], y_all[train_idx]
                X_test,  y_test  = X_all[test_idx],  y_all[test_idx]
            else:
                # Not enough data to have a separate test set
                X_train, y_train = X_all, y_all
                X_test  = np.zeros((0, X_all.shape[1]), dtype="float64")
                y_test  = np.zeros((0,), dtype="float64")

            lagged_by_p[p] = (X_all, y_all, X_train, y_train, X_test, y_test)

        # 3) Fit models for each (p, lambda) using *train* only, evaluate test
        for p in orders:
            X_all, y_all, X_train, y_train, X_test, y_test = lagged_by_p[p]
            n_total = int(X_all.shape[0])
            n_train = int(X_train.shape[0])
            n_test  = int(X_test.shape[0])

            for lam in lambdas:
                lam_key = 0.0 if lam in (None, 0) else float(lam)

                # Train fit
                beta, y_hat_train, eps_train, mse_train = ridge_fit(X_train, y_train, lam)
                rmse_train = float(np.sqrt(mse_train)) if mse_train == mse_train else float("nan")

                # Test evaluation
                if n_test > 0:
                    y_hat_test = X_test @ beta
                    eps_test = y_test - y_hat_test
                    mse_test = float(np.mean(eps_test**2))
                    rmse_test = float(np.sqrt(mse_test))
                else:
                    y_hat_test = np.array([])
                    eps_test = np.array([])
                    mse_test = float("nan")
                    rmse_test = float("nan")

                # For backward compatibility: MSE/RMSE = test if available, else train
                mse_main = mse_test if n_test > 0 else mse_train
                rmse_main = rmse_test if n_test > 0 else rmse_train

                results.append({
                    "mode": "RESAMPLED" if resample else ("RAW_ARX_dt" if include_dt_lags else "RAW_AR"),
                    "delta_s": (float(delta) if resample else None),
                    "order_p": int(p),
                    "lambda": lam_key,
                    "n_rows": n_total,
                    "n_train": n_train,
                    "n_test": n_test,
                    "MSE_train": mse_train,
                    "RMSE_train": rmse_train,
                    "MSE_test": mse_test,
                    "RMSE_test": rmse_test,
                    "MSE": mse_main,
                    "RMSE": rmse_main,
                })

                models[((float(delta) if resample else None), int(p), lam_key)] = {
                    "beta": beta,
                    "mse_train": mse_train,
                    "rmse_train": rmse_train,
                    "mse_test": mse_test,
                    "rmse_test": rmse_test,
                    "n_rows": n_total,
                    "n_train": n_train,
                    "n_test": n_test,
                    "feat_dim": int(X_train.shape[1]) if X_train.ndim == 2 else 0,
                    "eps_train": eps_train,
                    "y_hat_train": y_hat_train,
                    "eps_test": eps_test,
                    "y_hat_test": y_hat_test,
                }

    res_df = pd.DataFrame(results).sort_values(
        ["mode", "delta_s", "order_p", "lambda"]
    ).reset_index(drop=True)

    return res_df, models, debug_pairs




