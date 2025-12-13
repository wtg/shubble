# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 16:35:31 2025

@author: willi
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "3000.csv"
VEHICLE_COL = "vehicle_id"
MAX_GAP_INCLUDE_S = 15.0
RESAMPLE = True
DELTAS = [5.0, 6.0, 8.0]
ORDERS = [1,2,3,4,5,6,7,8,9,10]
LAMBDAS = [None, 1e-4, 1e-3, 1e-2]
DEMEAN_PER_SEGMENT = True
INCLUDE_DT_LAGS = True
TEST_COUNT = 500
OUT_PATH = os.path.join(os.path.dirname(INPUT_PATH), "ar_resample_sweep_train_test.csv")
PLOT_RESID_VS_FITTED = True

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

def pick_time_col(df):
    for c in ["timestamp", "created_at", "time", "datetime"]:
        if c in df.columns: return c
    raise ValueError("No timestamp-like column (timestamp/created_at/time/datetime).")

def compute_velocity_mps(df):
    time_col = pick_time_col(df)
    out = df.copy()
    t_secs = to_epoch_seconds(out[time_col])
    mask = ~np.isnan(t_secs)
    out = out.loc[mask].copy()
    out["t"] = t_secs[mask].astype("int64")
    out = out.sort_values("t").reset_index(drop=True)
    out["dt_s"] = out["t"].diff()
    out.loc[out.index[0], "dt_s"] = np.nan
    if "speed_mph" in out.columns:
        out["v_mps"] = pd.to_numeric(out["speed_mph"], errors="coerce") * 0.44704
        need_geo = out["v_mps"].isna()
    else:
        out["v_mps"] = np.nan
        need_geo = pd.Series(True, index=out.index)
    if need_geo.any():
        if {"latitude", "longitude"}.issubset(out.columns):
            lat = pd.to_numeric(out["latitude"], errors="coerce")
            lon = pd.to_numeric(out["longitude"], errors="coerce")
            dist = np.full(len(out), np.nan, dtype=float)
            for i in range(1, len(out)):
                if (pd.notna(lat.iloc[i-1]) and pd.notna(lat.iloc[i]) and
                    pd.notna(lon.iloc[i-1]) and pd.notna(lon.iloc[i])):
                    dist[i] = haversine_meters(lat.iloc[i-1], lon.iloc[i-1], lat.iloc[i], lon.iloc[i])
            v_geo = np.where(out["dt_s"] > 0, dist / out["dt_s"], np.nan)
            out.loc[need_geo, "v_mps"] = v_geo[need_geo.values]
        else:
            raise ValueError("Missing speed_mph and latitude/longitude; cannot compute velocities.")
    out = out.loc[(out["dt_s"] > 0) & out["v_mps"].notna()].reset_index(drop=True)
    return out[["t", "v_mps", "dt_s"]]

def split_segments_on_gaps(t_series, v_series, dt_series, max_gap_include_s=15.0):
    t = np.asarray(t_series, dtype="int64")
    v = np.asarray(v_series, dtype="float64")
    dt = np.asarray(dt_series, dtype="float64")
    segments = []; start = 0
    for i in range(1, len(t)):
        if (not np.isfinite(dt[i])) or (dt[i] > max_gap_include_s):
            if i - start >= 2:
                segments.append((t[start:i], v[start:i]))
            start = i
    if len(t) - start >= 2:
        segments.append((t[start:], v[start:]))
    return segments

def resample_linear(t_seg, v_seg, delta_s):
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
    T = len(v_seg)
    if T <= p:
        q = p + (p if include_dt_lags else 0)
        return np.zeros((0, q), dtype="float64"), np.zeros((0,), dtype="float64")
    dt_seg = np.diff(t_seg, prepend=np.nan)
    rows = []; targets = []
    for i in range(p, T):
        v_lags = v_seg[i - p : i][::-1]
        if include_dt_lags:
            dt_lags = dt_seg[i - p : i][::-1]
            if np.isnan(dt_lags).any():
                med = np.nanmedian(dt_seg[1:i]) if i > 1 else 0.0
                dt_lags = np.where(np.isnan(dt_lags), (med if np.isfinite(med) else 0.0), dt_lags)
            feat = np.concatenate([v_lags, dt_lags])
        else:
            feat = v_lags
        rows.append(feat); targets.append(v_seg[i])
    X = np.asarray(rows, dtype="float64")
    y = np.asarray(targets, dtype="float64")
    return X, y

def ridge_fit(X, y, lam):
    if X.shape[0] == 0:
        return np.zeros((X.shape[1],), dtype="float64"), np.array([]), np.array([]), float("nan")
    XT = X.T
    A = XT @ X + (0 if lam in (None, 0) else lam) * np.eye(X.shape[1])
    b = XT @ y
    beta = np.linalg.solve(A, b)
    y_hat = X @ beta
    eps = y - y_hat
    mse = float(np.mean(eps**2))
    
    print("beta: ", beta )
    print("order:", X[0].shape)
    
    return beta, y_hat, eps, mse

def build_design_for_dataset(df_raw, vehicle_col, max_gap_include_s, resample, delta, demean_per_segment, include_dt_lags, orders):
    if vehicle_col and vehicle_col in df_raw.columns:
        groups = list(df_raw.groupby(vehicle_col))
    else:
        groups = [(None, df_raw)]
    segs = []
    debug_pairs = []
    for _, df_g in groups:
        dfv = compute_velocity_mps(df_g)
        raw_segs = split_segments_on_gaps(dfv["t"], dfv["v_mps"], dfv["dt_s"], max_gap_include_s)
        for (t_seg, v_seg) in raw_segs:
            if demean_per_segment and len(v_seg) > 0:
                v_seg = v_seg - np.nanmean(v_seg)
            if resample:
                t_grid, v_grid = resample_linear(t_seg, v_seg, float(delta))
                if len(v_grid) >= 5:
                    segs.append(("resampled", t_grid, v_grid))
                    debug_pairs.append({"delta": float(delta), "t_raw": t_seg, "v_raw": v_seg, "t_grid": t_grid, "v_grid": v_grid})
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
                Xs, ys = build_arx_design_irregular(t_arr, v_arr, p, include_dt_lags=include_dt_lags)
            if Xs.shape[0] > 0:
                X_list.append(Xs); y_list.append(ys)
        if X_list:
            X_all = np.vstack(X_list); y_all = np.concatenate(y_list)
        else:
            feat_dim = (p if resample else (p + (p if include_dt_lags else 0)))
            X_all = np.zeros((0, feat_dim)); y_all = np.zeros((0,))
        lagged_by_p[p] = (X_all, y_all)
    return lagged_by_p, debug_pairs

def split_train_test_by_earliest(df, test_count):
    time_col = pick_time_col(df)
    d = df.copy()
    d["_ts_"] = to_epoch_seconds(d[time_col])
    d = d.loc[~np.isnan(d["_ts_"])].sort_values("_ts_").reset_index(drop=True)
    test = d.iloc[:test_count].drop(columns=["_ts_"])
    train = d.iloc[test_count:].drop(columns=["_ts_"])
    return train, test

def evaluate_ar_train_test(df_train, df_test, vehicle_col, max_gap_include_s, resample, deltas, orders, lambdas, demean_per_segment=True, include_dt_lags=True):
    results = []
    models = {}
    debug_pairs_any = []

    delta_list = deltas if resample else [None]
    for delta in delta_list:
        lag_train, dbg_train = build_design_for_dataset(df_train, vehicle_col, max_gap_include_s, resample, delta, demean_per_segment, include_dt_lags, orders)
        lag_test, dbg_test = build_design_for_dataset(df_test,  vehicle_col, max_gap_include_s, resample, delta, demean_per_segment, include_dt_lags, orders)
        debug_pairs_any.extend(dbg_train[:2])
        for p in orders:
            Xtr, ytr = lag_train[p]
            Xte, yte = lag_test[p]
            for lam in lambdas:
                lam_key = 0.0 if lam in (None, 0) else float(lam)
                beta, yhat_tr, eps_tr, mse_tr = ridge_fit(Xtr, ytr, lam)
                if Xte.shape[0] > 0:
                    yhat_te = Xte @ beta
                    eps_te = yte - yhat_te
                    mse_te = float(np.mean(eps_te**2))
                else:
                    yhat_te = np.array([]); eps_te = np.array([]); mse_te = float("nan")
                results.append({
                    "mode": "RESAMPLED" if resample else ("RAW_ARX_dt" if include_dt_lags else "RAW_AR"),
                    "delta_s": (float(delta) if resample else None),
                    "order_p": int(p),
                    "lambda": lam_key,
                    "n_train": int(Xtr.shape[0]),
                    "n_test": int(Xte.shape[0]),
                    "MSE_train": mse_tr,
                    "RMSE_train": (float(np.sqrt(mse_tr)) if mse_tr==mse_tr else float("nan")),
                    "MSE_test": mse_te,
                    "RMSE_test": (float(np.sqrt(mse_te)) if mse_te==mse_te else float("nan")),
                })
                models[((float(delta) if resample else None), int(p), lam_key)] = {
                    "beta": beta,
                    "eps_train": eps_tr, "yhat_train": yhat_tr,
                    "eps_test": eps_te,  "yhat_test": yhat_te,
                    "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
                }
    res_df = pd.DataFrame(results).sort_values(["mode","delta_s","order_p","lambda"]).reset_index(drop=True)
    return res_df, models, debug_pairs_any

if __name__ == "__main__" or True:
    df_all = pd.read_csv(INPUT_PATH)
    df_train, df_test = split_train_test_by_earliest(df_all, TEST_COUNT)

    res_df, models, debug_pairs = evaluate_ar_train_test(
        df_train,
        df_test,
        VEHICLE_COL,
        MAX_GAP_INCLUDE_S,
        RESAMPLE,
        DELTAS,
        ORDERS,
        LAMBDAS,
        DEMEAN_PER_SEGMENT,
        INCLUDE_DT_LAGS
    )

    res_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved sweep to: {OUT_PATH}")
    if not res_df.empty:
        leaderboard = res_df.sort_values("RMSE_test").head(20)
        print("\nTop 20 (lowest TEST RMSE):")
        cols = ["mode","delta_s","order_p","lambda","n_train","n_test","MSE_train","RMSE_train","MSE_test","RMSE_test"]
        print(leaderboard[cols].to_string(index=False))
    else:
        print("No rows produced (check input columns and parameters).")

    if not res_df.empty:
        best = res_df.sort_values("RMSE_test").iloc[0]
        key = ((float(best["delta_s"]) if pd.notna(best["delta_s"]) else None),
               int(best["order_p"]),
               float(best["lambda"]))
        if key in models and models[key]["n_test"] > 0:
            eps = models[key]["eps_test"]
      
    delta_use = (float(best["delta_s"]) if pd.notna(best["delta_s"]) else None)
    p_use = int(best["order_p"])
    lam_use = float(best["lambda"])
    beta_use = models[key]["beta"]

    lag_test_only, _ = build_design_for_dataset(
        df_test,
        VEHICLE_COL,
        MAX_GAP_INCLUDE_S,
        RESAMPLE,
        delta_use,
        DEMEAN_PER_SEGMENT,
        INCLUDE_DT_LAGS,
        [p_use]
    )
    Xte_best, yte_best = lag_test_only[p_use]

    
    # ----- per-window mini-sequence plots: p context points, then target (actual vs predicted)
# uses the same best (delta, p, lambda) you selected above
    # one figure per window
    n_show = 3000
    use_delta = float(best["delta_s"]) if pd.notna(best["delta_s"]) else None
    p_use = int(best["order_p"])
    lam_use = float(best["lambda"])

    lag_test_only, _ = build_design_for_dataset(
        df_test, VEHICLE_COL, MAX_GAP_INCLUDE_S,
        RESAMPLE, use_delta, DEMEAN_PER_SEGMENT, INCLUDE_DT_LAGS, [p_use]
    )
    Xte_best, yte_best = lag_test_only[p_use]
    if Xte_best.shape[0] > 0:
        beta_best = models[(use_delta, p_use, lam_use)]["beta"]
        yhat_te_best = Xte_best @ beta_best

        n_show = min(n_show, Xte_best.shape[0])
        for i in range(n_show):
            if RESAMPLE or not INCLUDE_DT_LAGS:
                v_lags = Xte_best[i][:p_use]
            else:
                v_lags = Xte_best[i][:p_use]
            ctx = v_lags[::-1]
            x_ctx = np.arange(1, p_use + 1)
            x_tgt = p_use + 1

            plt.figure(figsize=(4.2, 3.2))
            plt.plot(x_ctx, ctx, marker="o", linestyle="-", label="context")
            plt.plot([x_tgt], [yte_best[i]], marker="s", linestyle="", label="actual")
            plt.plot([x_tgt], [yhat_te_best[i]], marker="x", linestyle="", label="pred")
            plt.axvline(p_use + 0.5, linestyle="--", alpha=0.4)
            y_all = np.concatenate([ctx, [yte_best[i]], [yhat_te_best[i]]])
            pad = 0.05 * (np.max(y_all) - np.min(y_all) + 1e-9)
            plt.ylim(np.min(y_all) - pad, np.max(y_all) + pad)
            plt.xlim(0.5, p_use + 1.5)
            plt.xticks([1, p_use, p_use + 1], ["1", f"{p_use}", "t"])
            r = yte_best[i] - yhat_te_best[i]
            plt.title(f"window {i+1}  (p={p_use}, Δ={use_delta}, ε={r:.3f})")
            plt.xlabel("position in window")
            plt.ylabel("v (m/s)")
            plt.legend()
            plt.tight_layout()
            plt.show()
    else:
        print("No test windows to plot for the chosen configuration.")


    if Xte_best.shape[0] > 0:
        yhat_best = Xte_best @ beta_use
        n_show = min(20, Xte_best.shape[0])

        # Build a display table with lag columns (v_{t-1}, ..., v_{t-p})
        lag_cols = [f"v(t-{k})" for k in range(1, p_use+1)]
        df_show = pd.DataFrame(Xte_best[:n_show, :p_use], columns=lag_cols)
        df_show["y_pred"] = yhat_best[:n_show]
        df_show["y_actual"] = yte_best[:n_show]
        df_show["residual"] = df_show["y_actual"] - df_show["y_pred"]

        print(f"\nFirst {n_show} windows for best config "
              f"(mode={best['mode']}, Δ={best['delta_s']}, p={p_use}, λ={lam_use}):")
        print(df_show.to_string(index=False))

        x_idx = np.arange(1, n_show + 1)
        
        plt.figure()
        plt.plot(x_idx, df_show["y_actual"].to_numpy(), marker="o", label="actual")
        plt.plot(x_idx, df_show["y_pred"].to_numpy(), marker="x", label="predicted")
        plt.xlabel("window index (first 20)")
        plt.ylabel("velocity (m/s)")
        plt.title(f"Actual vs Predicted — first {n_show} windows (p={p_use}, Δ={best['delta_s']}, λ={lam_use})")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        plt.figure()
        plt.bar(x_idx, df_show["residual"].to_numpy())
        plt.axhline(0.0)
        plt.xlabel("window index (first 20)")
        plt.ylabel("residual (y - ŷ)")
        plt.title(f"Residuals — first {n_show} windows (p={p_use}, Δ={best['delta_s']}, λ={lam_use})")
        plt.tight_layout()
        plt.show()
        
        plt.figure()
        plt.scatter(df_show["y_actual"].to_numpy(), df_show["y_pred"].to_numpy(), s=30)
        lims = [
            min(df_show["y_actual"].min(), df_show["y_pred"].min()),
            max(df_show["y_actual"].max(), df_show["y_pred"].max())
        ]
        plt.plot(lims, lims)
        plt.xlabel("actual")
        plt.ylabel("predicted")
        plt.title(f"Predicted vs Actual — first {n_show} windows (p={p_use}, Δ={best['delta_s']}, λ={lam_use})")
        plt.tight_layout()
        plt.show()
  
        x_idx = np.arange(1, n_show + 1)
        
        plt.figure()
        plt.plot(x_idx, df_show["y_actual"].to_numpy(), marker="o", label="actual")
        plt.plot(x_idx, df_show["y_pred"].to_numpy(), marker="x", label="predicted")
        plt.xlabel("window index (first 20)")
        plt.ylabel("velocity (m/s)")
        plt.title(f"Actual vs Predicted — first {n_show} windows (p={p_use}, Δ={best['delta_s']}, λ={lam_use})")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        plt.figure()
        plt.bar(x_idx, df_show["residual"].to_numpy())
        plt.axhline(0.0)
        plt.xlabel("window index (first 20)")
        plt.ylabel("residual (y - ŷ)")
        plt.title(f"Residuals — first {n_show} windows (p={p_use}, Δ={best['delta_s']}, λ={lam_use})")
        plt.tight_layout()
        plt.show()
        
        plt.figure()
        plt.scatter(df_show["y_actual"].to_numpy(), df_show["y_pred"].to_numpy(), s=30)
        lims = [
            min(df_show["y_actual"].min(), df_show["y_pred"].min()),
            max(df_show["y_actual"].max(), df_show["y_pred"].max())
        ]
        plt.plot(lims, lims)  # y = x reference
        plt.xlabel("actual")
        plt.ylabel("predicted")
        plt.title(f"Predicted vs Actual — first {n_show} windows (p={p_use}, Δ={best['delta_s']}, λ={lam_use})")
        plt.tight_layout()
        plt.show()


    else:
        print("\nBest config produced no test windows to display.")