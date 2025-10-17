# %% script


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- params -----------------------------------
INPUT_PATH  = "100.csv"                 
OUTPUT_PATH = "sequences_100.csv"     
L           = 4                         
T           = 15.0                       
PREVIEW_ROWS = 10                       

TIME_COL    = "timestamp"   
SECONDS_COL = None   #
DIST_COL    = None   
LAT_COL     = "latitude" 
LON_COL     = "longitude"   
VEH_COL     = "vehical_id"   

SHOW_PLOTS  = True   
# ----------------------------------------------

def _norm_key(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "")

def _find_col(candidates, columns):
    cols_norm = {_norm_key(c): c for c in columns}
    for cand in candidates:
        key = _norm_key(cand)
        if key in cols_norm:
            return cols_norm[key]
    return None

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def load_and_prepare(input_path,
                     time_col=None, seconds_col=None,
                     dist_col=None, lat_col=None, lon_col=None,
                     veh_col=None):
    
    df = pd.read_csv(input_path)

    # Auto-detect columns if not provided
    if time_col is None:
        time_col = _find_col(
            ["timestamp","time","datetime","ts","event_time","created_at","updated_at"],
            df.columns
        )
    if seconds_col is None:
        seconds_col = _find_col(["seconds","secs","epoch_s","time_s"], df.columns)
    if dist_col is None:
        dist_col = _find_col(
            ["distance_km","distance","dist_km","odometer_km","odometer",
             "cum_distance_km","cumdistkm","meters","distance_m"], df.columns
        )
    if lat_col is None:
        lat_col = _find_col(["lat","latitude"], df.columns)
    if lon_col is None:
        lon_col = _find_col(["lon","lng","longitude"], df.columns)
    if veh_col is None:
        veh_col = _find_col(["vehicle_id","vehicle","bus_id","shuttle_id","id"], df.columns)

    # Build numeric time seconds
    if time_col is not None and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.dropna(subset=[time_col])
        df["_time_s"] = df[time_col].astype("int64") / 1e9  
    elif seconds_col is not None and seconds_col in df.columns:
        df["_time_s"] = pd.to_numeric(df[seconds_col], errors="coerce")
        df = df.dropna(subset=["_time_s"])
    else:

        df["_time_s"] = np.arange(len(df), dtype=float)
        
    if (lat_col is not None and lat_col in df.columns and
          lon_col is not None and lon_col in df.columns):

        sort_keys = ([veh_col] if (veh_col and veh_col in df.columns) else []) + ["_time_s"]
        df = df.sort_values(sort_keys).reset_index(drop=True)
        if not veh_col or veh_col not in df.columns:
            lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
            lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()
            ok = ~np.isnan(lat) & ~np.isnan(lon)
            df = df[ok].copy()
            lat = lat[ok]; lon = lon[ok]
            dstep = np.zeros(len(df))
            if len(df) > 1:
                dstep[1:] = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
            df["_distance_km"] = np.cumsum(dstep)
        else:
            df["_distance_km"] = np.nan
            for vid, g in df.groupby(veh_col, sort=False):
                idx = g.index.to_numpy()
                lat = pd.to_numeric(g[lat_col], errors="coerce").to_numpy()
                lon = pd.to_numeric(g[lon_col], errors="coerce").to_numpy()
                ok = ~np.isnan(lat) & ~np.isnan(lon)
                sub_idx = idx[ok]
                sub_lat, sub_lon = lat[ok], lon[ok]
                dstep = np.zeros(len(sub_idx))
                if len(sub_idx) > 1:
                    dstep[1:] = _haversine_km(sub_lat[:-1], sub_lon[:-1], sub_lat[1:], sub_lon[1:])
                df.loc[sub_idx, "_distance_km"] = np.cumsum(dstep)
            df = df.dropna(subset=["_distance_km"])
    else:
        # data integrity
        df["_distance_km"] = 0.0

    # Clean + sort
    df = df.dropna(subset=["_time_s","_distance_km"]).copy()
    sort_keys = ([veh_col] if (veh_col and veh_col in df.columns) else []) + ["_time_s"]
    df = df.sort_values(sort_keys).reset_index(drop=True)
    return df, veh_col

def build_windows(df, veh_col, L, T_sec):
    rows = []

    def process_group(g):
        times = g["_time_s"].to_numpy()
        dists = g["_distance_km"].to_numpy()
        n = len(g)
        for start in range(0, n - L + 1):
            end = start + L
            tw = times[start:end]
            dw = dists[start:end]
            dt = np.diff(tw)
            if (dt <= 0).any():
                continue
            if np.max(dt) > T_sec:
                continue
            # normalize to first point
            t0, d0 = tw[0], dw[0]
            t_rel = tw - t0
            d_rel = dw - d0
            row = []
            for i in range(L):
                row.extend([float(t_rel[i]), float(d_rel[i])])
            rows.append(row)

    if veh_col and veh_col in df.columns:
        for _, g in df.groupby(veh_col, sort=False):
            process_group(g)
    else:
        process_group(df)

    # headers
    cols = []
    for i in range(1, L+1):
        cols += [f"Timedelta{i}", f"Distance{i}"]
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out["Timedelta1"] = 0.0
        out["Distance1"] = 0.0
    return out

def plot_first_five(dataset, L):
    k = min(4, len(dataset))
    for r in range(k):
        row = dataset.iloc[r]
        times = [row[f"Timedelta{i}"] for i in range(1, L+1)]
        dists = [row[f"Distance{i}"] for i in range(1, L+1)]
        plt.figure()
        plt.plot(times, dists, marker="o")
        plt.title(f"Row {r+1}: Distance vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance from start (km)")
        plt.grid(True)
        plt.show()

# %% Run
df, veh = load_and_prepare(INPUT_PATH, TIME_COL, SECONDS_COL, DIST_COL, LAT_COL, LON_COL, VEH_COL)
seq = build_windows(df, veh, L=L, T_sec=T)
seq.to_csv(OUTPUT_PATH, index=False)
print(f"Saved: {OUTPUT_PATH}")
print(seq.head(PREVIEW_ROWS).to_string(index=False))

if SHOW_PLOTS and len(seq) > 0:
    plot_first_five(seq, L)
