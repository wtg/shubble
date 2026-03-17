"""
Learning curve: start with 100 points, add 2 every step, refit KNN, measure error.

X-axis = training set size (100, 102, 104, ...)
Y-axis = prediction error (MAE in seconds) on a fixed test set.

Why error can be huge (10k+ seconds): KNN uses only (lat, lon). The same location
can have very different ETAs (e.g. 2 min when approaching the stop, 2 hr when
just left). Neighbors mix different moments, so predictions are unstable. To see
error on "soon" predictions only, set MAX_TRUE_ETA_SEC below.

Run: python -m ml.online_knn.run_learning_curve
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ml.cache import PREPROCESSED_CSV, load_cached_csv
from ml.online_knn.clean import (
    add_minutes_since_last_stop_passage,
    add_segment_by_time_gap,
    clean_preprocessed_for_route,
)
from ml.online_knn.loaders import load_raw_locations_csv
from ml.online_knn.labels import compute_trajectory_etas
from ml.online_knn.model import TimeToStopKNN
from ml.online_knn.stops_config import STOPS_STUDENT_UNION_COLONIE

# Only Student Union and Colonie (no other stops; routes.json not used)
STOPS = STOPS_STUDENT_UNION_COLONIE

# Radius (km) to count as "at stop".
STOP_RADIUS_KM = 0.08  # 80 m

# Use NORTH so both Student Union and Colonie are on the route (Colonie is not on WEST)
ROUTE_FILTER = "NORTH"
# Max distance to route (km) — drop points farther than this.
MAX_DIST_TO_ROUTE_KM = 0.08  # 80 m
# Max time gap (seconds) between consecutive points (same vehicle). Larger = new segment (shuttle left and came back); ETAs are not computed across this gap.
MAX_GAP_SEC = 3600  # 1 hour
# None = use all dates (more data); set to "2025-07-31" to restrict to one day
DATE_FILTER = None
# Only use data on or after this date (YYYY-MM-DD). None = no cutoff.
MIN_DATE = "2025-09-01"
MAX_RAW_ROWS = 500_000  # use up to 500k raw rows so we get many points with both ETAs
MAX_TRAIN_SIZE = 30_000  # cap learning curve at this many training points
INITIAL_TRAIN_SIZE = 100
STEP = 2
TEST_SIZE = 500
PRINT_EVERY = 300  # print progress every this many points
N_NEIGHBORS = 3  # k: number of nearest (lat, lon) neighbors used to predict ETA
# If True, weight neighbors by minutes_since_last_stop so "same location, same phase" counts more. Set False if MAE is worse.
USE_TIME_WEIGHTING = True
# MAE is computed only on test rows where both true ETAs <= this (seconds).
# 30 min keeps error in a meaningful range; None = use all (often 100k+ s MAE).
MAX_TRUE_ETA_SEC = 1800  # 30 minutes
# How to split train vs test:
#   "temporal" = train on first N rows (chronological), test on last TEST_SIZE rows. Realistic for
#     online learning: "more data" = more history; we never see future or shuffled data in production.
#     MAE often plateaus because test points' k-nearest neighbors are usually in recent history.
#   "random" = shuffle then split; MAE can improve with more data but is not realistic for online.
SPLIT_MODE = "temporal"  # "temporal" (realistic) | "random" (debug only)
RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).parent.parent / "cache" / "shared"


def main():
    if ROUTE_FILTER:
        print(f"Loading preprocessed locations (route filter = {ROUTE_FILTER})...")
        df = load_cached_csv(PREPROCESSED_CSV, "preprocessed locations")
        if df is None or df.empty:
            raise SystemExit(
                "Preprocessed CSV not found or empty. Run: python -m ml.pipelines preprocess"
            )
        if "route" not in df.columns or "dist_to_route" not in df.columns:
            raise SystemExit(
                "Preprocessed CSV needs 'route' and 'dist_to_route'. Re-run: python -m ml.pipelines preprocess"
            )
        before = len(df)
        df = clean_preprocessed_for_route(
            df, ROUTE_FILTER, max_dist_to_route_km=MAX_DIST_TO_ROUTE_KM
        )
        print(f"  Kept {len(df):,} rows on route {ROUTE_FILTER} within {MAX_DIST_TO_ROUTE_KM} km (dropped {before - len(df):,} off-route/far)")
    else:
        print("Loading raw locations...")
        df = load_raw_locations_csv(use_cache=True)
        if df.empty:
            raise SystemExit("No data in locations_raw.csv. Run the load pipeline or add a CSV.")

    if len(df) > MAX_RAW_ROWS:
        df = df.head(MAX_RAW_ROWS)
    if DATE_FILTER is not None:
        df["_date"] = pd.to_datetime(df["timestamp"]).dt.date
        df = df[df["_date"].astype(str) == DATE_FILTER].drop(columns=["_date"])
    if MIN_DATE is not None:
        df = df[pd.to_datetime(df["timestamp"]).dt.date >= pd.to_datetime(MIN_DATE).date()]
    if df.empty:
        raise SystemExit("No rows after filtering (check DATE_FILTER, MIN_DATE, ROUTE_FILTER).")

    # Segment by time gap so we don't compute ETAs across "left and came back" gaps
    df = add_segment_by_time_gap(df, max_gap_seconds=MAX_GAP_SEC, segment_column="segment_id")
    n_segments = df["segment_id"].nunique()
    print(f"  Segmented into {n_segments:,} runs (gap > {MAX_GAP_SEC}s starts new segment)")

    print("Computing trajectory ETAs...")
    labeled = compute_trajectory_etas(
        df, stops=STOPS, radius_km=STOP_RADIUS_KM, segment_column="segment_id"
    )
    eta_cols = [f"eta_seconds_stop_{i}" for i in range(len(STOPS))]
    # Only rows with both ETAs so we can compute error for both stops
    valid = labeled.dropna(subset=eta_cols).sort_values("timestamp").reset_index(drop=True)
    if USE_TIME_WEIGHTING:
        valid = add_minutes_since_last_stop_passage(
            valid, stops=STOPS, radius_km=STOP_RADIUS_KM, segment_column="segment_id"
        )
    n_valid = len(valid)
    min_required = 50 + 50  # initial 50 + test 50
    if n_valid < min_required:
        raise SystemExit(
            f"Need at least {min_required} rows with both ETAs; got {n_valid}."
        )

    # When data is limited, use smaller initial train and test so we still get a curve
    if n_valid >= INITIAL_TRAIN_SIZE + TEST_SIZE:
        initial_train = INITIAL_TRAIN_SIZE
        test_size = TEST_SIZE
    else:
        initial_train = 50
        test_size = min(100, n_valid - initial_train - 10)
        if test_size < 50:
            test_size = n_valid - initial_train
        print(f"Using initial_train={initial_train}, test_size={test_size} (only {n_valid} rows with both ETAs).")

    # Train/test split
    if SPLIT_MODE == "random":
        rng = np.random.default_rng(RANDOM_SEED)
        perm = rng.permutation(len(valid))
        valid = valid.iloc[perm].reset_index(drop=True)
        test_df = valid.iloc[-test_size:].copy()
        train_pool = valid.iloc[: -test_size]
        # Shuffle train pool so head(n) is a random subset; then more data = better coverage
        train_perm = rng.permutation(len(train_pool))
        train_pool = train_pool.iloc[train_perm].reset_index(drop=True)
        print(f"  Split: random (seed={RANDOM_SEED}), train_pool={len(train_pool):,}, test={test_size}")
    else:
        # Temporal: last test_size rows as test (realistic for online learning)
        train_pool = valid.iloc[: -test_size]
        test_df = valid.iloc[-test_size:].copy()
        print(f"  Split: temporal (last {test_size} rows = test), train_pool={len(train_pool):,}")

    max_train = min(len(train_pool), MAX_TRAIN_SIZE)

    train_sizes = list(range(initial_train, max_train + 1, STEP))
    if not train_sizes:
        raise SystemExit("No training sizes in range.")

    # errors_per_stop[i] = list of MAE values (one per train size) for stop i
    errors_per_stop = [[] for _ in eta_cols]
    stop_names = ["Student Union", "Colonie"]
    print(f"Refitting KNN every {STEP} points from {initial_train} to {max_train}...")
    for n in train_sizes:
        train_sub = train_pool.head(n)
        model = TimeToStopKNN(n_neighbors=min(N_NEIGHBORS, len(train_sub)))
        model.fit(
            train_sub,
            eta_cols=eta_cols,
            time_col="minutes_since_last_stop" if USE_TIME_WEIGHTING else None,
        )
        pred = model.predict_batch(
            test_df,
            lat_col="latitude",
            lon_col="longitude",
            time_col="minutes_since_last_stop" if USE_TIME_WEIGHTING else None,
        )

        for stop_idx, c in enumerate(eta_cols):
            t = test_df[c].values
            p = pred[c].values
            if MAX_TRUE_ETA_SEC is not None:
                soon = (t <= MAX_TRUE_ETA_SEC)
            else:
                soon = np.ones(len(test_df), dtype=bool)
            valid_mask = soon & ~(np.isnan(t) | np.isnan(p))
            if valid_mask.any():
                mae = float(np.abs(t[valid_mask] - p[valid_mask]).mean())
            else:
                mae = np.nan
            errors_per_stop[stop_idx].append(mae)
        mean_mae = float(np.nanmean([errors_per_stop[i][-1] for i in range(len(eta_cols))])) if any(np.isfinite(errors_per_stop[i][-1]) for i in range(len(eta_cols))) else np.nan
        if n % PRINT_EVERY == 0:
            if np.isfinite(mean_mae):
                print(f"  n = {n:,}  MAE = {mean_mae:.1f} s")
            else:
                print(f"  n = {n:,}  MAE = (no valid test points)" + (f" within {MAX_TRUE_ETA_SEC}s" if MAX_TRUE_ETA_SEC else ""))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("Install matplotlib to generate the plot: pip install matplotlib")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stop_idx, name in enumerate(stop_names):
        slug = "student_union" if stop_idx == 0 else "colonie"
        fig, ax = plt.subplots()
        ax.plot(train_sizes, errors_per_stop[stop_idx], color="steelblue", linewidth=1.5)
        ax.set_xlabel("Training set size (number of data points)")
        ax.set_ylabel("Mean absolute error (seconds)")
        ax.set_title(f"KNN prediction error vs training data — {name}\n(split={SPLIT_MODE}, evaluate every {STEP} points)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = OUTPUT_DIR / f"knn_error_vs_data_{slug}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"Plot saved to {path}")


if __name__ == "__main__":
    main()
