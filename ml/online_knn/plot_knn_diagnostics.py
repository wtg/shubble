"""
Diagnostic plots to understand why KNN ETA prediction performs poorly.

Generates 4 plot types, each saved separately for Student Union and Colonie (8 files total):
  1. Predicted vs true ETA (scatter) — shows bias and spread; points off diagonal = error.
  2. Error vs minutes_since_last_stop — when in the run we fail most.
  3. Neighbor disagreement vs error — when k neighbors have very different ETAs, prediction is unstable.
  4. Same location, different ETAs — at nearby (lat,lon), true ETAs vary a lot (fundamental ambiguity).

Run: python -m ml.online_knn.plot_knn_diagnostics

Requires preprocessed CSV. Outputs: ml/cache/shared/knn_diagnostics_*_student_union.png, *_colonie.png
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
from ml.online_knn.labels import compute_trajectory_etas
from ml.online_knn.model import TimeToStopKNN
from ml.online_knn.stops_config import STOPS_STUDENT_UNION_COLONIE

# Match run_learning_curve
STOPS = STOPS_STUDENT_UNION_COLONIE
STOP_RADIUS_KM = 0.08
ROUTE_FILTER = "WEST"
MAX_DIST_TO_ROUTE_KM = 0.08
MAX_GAP_SEC = 3600
MIN_DATE = "2025-09-01"
MAX_RAW_ROWS = 200_000
TRAIN_SIZE = 10_000
TEST_SIZE = 2000
N_NEIGHBORS = 3
USE_TIME_WEIGHTING = True
RANDOM_SEED = 42
# Only evaluate on rows where true ETA <= this (seconds) so we see meaningful range
MAX_TRUE_ETA_SEC = 1800  # 30 min

CACHE_DIR = Path(__file__).parent.parent / "cache" / "shared"


def load_and_prepare():
    """Load data, compute ETAs and minutes_since_last_stop, return train/test."""
    df = load_cached_csv(PREPROCESSED_CSV, "preprocessed locations")
    if df is None or df.empty:
        raise SystemExit("Preprocessed CSV not found. Run: python -m ml.pipelines preprocess")
    df = clean_preprocessed_for_route(df, ROUTE_FILTER, max_dist_to_route_km=MAX_DIST_TO_ROUTE_KM)
    df = df[pd.to_datetime(df["timestamp"]).dt.date >= pd.to_datetime(MIN_DATE).date()]
    if len(df) > MAX_RAW_ROWS:
        df = df.head(MAX_RAW_ROWS)
    df = add_segment_by_time_gap(df, max_gap_seconds=MAX_GAP_SEC, segment_column="segment_id")
    labeled = compute_trajectory_etas(
        df, stops=STOPS, radius_km=STOP_RADIUS_KM, segment_column="segment_id"
    )
    eta_cols = [f"eta_seconds_stop_{i}" for i in range(len(STOPS))]
    valid = labeled.dropna(subset=eta_cols).sort_values("timestamp").reset_index(drop=True)
    valid = add_minutes_since_last_stop_passage(
        valid, stops=STOPS, radius_km=STOP_RADIUS_KM, segment_column="segment_id"
    )
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(valid))
    valid = valid.iloc[perm].reset_index(drop=True)
    test_df = valid.iloc[-TEST_SIZE:].copy()
    train_df = valid.iloc[: -TEST_SIZE].head(TRAIN_SIZE)
    return train_df, test_df, eta_cols


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("Install matplotlib: pip install matplotlib")

    print("Loading data and fitting KNN...")
    train_df, test_df, eta_cols = load_and_prepare()
    model = TimeToStopKNN(n_neighbors=N_NEIGHBORS)
    model.fit(
        train_df,
        eta_cols=eta_cols,
        time_col="minutes_since_last_stop" if USE_TIME_WEIGHTING else None,
    )
    pred = model.predict_batch(
        test_df,
        lat_col="latitude",
        lon_col="longitude",
        time_col="minutes_since_last_stop" if USE_TIME_WEIGHTING else None,
    )
    ind = model.get_neighbor_indices_batch(test_df, k=N_NEIGHBORS)  # (n_test, k)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stop_names = ["Student Union", "Colonie"]
    stop_slugs = ["student_union", "colonie"]

    t_min_col = "minutes_since_last_stop"
    t_min_vals = test_df[t_min_col].values if t_min_col in test_df.columns else np.zeros(len(test_df))

    for stop_idx, name, slug in zip(range(2), stop_names, stop_slugs):
        # ---- 1. Predicted vs True (scatter) ----
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
        t = test_df[eta_cols[stop_idx]].values
        p = pred[eta_cols[stop_idx]].values
        soon = t <= MAX_TRUE_ETA_SEC if MAX_TRUE_ETA_SEC else np.ones(len(t), dtype=bool)
        valid = soon & np.isfinite(t) & np.isfinite(p)
        if valid.sum() == 0:
            ax.text(0.5, 0.5, "No valid points", ha="center", va="center")
        else:
            t_min, p_min = t[valid] / 60, p[valid] / 60
            ax.scatter(t_min, p_min, alpha=0.4, s=8)
            lim = max(t_min.max(), p_min.max(), 1)
            ax.plot([0, lim], [0, lim], "k--", alpha=0.7, label="Perfect")
            ax.set_aspect("equal")
        ax.set_xlabel(f"True ETA to {name} (min)")
        ax.set_ylabel(f"Predicted ETA (min)")
        ax.set_title(f"{name}: predicted vs true ETA")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = CACHE_DIR / f"knn_diagnostics_1_predicted_vs_true_{slug}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  Saved {path}")

    for stop_idx, name, slug in zip(range(2), stop_names, stop_slugs):
        # ---- 2. Error vs minutes_since_last_stop ----
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
        t = test_df[eta_cols[stop_idx]].values
        p = pred[eta_cols[stop_idx]].values
        soon = (t <= MAX_TRUE_ETA_SEC) if MAX_TRUE_ETA_SEC else np.ones(len(t), dtype=bool)
        valid = soon & np.isfinite(t) & np.isfinite(p)
        if valid.sum() >= 10:
            err_min = np.abs(t[valid] - p[valid]) / 60
            ax.scatter(t_min_vals[valid], err_min, alpha=0.35, s=8)
            bins = np.linspace(0, np.nanmax(t_min_vals[valid]) + 1, 12)
            bin_idx = np.digitize(t_min_vals[valid], bins) - 1
            bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
            means = np.array([err_min[bin_idx == i].mean() for i in range(len(bins) - 1)])
            counts = np.array([(bin_idx == i).sum() for i in range(len(bins) - 1)])
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.plot(bin_centers[counts >= 5], means[counts >= 5], "r-o", linewidth=2, label="Mean error (binned)")
        ax.set_xlabel("Minutes since last stop")
        ax.set_ylabel("Absolute error (min)")
        ax.set_title(f"{name}: error vs position in run")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = CACHE_DIR / f"knn_diagnostics_2_error_vs_minutes_since_stop_{slug}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  Saved {path}")

    for stop_idx, name, slug in zip(range(2), stop_names, stop_slugs):
        # ---- 3. Neighbor disagreement vs error ----
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
        neighbor_etas = model._y[ind, stop_idx] / 60
        std_neighbor = np.nanstd(neighbor_etas, axis=1)
        t = test_df[eta_cols[stop_idx]].values / 60
        p = pred[eta_cols[stop_idx]].values / 60
        err_min = np.abs(t - p)
        soon = (test_df[eta_cols[stop_idx]].values <= MAX_TRUE_ETA_SEC) if MAX_TRUE_ETA_SEC else np.ones(len(t), dtype=bool)
        valid = soon & np.isfinite(t) & np.isfinite(p) & np.isfinite(std_neighbor)
        if valid.sum() >= 10:
            ax.scatter(std_neighbor[valid], err_min[valid], alpha=0.4, s=10)
        ax.set_xlabel("Std of k neighbors' true ETAs (min)")
        ax.set_ylabel("Absolute error (min)")
        ax.set_title(f"{name}: neighbor disagreement vs error")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = CACHE_DIR / f"knn_diagnostics_3_neighbor_disagreement_vs_error_{slug}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  Saved {path}")

    for stop_idx, name, slug in zip(range(2), stop_names, stop_slugs):
        # ---- 4. Same location, different ETAs (ambiguity) ----
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
        neighbor_etas_min = model._y[ind, stop_idx] / 60
        range_neighbor = np.nanmax(neighbor_etas_min, axis=1) - np.nanmin(neighbor_etas_min, axis=1)
        soon = (test_df[eta_cols[stop_idx]].values <= MAX_TRUE_ETA_SEC) if MAX_TRUE_ETA_SEC else np.ones(len(neighbor_etas_min), dtype=bool)
        valid = soon & np.isfinite(range_neighbor)
        if valid.sum() > 0:
            ax.hist(range_neighbor[valid], bins=30, edgecolor="black", alpha=0.7)
            ax.axvline(np.median(range_neighbor[valid]), color="red", linestyle="--", label=f"Median range = {np.median(range_neighbor[valid]):.0f} min")
        ax.set_xlabel("Range of k neighbors' true ETAs at same (lat,lon) (min)")
        ax.set_ylabel("Count of test points")
        ax.set_title(f"{name}: same location, different ETAs")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = CACHE_DIR / f"knn_diagnostics_4_same_location_eta_range_{slug}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  Saved {path}")

    print("Done. Open the knn_diagnostics_*_student_union.png and knn_diagnostics_*_colonie.png files in ml/cache/shared/.")


if __name__ == "__main__":
    main()
