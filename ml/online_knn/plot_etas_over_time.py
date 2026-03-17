"""
Plot time-to-stop_0 and time-to-stop_1 over time so you can see why values get large.

Uses the same data and cleaning as run_learning_curve. Picks one segment (one
continuous run) and plots true ETAs vs timestamp. Run:

  python -m ml.online_knn.plot_etas_over_time

Output: ml/cache/shared/etas_over_time.png
"""
from pathlib import Path

import pandas as pd

from ml.cache import PREPROCESSED_CSV, load_cached_csv
from ml.online_knn.clean import (
    add_minutes_since_last_stop_passage,
    add_segment_by_time_gap,
    clean_preprocessed_for_route,
)
from ml.online_knn.labels import compute_trajectory_etas
from ml.online_knn.stops_config import STOPS_STUDENT_UNION_COLONIE

# Match run_learning_curve config
STOPS = STOPS_STUDENT_UNION_COLONIE
STOP_RADIUS_KM = 0.08   # 80 m
# Use NORTH so both Student Union and Colonie are on the route
ROUTE_FILTER = "NORTH"
MAX_DIST_TO_ROUTE_KM = 0.08   # 80 m
MAX_GAP_SEC = 3600
MIN_DATE = "2025-09-01"
MAX_POINTS_PLOT = 3000  # plot at most this many points (one segment) so the figure is readable
OUTPUT_PATH = Path(__file__).parent.parent / "cache" / "shared" / "etas_over_time.png"


def main():
    print("Loading and cleaning (same as learning curve)...")
    df = load_cached_csv(PREPROCESSED_CSV, "preprocessed locations")
    if df is None or df.empty:
        raise SystemExit("Preprocessed CSV not found. Run: python -m ml.pipelines preprocess")
    if "route" not in df.columns or "dist_to_route" not in df.columns:
        raise SystemExit("Preprocessed CSV needs 'route' and 'dist_to_route'.")
    df = clean_preprocessed_for_route(df, ROUTE_FILTER, max_dist_to_route_km=MAX_DIST_TO_ROUTE_KM)
    if MIN_DATE is not None:
        df = df[pd.to_datetime(df["timestamp"]).dt.date >= pd.to_datetime(MIN_DATE).date()]
    if df.empty:
        raise SystemExit("No rows after filtering.")
    df = add_segment_by_time_gap(df, max_gap_seconds=MAX_GAP_SEC, segment_column="segment_id")

    print("Computing trajectory ETAs...")
    labeled = compute_trajectory_etas(
        df, stops=STOPS, radius_km=STOP_RADIUS_KM, segment_column="segment_id"
    )
    eta_cols = [f"eta_seconds_stop_{i}" for i in range(len(STOPS))]
    # Use rows that have both ETAs so the plot shows full segments
    valid = labeled.dropna(subset=eta_cols).sort_values("timestamp").reset_index(drop=True)
    if valid.empty:
        raise SystemExit("No rows with both ETAs. Check MIN_DATE and data.")

    valid = add_minutes_since_last_stop_passage(
        valid, stops=STOPS, radius_km=STOP_RADIUS_KM, segment_column="segment_id"
    )

    # Pick one segment with enough points (e.g. the largest)
    seg_sizes = valid.groupby("segment_id").size()
    seg_id = seg_sizes.idxmax()
    plot_df = valid[valid["segment_id"] == seg_id].copy()
    if len(plot_df) > MAX_POINTS_PLOT:
        plot_df = plot_df.iloc[: MAX_POINTS_PLOT]
    plot_df = plot_df.sort_values("timestamp").reset_index(drop=True)
    # X-axis: time (monotonic minutes from segment start) so the run flows left to right
    t0 = pd.to_datetime(plot_df["timestamp"]).iloc[0]
    time_min = (pd.to_datetime(plot_df["timestamp"]) - t0).dt.total_seconds() / 60
    # Feature: minutes since last stop passage (resets when passing a stop), graphed alongside ETAs
    minutes_since_last_stop = plot_df["minutes_since_last_stop"].values
    print(f"Plotting segment {seg_id} ({len(plot_df)} points, {time_min.iloc[-1]:.0f} min span)...")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("Install matplotlib: pip install matplotlib")

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    eta0_min = plot_df["eta_seconds_stop_0"] / 60
    eta1_min = plot_df["eta_seconds_stop_1"] / 60

    ax0.plot(time_min, eta0_min, color="C0", alpha=0.8, linewidth=0.8, label="ETA to Student Union")
    ax0.plot(time_min, minutes_since_last_stop, color="gray", linestyle=":", linewidth=1, alpha=0.9, label="Minutes since last stop")
    ax0.axhline(30, color="gray", linestyle="--", alpha=0.5, label="30 min ref")
    ax0.set_ylabel("Minutes")
    ax0.set_title("Student Union: ETA and minutes since last stop over time")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right")
    ax0.set_ylim(bottom=0)

    ax1.plot(time_min, eta1_min, color="C1", alpha=0.8, linewidth=0.8, label="ETA to Colonie")
    ax1.plot(time_min, minutes_since_last_stop, color="gray", linestyle=":", linewidth=1, alpha=0.9, label="Minutes since last stop")
    ax1.axhline(30, color="gray", linestyle="--", alpha=0.5, label="30 min ref")
    ax1.set_xlabel("Time (minutes from segment start)")
    ax1.set_ylabel("Minutes")
    ax1.set_title("Colonie: ETA and minutes since last stop over time")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_ylim(bottom=0)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=120)
    plt.close()
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
