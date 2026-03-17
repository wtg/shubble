"""
Pipeline: load raw data -> compute ETA labels -> fit KNN -> predict.

Uses CSV for testing; data source is pluggable for real-time later.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ml.online_knn.loaders import load_raw_locations_csv, RawLocationsLoader
from ml.online_knn.labels import compute_trajectory_etas
from ml.online_knn.model import TimeToStopKNN


def run_from_csv(
    stops: List[Tuple[float, float]],
    csv_path: Optional[Path] = None,
    n_neighbors: int = 5,
    stop_radius_km: float = 0.005,
    date_filter: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> TimeToStopKNN:
    """
    Load raw locations from CSV, compute time-to-stop labels, fit KNN.

    For testing. For real-time, use a custom RawLocationsLoader and call
    compute_trajectory_etas + model.fit() on incremental data.

    Args:
        stops: List of (lat, lon) for the 2 (or N) known stops.
        csv_path: Path to raw CSV. If None, uses ml/cache/shared/locations_raw.csv.
        n_neighbors: K for KNN.
        stop_radius_km: Consider "at stop" when within this distance (km).
        date_filter: If set (e.g. '2025-07-31'), keep only rows with that date in timestamp.
            Enables same-day simulation when CSV spans multiple days.
        max_rows: If set, use only this many rows (for fast testing on large CSV).

    Returns:
        Fitted TimeToStopKNN model.
    """
    df = load_raw_locations_csv(path=csv_path, use_cache=(csv_path is None))
    if df.empty:
        raise ValueError("No raw location data loaded")

    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    if date_filter:
        df["_date"] = pd.to_datetime(df["timestamp"]).dt.date
        df = df[df["_date"].astype(str) == date_filter].drop(columns=["_date"])
        if df.empty:
            raise ValueError(f"No rows for date_filter={date_filter}")

    labeled = compute_trajectory_etas(df, stops=stops, radius_km=stop_radius_km)
    model = TimeToStopKNN(n_neighbors=n_neighbors)
    model.fit(labeled, eta_cols=[f"eta_seconds_stop_{i}" for i in range(len(stops))])
    return model


def run_with_loader(
    loader: RawLocationsLoader,
    stops: List[Tuple[float, float]],
    n_neighbors: int = 5,
    stop_radius_km: float = 0.005,
) -> TimeToStopKNN:
    """
    Run pipeline with a pluggable loader (e.g. CSV or future DB/stream).

    Real-time: implement a loader that returns today's data (or incremental batch);
    call this periodically or on each batch to refit/update the model.
    """
    df = loader.load()
    if df.empty:
        raise ValueError("Loader returned no data")
    labeled = compute_trajectory_etas(df, stops=stops, radius_km=stop_radius_km)
    model = TimeToStopKNN(n_neighbors=n_neighbors)
    model.fit(labeled, eta_cols=[f"eta_seconds_stop_{i}" for i in range(len(stops))])
    return model
