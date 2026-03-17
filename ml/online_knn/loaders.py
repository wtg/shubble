"""
Data loaders for online KNN.

CSV loader is used for testing. Interface is designed so a real-time loader
(DB or stream) can be plugged in later without changing the rest of the pipeline.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.cache import RAW_CSV, load_cached_csv


# Required columns for raw location data (used by both CSV and future real-time source)
RAW_COLUMNS = ["vehicle_id", "latitude", "longitude", "timestamp"]


def load_raw_locations_csv(
    path: Optional[Path] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load raw vehicle locations from CSV (for testing).

    Uses ml/cache/shared/locations_raw.csv by default. Caller can filter by date
    (e.g. campus today) if needed.

    Args:
        path: CSV path. If None, uses RAW_CSV (ml/cache/shared/locations_raw.csv).
        use_cache: If True and path is None, use cache loader (with logging).
            If False, read from path (or RAW_CSV) without cache layer.

    Returns:
        DataFrame with columns: vehicle_id, latitude, longitude, timestamp.
        timestamp is datetime64.
    """
    if path is None:
        path = RAW_CSV
    if use_cache and path == RAW_CSV:
        df = load_cached_csv(path, "raw locations")
        if df is not None:
            return df
    if not path.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    for col in RAW_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    return df[RAW_COLUMNS]


class RawLocationsLoader:
    """
    Pluggable loader for raw location data.

    Use for testing with CSV, or replace with a real-time implementation that
    returns DataFrames (or async iterator of DataFrames) for the same schema.
    """

    def load(self) -> pd.DataFrame:
        """
        Load raw locations (vehicle_id, latitude, longitude, timestamp).

        Override for real-time: e.g. query DB for today, or return incremental
        batch from a stream.
        """
        return load_raw_locations_csv(use_cache=True)
