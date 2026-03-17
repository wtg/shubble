"""
Online KNN model for predicting time to known stops.

Uses same-day (or provided) raw location data; routes are not used.
Data source is pluggable: CSV for testing, DB/stream for real-time later.
"""
from ml.online_knn.clean import (
    add_minutes_from_segment_start,
    add_segment_by_time_gap,
    clean_preprocessed_for_route,
)
from ml.online_knn.loaders import load_raw_locations_csv, RawLocationsLoader
from ml.online_knn.labels import compute_trajectory_etas
from ml.online_knn.model import TimeToStopKNN
from ml.online_knn.pipeline import run_from_csv, run_with_loader

__all__ = [
    "add_minutes_from_segment_start",
    "add_segment_by_time_gap",
    "clean_preprocessed_for_route",
    "load_raw_locations_csv",
    "RawLocationsLoader",
    "compute_trajectory_etas",
    "TimeToStopKNN",
    "run_from_csv",
    "run_with_loader",
]
