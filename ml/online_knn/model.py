"""
KNN model that predicts time (seconds) to each of two known stops from (lat, lon).

Optional: use a time feature (e.g. minutes_since_last_stop) to weight neighbors
so "same location, different moment in the run" is disambiguated and the model
can follow the predictable sawtooth pattern.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Haversine expects lat, lon in radians for sklearn
LATLON_TO_RADIANS = np.pi / 180.0

# When using time weighting: exp(-|delta_min| / TIME_SCALE_MIN). 15 min = weight ~0.37
TIME_SCALE_MIN = 15.0


class TimeToStopKNN:
    """
    Predicts seconds-to-stop for 2 (or N) known stops using KNN on (lat, lon).
    Optional time_column: weight neighbors by time so the model can use "where we
    are in the run" and learn the predictable ETA pattern.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "haversine",
        time_scale_min: float = TIME_SCALE_MIN,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.time_scale_min = time_scale_min
        self._nn: Optional[NearestNeighbors] = None
        self._X_rad: Optional[np.ndarray] = None
        self._t_min: Optional[np.ndarray] = None  # minutes_since_last_stop per row (when time_col used)
        self._eta_columns: List[str] = []
        self._n_stops: int = 0
        self._time_col: Optional[str] = None

    def fit(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        eta_cols: Optional[List[str]] = None,
        time_col: Optional[str] = None,
    ) -> "TimeToStopKNN":
        """
        Fit KNN on labeled points. If time_col is set (e.g. 'minutes_since_last_stop'),
        neighbors are time-weighted at predict so the model can use position-in-run.
        """
        if eta_cols is None:
            eta_cols = [c for c in df.columns if c.startswith("eta_seconds_stop_")]
        if not eta_cols:
            raise ValueError("No eta_seconds_stop_* columns found")
        self._eta_columns = sorted(eta_cols)
        self._n_stops = len(self._eta_columns)
        self._time_col = time_col

        # Drop rows where all ETAs are NaN (no label)
        mask = df[self._eta_columns].notna().any(axis=1)
        sub = df.loc[mask].copy()
        if sub.empty:
            raise ValueError("No rows with at least one non-NaN ETA")

        lat = sub[lat_col].values
        lon = sub[lon_col].values
        X = np.column_stack([lat * LATLON_TO_RADIANS, lon * LATLON_TO_RADIANS])
        self._X_rad = X

        if time_col and time_col in sub.columns:
            self._t_min = sub[time_col].values.astype(float)
        else:
            self._t_min = None

        self._nn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(sub)),
            metric=self.metric,
            algorithm="ball_tree",
        )
        self._nn.fit(X)
        self._y = sub[self._eta_columns].values
        return self

    def _weights_for_query(self, ind: np.ndarray, t_query_min: Optional[float]) -> np.ndarray:
        """Weights for neighbors: 1 if no time; exp(-|t_query - t_i|/scale) if time used."""
        if t_query_min is None or self._t_min is None:
            return np.ones(len(ind))
        t_neigh = self._t_min[ind]
        valid = ~np.isnan(t_neigh)
        w = np.zeros(len(ind))
        w[valid] = np.exp(-np.abs(t_query_min - t_neigh[valid]) / self.time_scale_min)
        return w

    def predict(
        self,
        latitude: float,
        longitude: float,
        k: Optional[int] = None,
        aggregate: str = "median",
        time_min: Optional[float] = None,
    ) -> Tuple[float, ...]:
        """
        Predict seconds to each stop. Uses median (or mean) of the k neighbors' ETAs.
        time_min is accepted for API compatibility but not used for aggregation.
        """
        if self._nn is None or self._X_rad is None:
            raise RuntimeError("Model not fitted")
        k = k or self.n_neighbors
        k = min(k, self._X_rad.shape[0])

        q = np.array([[latitude * LATLON_TO_RADIANS, longitude * LATLON_TO_RADIANS]])
        dist, ind = self._nn.kneighbors(q, n_neighbors=k)
        ind = ind[0]

        out = []
        for stop_idx in range(self._n_stops):
            vals = self._y[ind, stop_idx]
            valid = ~np.isnan(vals)
            if not valid.any():
                out.append(np.nan)
                continue
            v = vals[valid]
            if aggregate == "median":
                out.append(float(np.median(v)))
            elif aggregate == "mean":
                out.append(float(np.mean(v)))
            else:
                raise ValueError(f"aggregate must be 'median' or 'mean', got {aggregate}")
        return tuple(out)

    def predict_batch(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        k: Optional[int] = None,
        aggregate: str = "median",
        time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Predict seconds to each stop. Uses median (or mean) of the k neighbors' ETAs.
        time_col is accepted for API compatibility but not used for aggregation.
        """
        if self._nn is None or self._X_rad is None:
            raise RuntimeError("Model not fitted")
        k = k or self.n_neighbors
        k = min(k, self._X_rad.shape[0])

        lat = df[lat_col].values
        lon = df[lon_col].values
        X = np.column_stack([lat * LATLON_TO_RADIANS, lon * LATLON_TO_RADIANS])
        dist, ind = self._nn.kneighbors(X, n_neighbors=k)

        result = {}
        for stop_idx, col in enumerate(self._eta_columns):
            vals = np.take(self._y[:, stop_idx], ind)  # (n_queries, k)
            if aggregate == "median":
                result[col] = np.nanmedian(vals, axis=1)
            else:
                result[col] = np.nanmean(vals, axis=1)
        return pd.DataFrame(result, index=df.index)

    def get_neighbor_indices_batch(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        k: Optional[int] = None,
    ) -> np.ndarray:
        """Return (n_queries, k) indices into the training set for each row in df. For diagnostics."""
        if self._nn is None or self._X_rad is None:
            raise RuntimeError("Model not fitted")
        k = k or self.n_neighbors
        k = min(k, self._X_rad.shape[0])
        lat = df[lat_col].values
        lon = df[lon_col].values
        X = np.column_stack([lat * LATLON_TO_RADIANS, lon * LATLON_TO_RADIANS])
        _, ind = self._nn.kneighbors(X, n_neighbors=k)
        return ind
