"""
Uniform time-grid resampling for LSTM inference and training on irregular GPS samples.

Raw vehicle rows arrive at arbitrary intervals; the LSTM expects a fixed-length
sequence. We interpolate input features at
t_ref - N*Δ, …, t_ref - Δ (oldest → newest in time) so spacing is consistent
regardless of broadcast cadence.

Used by production inference (``predict_eta`` / ``generate_eta``) and by
``train_lstm`` / ``evaluate_lstm`` when resampling is enabled (default).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd

ResampleInterpolation = Literal["linear", "quadratic", "cubic"]

VALID_RESAMPLE_INTERPOLATIONS: frozenset[str] = frozenset({"linear", "quadratic", "cubic"})


def normalize_resample_interpolation(kind: str) -> str:
    k = kind.strip().lower()
    if k not in VALID_RESAMPLE_INTERPOLATIONS:
        raise ValueError(
            f"interpolation must be one of {sorted(VALID_RESAMPLE_INTERPOLATIONS)}, got {kind!r}"
        )
    return k


def _interp_column(
    tx: np.ndarray,
    yy: np.ndarray,
    t_target: np.ndarray,
    kind: str,
) -> np.ndarray:
    """Interpolate along time ``tx`` for each target time ``t_target`` (1D per feature)."""
    if len(tx) == 1:
        return np.full(len(t_target), float(yy[0]), dtype=np.float64)

    order = np.argsort(tx)
    tx = tx[order].astype(np.float64)
    yy = yy[order].astype(np.float64)

    if kind == "linear":
        return np.interp(t_target, tx, yy, left=yy[0], right=yy[-1])

    min_pts = {"quadratic": 3, "cubic": 4}
    if len(tx) < min_pts.get(kind, 3):
        return np.interp(t_target, tx, yy, left=yy[0], right=yy[-1])

    from scipy.interpolate import interp1d

    f = interp1d(
        tx,
        yy,
        kind=kind,
        bounds_error=False,
        fill_value=(float(yy[0]), float(yy[-1])),
    )
    out = f(t_target)
    return np.asarray(out, dtype=np.float64).ravel()


def resample_lstm_features(
    vehicle_df: pd.DataFrame,
    *,
    sequence_length: int = 10,
    interval_seconds: float = 10.0,
    input_columns: list[str] | None = None,
    timestamp_column: str = "timestamp",
    interpolation: ResampleInterpolation | str = "linear",
) -> tuple[np.ndarray | None, datetime]:
    """
    Build a (sequence_length, n_features) feature matrix by time interpolation
    at evenly spaced times before the latest observation.

    Target sample times (chronological order, oldest row first):
      t_ref - sequence_length * interval_seconds,
      t_ref - (sequence_length - 1) * interval_seconds,
      …,
      t_ref - interval_seconds

    Args:
        vehicle_df: Rows for one vehicle, must include 'timestamp' and feature columns.
        sequence_length: LSTM sequence length (default 10).
        interval_seconds: Spacing between samples (default 10 → 100s total lookback).
        input_columns: Feature columns (default lat, lon, speed_kmh).
        interpolation: ``linear`` (``numpy.interp``), ``quadratic`` or ``cubic`` (``scipy.interpolate.interp1d``).

    Returns:
        (features, t_ref) where features has shape (sequence_length, len(input_columns)),
        or (None, t_ref) if there is no usable timestamp.
    """
    kind = normalize_resample_interpolation(interpolation)
    if input_columns is None:
        input_columns = ["latitude", "longitude", "speed_kmh"]

    if vehicle_df is None or len(vehicle_df) == 0:
        return None, datetime.now(timezone.utc)

    if timestamp_column not in vehicle_df.columns:
        return None, datetime.now(timezone.utc)

    df = vehicle_df.sort_values(timestamp_column).copy()
    for col in input_columns:
        if col not in df.columns:
            df[col] = 0.0

    # One row per timestamp (mean features if duplicates)
    ts = pd.to_datetime(df[timestamp_column], utc=True)
    df = df.assign(_ts=ts)
    agg = df.groupby("_ts")[input_columns].mean().reset_index()

    if agg.empty:
        return None, datetime.now(timezone.utc)

    anchor = pd.Timestamp(agg["_ts"].iloc[-1])
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize("UTC")
    else:
        anchor = anchor.tz_convert("UTC")
    t_ref_dt = anchor.to_pydatetime()
    if t_ref_dt.tzinfo is None:
        t_ref_dt = t_ref_dt.replace(tzinfo=timezone.utc)

    # Oldest → newest: (t_ref - 100s) … (t_ref - 10s) for seq=10, interval=10
    offsets_sec = np.arange(sequence_length, 0, -1, dtype=np.float64) * float(interval_seconds)
    target_times = anchor - pd.to_timedelta(offsets_sec, unit="s")
    t_target_s = target_times.astype(np.int64).astype(np.float64) / 1.0e9

    t_obs = pd.to_datetime(agg["_ts"], utc=True)
    t_obs_ns = t_obs.values.astype("datetime64[ns]").astype(np.int64)
    t_obs_s = t_obs_ns.astype(np.float64) / 1.0e9

    rows: list[np.ndarray] = []
    for col in input_columns:
        y_obs = agg[col].astype(np.float64).values
        if len(t_obs_s) == 1:
            y_tgt = np.full(sequence_length, float(y_obs[0]), dtype=np.float64)
        else:
            order = np.argsort(t_obs_s)
            tx = t_obs_s[order]
            yy = y_obs[order]
            y_tgt = np.interp(t_target_s, tx, yy, left=yy[0], right=yy[-1])
        rows.append(y_tgt.astype(np.float32))

    features = np.stack(rows, axis=1)
    return features, t_ref_dt


def min_rows_for_lstm_resample(resample_enabled: bool, sequence_length: int) -> int:
    """Minimum raw rows required before building an LSTM sequence."""
    if resample_enabled:
        return 1
    return sequence_length


def build_lstm_sequences_from_block(
    block_df: pd.DataFrame,
    *,
    input_columns: list[str],
    output_columns: list[str],
    sequence_length: int,
    resample_enabled: bool,
    resample_interval_seconds: float = 10.0,
    timestamp_column: str = "timestamp",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Build (X, y) training/eval sequences from one contiguous block (e.g. one segment_id).

    When ``resample_enabled`` is True, matches inference: ``t_ref`` is the timestamp of
    the last row of each input window; features are linearly interpolated on a uniform
    grid. When False, uses the legacy sliding window over consecutive rows.
    """
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    if block_df is None or len(block_df) == 0:
        return X_list, y_list

    if resample_enabled and timestamp_column not in block_df.columns:
        raise ValueError(
            f"resample_enabled requires column {timestamp_column!r} in training data"
        )

    block_df = block_df.sort_values(timestamp_column)
    if len(block_df) <= sequence_length:
        return X_list, y_list

    if resample_enabled:
        for i in range(len(block_df) - sequence_length):
            end_idx = i + sequence_length - 1
            window_df = block_df.iloc[: end_idx + 1].copy()
            feat, _ = resample_lstm_features(
                window_df,
                sequence_length=sequence_length,
                interval_seconds=resample_interval_seconds,
                input_columns=input_columns,
                timestamp_column=timestamp_column,
            )
            if feat is None:
                continue
            X_list.append(feat.astype(np.float32))
            y_row = block_df.iloc[i + sequence_length]
            y_list.append(y_row[output_columns].values.astype(np.float64))
    else:
        data_in = block_df[input_columns].values
        data_out = block_df[output_columns].values
        for i in range(len(block_df) - sequence_length):
            X_list.append(data_in[i : i + sequence_length].astype(np.float32))
            y_list.append(data_out[i + sequence_length])

    return X_list, y_list
