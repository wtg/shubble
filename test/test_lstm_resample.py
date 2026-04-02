"""Tests for uniform LSTM input resampling."""
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from ml.lstm_resample import (
    build_lstm_sequences_from_block,
    min_rows_for_lstm_resample,
    normalize_resample_interpolation,
    resample_lstm_features,
)


def test_resample_produces_fixed_grid():
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    rows = []
    t = base - timedelta(seconds=200)
    while t <= base:
        rows.append(
            {
                "timestamp": t,
                "latitude": 40.0 + 0.0001 * len(rows),
                "longitude": -73.0,
                "speed_kmh": 25.0,
            }
        )
        t += timedelta(seconds=7 if len(rows) % 2 == 0 else 22)
    df = pd.DataFrame(rows)
    feat, t_ref = resample_lstm_features(
        df, sequence_length=10, interval_seconds=10.0
    )
    assert feat is not None
    assert feat.shape == (10, 3)
    # t_ref is latest observation in the frame (may be slightly before `base` depending on loop)
    assert pd.Timestamp(t_ref).tz_convert("UTC") <= pd.Timestamp(base).tz_convert("UTC")


def test_resample_single_raw_row():
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    df = pd.DataFrame(
        [
            {
                "timestamp": base,
                "latitude": 42.0,
                "longitude": -73.0,
                "speed_kmh": 20.0,
            }
        ]
    )
    feat, _ = resample_lstm_features(df, sequence_length=10, interval_seconds=10.0)
    assert feat is not None
    assert np.allclose(feat[:, 0], 42.0)
    assert np.allclose(feat[:, 2], 20.0)


def test_min_rows():
    assert min_rows_for_lstm_resample(True, 10) == 1
    assert min_rows_for_lstm_resample(False, 10) == 10


def test_normalize_resample_interpolation():
    assert normalize_resample_interpolation("LINEAR") == "linear"


def test_resample_interpolation_kinds_produce_same_shape():
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    rows = []
    t = base - timedelta(seconds=200)
    while t <= base:
        rows.append(
            {
                "timestamp": t,
                "latitude": 40.0 + 0.0001 * len(rows) ** 1.2,
                "longitude": -73.0,
                "speed_kmh": 20.0 + len(rows) * 0.1,
            }
        )
        t += timedelta(seconds=6 if len(rows) % 2 == 0 else 14)
    df = pd.DataFrame(rows)
    lin, _ = resample_lstm_features(
        df, sequence_length=10, interval_seconds=10.0, interpolation="linear"
    )
    cub, _ = resample_lstm_features(
        df, sequence_length=10, interval_seconds=10.0, interpolation="cubic"
    )
    assert lin is not None and cub is not None
    assert lin.shape == cub.shape == (10, 3)
    assert not np.allclose(lin, cub)


def test_build_sequences_resample_vs_rows():
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(20):
        rows.append(
            {
                "timestamp": base + timedelta(seconds=i * 6),
                "latitude": 40.0 + i * 0.001,
                "longitude": -73.0,
                "speed_kmh": 20.0,
                "eta_seconds": float(i),
            }
        )
    df = pd.DataFrame(rows)
    xr, yr = build_lstm_sequences_from_block(
        df,
        input_columns=["latitude", "longitude", "speed_kmh"],
        output_columns=["eta_seconds"],
        sequence_length=10,
        resample_enabled=True,
        resample_interval_seconds=10.0,
        timestamp_column="timestamp",
    )
    xn, yn = build_lstm_sequences_from_block(
        df,
        input_columns=["latitude", "longitude", "speed_kmh"],
        output_columns=["eta_seconds"],
        sequence_length=10,
        resample_enabled=False,
        timestamp_column="timestamp",
    )
    assert len(xr) == len(yr) and len(xn) == len(yn)
    assert xr[0].shape == (10, 3)
    assert xn[0].shape == (10, 3)
