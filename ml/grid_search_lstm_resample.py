"""
Grid search over LSTM time resampling: interval, sequence length, and interpolation kind.

Compares each resampling configuration against a no-resample baseline (consecutive rows)
for the same sequence length. Uses the same train/test split as the LSTM pipeline.

Typical workflow:
  1. Generate per-polyline train.csv / test.csv once, e.g.:
     ``uv run python -m ml.pipelines lstm --train`` (or use ``--prepare-data`` below)
  2. Run the search (reads cache by default):
     ``uv run python -m ml.grid_search_lstm_resample``

Requires the ``ml`` dependency group (torch, scipy, scikit-learn, ...).
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from ml.cache import LSTM_CACHE_DIR
from ml.evaluation.evaluate import evaluate_lstm
from ml.pipelines import eta_pipeline, split_by_polyline_pipeline, stops_pipeline
from ml.training.train import segmented_train_test_split, train_lstm

logger = logging.getLogger("ml.grid_search_lstm_resample")

# Default search space (edit here or extend CLI later)
DEFAULT_INTERVALS_S = (5.0, 8.0, 15.0)
DEFAULT_SEQUENCE_LENGTHS = (10, 15, 20)
DEFAULT_INTERPOLATIONS = ("linear", "quadratic", "cubic")


@dataclass(frozen=True)
class SearchConfig:
    label: str
    resample_enabled: bool
    sequence_length: int
    interval_seconds: float | None
    interpolation: str


def _parse_polyline_dir_name(dir_name: str) -> tuple[str, int]:
    parts = dir_name.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected polyline directory name: {dir_name!r}")
    route_safe, idx_s = parts
    return route_safe.replace("_", " "), int(idx_s)


def iter_cached_polyline_splits() -> Iterator[tuple[str, int, pd.DataFrame, pd.DataFrame]]:
    """Yield (route_name, polyline_idx, train_df, test_df) for each cached split."""
    if not LSTM_CACHE_DIR.is_dir():
        return
    for sub in sorted(LSTM_CACHE_DIR.iterdir()):
        if not sub.is_dir():
            continue
        tr_path = sub / "train.csv"
        te_path = sub / "test.csv"
        if not tr_path.exists() or not te_path.exists():
            continue
        try:
            route_name, polyline_idx = _parse_polyline_dir_name(sub.name)
        except ValueError:
            continue
        train_df = pd.read_csv(tr_path)
        test_df = pd.read_csv(te_path)
        yield route_name, polyline_idx, train_df, test_df


def _clean_df(
    df: pd.DataFrame,
    input_columns: list[str],
    output_columns: list[str],
) -> pd.DataFrame:
    if "speed_kmh" in df.columns:
        df = df.copy()
        df["speed_kmh"] = df["speed_kmh"].fillna(0)
    req = [c for c in input_columns + output_columns if c in df.columns]
    return df.dropna(subset=req)


def prepare_splits_from_pipeline(
    *,
    input_columns: list[str],
    output_columns: list[str],
    test_ratio: float,
    random_seed: int,
    limit_polylines: int | None,
    window_size: int = 5,
    **kwargs: Any,
) -> dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame]]:
    """Build train/test frames the same way as ``lstm_pipeline`` (without training)."""
    stops_df = stops_pipeline(window_size=window_size, **kwargs)
    eta_df = eta_pipeline(df=stops_df, **kwargs)
    polyline_dfs = split_by_polyline_pipeline(df=eta_df)
    keys = sorted(polyline_dfs.keys())
    if limit_polylines is not None:
        keys = keys[:limit_polylines]

    out: dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for key in keys:
        df = polyline_dfs[key]
        if df["segment_id"].nunique() < 2:
            continue
        train_df, test_df = segmented_train_test_split(
            df,
            test_ratio=test_ratio,
            random_seed=random_seed,
            timestamp_column="timestamp",
            segment_column="segment_id",
        )
        train_df = _clean_df(train_df, input_columns, output_columns)
        test_df = _clean_df(test_df, input_columns, output_columns)
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        out[key] = (train_df, test_df)
    return out


def build_search_configs(
    intervals: tuple[float, ...],
    sequence_lengths: tuple[int, ...],
    interpolations: tuple[str, ...],
) -> list[SearchConfig]:
    configs: list[SearchConfig] = []
    for seq in sequence_lengths:
        configs.append(
            SearchConfig(
                label=f"baseline_no_resample_seq{seq}",
                resample_enabled=False,
                sequence_length=seq,
                interval_seconds=None,
                interpolation="linear",
            )
        )
    for seq in sequence_lengths:
        for dt in intervals:
            for interp in interpolations:
                configs.append(
                    SearchConfig(
                        label=f"resample_{interp}_dt{dt:g}_seq{seq}",
                        resample_enabled=True,
                        sequence_length=seq,
                        interval_seconds=dt,
                        interpolation=interp,
                    )
                )
    return configs


def run_one(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: SearchConfig,
    *,
    input_columns: list[str],
    output_columns: list[str],
    hidden_size: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    verbose_train: bool,
) -> dict[str, Any]:
    model = train_lstm(
        train_df,
        input_columns=input_columns,
        output_columns=output_columns,
        sequence_length=cfg.sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        segment_column="segment_id",
        verbose=verbose_train,
        resample_enabled=cfg.resample_enabled,
        resample_interval_seconds=cfg.interval_seconds or 10.0,
        resample_interpolation=cfg.interpolation,
        timestamp_column="timestamp",
    )
    return evaluate_lstm(
        model,
        test_df,
        input_columns=input_columns,
        output_columns=output_columns,
        sequence_length=cfg.sequence_length,
        segment_column="segment_id",
        resample_enabled=cfg.resample_enabled,
        resample_interval_seconds=cfg.interval_seconds or 10.0,
        resample_interpolation=cfg.interpolation,
        timestamp_column="timestamp",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grid search LSTM resampling vs no-resample baseline (per polyline train/test CSV)."
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Run stops/eta/split pipelines instead of reading CSV cache (default: read cache)",
    )
    parser.add_argument("--output", type=Path, default=LSTM_CACHE_DIR / "lstm_resample_grid_results.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=50)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--limit-polylines", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=5, help="NaN route cleaning window (with --prepare-data)")
    parser.add_argument("--limit-configs", type=int, default=None, help="Debug: only first N configs")
    parser.add_argument("--verbose-train", action="store_true")
    parser.add_argument(
        "--intervals",
        type=str,
        default=",".join(str(x) for x in DEFAULT_INTERVALS_S),
        help="Comma-separated seconds, e.g. 5,8,15",
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEQUENCE_LENGTHS),
        help="Comma-separated sequence lengths, e.g. 10,15,20",
    )
    parser.add_argument(
        "--interpolations",
        type=str,
        default=",".join(DEFAULT_INTERPOLATIONS),
        help="Comma-separated: linear,quadratic,cubic",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    intervals = tuple(float(x.strip()) for x in args.intervals.split(",") if x.strip())
    seq_lens = tuple(int(x.strip()) for x in args.seq_lens.split(",") if x.strip())
    interps = tuple(x.strip() for x in args.interpolations.split(",") if x.strip())

    input_columns = ["latitude", "longitude", "speed_kmh", "dist_to_end"]
    output_columns = ["eta_seconds"]

    configs = build_search_configs(intervals, seq_lens, interps)
    if args.limit_configs is not None:
        configs = configs[: args.limit_configs]

    if args.prepare_data:
        splits = prepare_splits_from_pipeline(
            input_columns=input_columns,
            output_columns=output_columns,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed,
            limit_polylines=args.limit_polylines,
            window_size=args.window_size,
        )
        if not splits:
            logger.error("No polyline splits after --prepare-data.")
            return 1
        polyline_items = [(k[0], k[1], tr, te) for k, (tr, te) in splits.items()]
    else:
        polyline_items = list(iter_cached_polyline_splits())
        if args.limit_polylines is not None:
            polyline_items = polyline_items[: args.limit_polylines]
        if not polyline_items:
            logger.error(
                "No cached train.csv/test.csv under %s. Run lstm pipeline once or use --prepare-data.",
                LSTM_CACHE_DIR,
            )
            return 1

    rows: list[dict[str, Any]] = []
    total = len(polyline_items) * len(configs)
    done = 0

    for route_name, polyline_idx, train_df, test_df in polyline_items:
        train_df = _clean_df(train_df, input_columns, output_columns)
        test_df = _clean_df(test_df, input_columns, output_columns)
        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning("Skip %s %s: empty after clean", route_name, polyline_idx)
            continue

        for cfg in configs:
            done += 1
            try:
                results = run_one(
                    train_df,
                    test_df,
                    cfg,
                    input_columns=input_columns,
                    output_columns=output_columns,
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    verbose_train=args.verbose_train,
                )
                row = {
                    "route": route_name,
                    "polyline_idx": polyline_idx,
                    "config": cfg.label,
                    "resample_enabled": cfg.resample_enabled,
                    "sequence_length": cfg.sequence_length,
                    "interval_seconds": cfg.interval_seconds if cfg.resample_enabled else "",
                    "interpolation": cfg.interpolation if cfg.resample_enabled else "",
                    "rmse": results["rmse"],
                    "mae": results["mae"],
                    "mse": results["mse"],
                    "num_predictions": results["num_predictions"],
                }
                rows.append(row)
                logger.info(
                    "[%s/%s] %s | %s | RMSE=%.4f",
                    done,
                    total,
                    f"{route_name} #{polyline_idx}",
                    cfg.label,
                    results["rmse"],
                )
            except Exception as e:
                logger.exception("Failed %s %s config=%s: %s", route_name, polyline_idx, cfg.label, e)
                rows.append(
                    {
                        "route": route_name,
                        "polyline_idx": polyline_idx,
                        "config": cfg.label,
                        "resample_enabled": cfg.resample_enabled,
                        "sequence_length": cfg.sequence_length,
                        "interval_seconds": cfg.interval_seconds if cfg.resample_enabled else "",
                        "interpolation": cfg.interpolation if cfg.resample_enabled else "",
                        "rmse": "",
                        "mae": "",
                        "mse": "",
                        "num_predictions": "",
                        "error": str(e),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "route",
        "polyline_idx",
        "config",
        "resample_enabled",
        "sequence_length",
        "interval_seconds",
        "interpolation",
        "rmse",
        "mae",
        "mse",
        "num_predictions",
        "error",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    logger.info("Wrote %s rows to %s", len(rows), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
