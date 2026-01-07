"""
ML Pipeline Utilities

Complete end-to-end pipelines for preprocessing and train/test splitting.
"""
import pandas as pd
from pathlib import Path

# Cache Configuration
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = CACHE_DIR / "raw_vehicle_locations.csv"
PREPROCESSED_CSV = CACHE_DIR / "preprocessed_vehicle_locations.csv"
SEGMENTED_CSV = CACHE_DIR / "segmented_vehicle_locations.csv"
TRAIN_CSV = CACHE_DIR / "train.csv"
TEST_CSV = CACHE_DIR / "test.csv"


def load_pipeline(**kwargs) -> pd.DataFrame:
    """
    Load raw vehicle location data.

    Args:
        **kwargs: Optional arguments:
            - force_reload: If True, fetch from database even if cached file exists.

    Returns:
        DataFrame with raw vehicle location data.
    """
    from ml.data.load import load_vehicle_locations

    force_reload = kwargs.get('force_reload', False)
    print(f"Running Load Pipeline (target: {RAW_CSV})...")

    # Check cache if not forcing reload
    if not force_reload and RAW_CSV.exists():
        print(f"Loading vehicle locations from cache: {RAW_CSV}")
        df = pd.read_csv(RAW_CSV)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        print(f"Loaded {len(df)} location records from CSV")
        return df

    # Fetch from database
    df = load_vehicle_locations()

    # Save to cache
    print(f"Saving fetched data to cache: {RAW_CSV}")
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    return df


def preprocess_pipeline(**kwargs) -> pd.DataFrame:
    """
    Run the preprocessing pipeline on vehicle location data.

    This function applies initial preprocessing steps:
    1. Load vehicle locations (using load_pipeline)
    2. Convert timestamps to epoch seconds (since 2025-01-01)
    3. Add closest route information (route name, distance, closest point)

    Results are cached in 'ml/cache/preprocessed_vehicle_locations.csv'.

    Args:
        **kwargs: Optional arguments:
            - force_repreprocess: If True, recompute preprocessing even if cached file exists.
            - force_reload: Passed to load_pipeline.

    Returns:
        Preprocessed DataFrame.
    """
    from ml.data.preprocess import (
        to_epoch_seconds,
        add_closest_points
    )

    force_repreprocess = kwargs.get('force_repreprocess', False)

    # Check if we can use cached data
    if not force_repreprocess and PREPROCESSED_CSV.exists():
        print(f"Loading preprocessed data from {PREPROCESSED_CSV}")
        df = pd.read_csv(PREPROCESSED_CSV)
        # Convert timestamp back to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        print(f"Loaded {len(df)} preprocessed records from cache")
        return df

    print("Running Preprocessing Pipeline...")
    print("="*60)

    # Step 1: Load data (will use cache if available and not forced)
    df = load_pipeline(**kwargs)

    # Define pipeline functions that work with pipe
    def add_epoch_seconds(df):
        print("Step 2: Converting timestamps to epoch seconds...")
        to_epoch_seconds(df, 'timestamp', 'epoch_seconds')
        return df

    def add_route_info(df):
        print("Step 3: Adding closest route information...")
        add_closest_points(df, 'latitude', 'longitude', {
            'distance': 'dist_to_route',
            'route_name': 'route',
            'closest_point_lat': 'closest_lat',
            'closest_point_lon': 'closest_lon',
            'polyline_index': 'polyline_idx'
        })
        return df

    # Run the pipeline using pipe
    df = (df
          .pipe(add_epoch_seconds)
          .pipe(add_route_info))

    # Save preprocessed data
    print("="*60)
    print(f"Saving preprocessed data to {PREPROCESSED_CSV}")
    df.to_csv(PREPROCESSED_CSV, index=False)
    print(f"Saved {len(df)} preprocessed records")
    return df


def speed_pipeline(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Calculate distance and speed for segmented data.

    Args:
        df: DataFrame with 'segment_id' column.
        **kwargs: Additional arguments (unused but accepted for compatibility).

    Returns:
        DataFrame with added 'distance_km' and 'speed_kmh' columns.
    """
    from ml.data.preprocess import distance_delta, speed

    import numpy as np

    print("Running Speed Pipeline...")
    print("  Calculating distance and speed...")

    # Calculate distance and speed for all points
    df = (df
          .pipe(distance_delta, 'closest_lat', 'closest_lon', 'distance_km')
          .pipe(speed, 'distance_km', 'epoch_seconds', 'speed_kmh'))

    # Mark segment boundaries (where segment_id changes)
    df['_new_segment'] = df['segment_id'] != df['segment_id'].shift(1)

    # Set distance and speed to NaN at segment boundaries
    # This prevents calculating speed/distance between points from different segments
    df.loc[df['_new_segment'], 'distance_km'] = np.nan
    df.loc[df['_new_segment'], 'speed_kmh'] = np.nan

    # Drop working column
    df = df.drop(columns=['_new_segment'])

    print("  ✓ Calculated segment-local speeds")
    print()
    return df


def segment_pipeline(
    max_timedelta: float = 15,
    max_distance: float = 0.005,
    min_segment_length: int = 3,
    **kwargs
) -> pd.DataFrame:
    """
    Segment pipeline: Preprocess -> Segment -> Speed -> Filter.

    Args:
        max_timedelta: Maximum time gap (seconds) for consecutive points.
        max_distance: Maximum distance from route (km) to split segments.
        min_segment_length: Minimum number of points required to keep a segment.
        **kwargs: Optional arguments:
            - force_resegment: Force re-run of segmentation.
            - force_repreprocess: Force re-run of preprocessing (also forces segmentation).
            - force_reload: Passed to load_pipeline.

    Returns:
        Segmented DataFrame with speeds.
    """
    from ml.data.preprocess import segment_by_consecutive, filter_segments_by_length

    force_resegment = kwargs.get('force_resegment', False)
    force_repreprocess = kwargs.get('force_repreprocess', False)

    # Check if we can use cached data
    # Skip cache if either segmentation or preprocessing is forced
    if not (force_resegment or force_repreprocess) and SEGMENTED_CSV.exists():
        print(f"Loading segmented data from {SEGMENTED_CSV}")
        df = pd.read_csv(SEGMENTED_CSV)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        print(f"Loaded {len(df)} segmented records from cache")
        return df

    print("="*70)
    print("SEGMENT PIPELINE: PREPROCESS → SEGMENT → SPEED → FILTER")
    print("="*70)
    print()

    # Step 1: Preprocess data (will use cache if available)
    print("Step 1/4: Ensuring initial preprocessed data...")
    df = preprocess_pipeline(**kwargs)
    print(f"  ✓ Preprocessed {len(df)} location points")
    print()

    # Step 2: Segment into consecutive trips (with distance-based splitting)
    print("Step 2/4: Segmenting into consecutive trips...")
    print(f"  Max time gap: {max_timedelta}s ({max_timedelta/60:.1f} minutes)")
    print(f"  Max distance from route: {max_distance} km (splits segments)")
    df = segment_by_consecutive(
        df,
        max_timedelta=max_timedelta,
        segment_column='segment_id',
        distance_column='dist_to_route',
        max_distance_to_route=max_distance
    )
    num_segments = df['segment_id'].nunique()
    print(f"  ✓ Created {num_segments} segments")
    print()

    # Step 3: Speed Pipeline
    print("Step 3/4: Running Speed Pipeline...")
    df = speed_pipeline(df, **kwargs)

    # Step 4: Filter short segments
    print("Step 4/4: Filtering short segments...")
    print(f"  Minimum segment length: {min_segment_length} points")
    initial_segments = df['segment_id'].nunique()
    df = filter_segments_by_length(df, 'segment_id', min_segment_length)
    final_segments = df['segment_id'].nunique()
    removed_segments = initial_segments - final_segments
    if removed_segments > 0:
        print(f"  ✓ Removed {removed_segments} segments with < {min_segment_length} points")
    print(f"  ✓ Valid segments: {final_segments}")
    print()

    # Save segmented data
    print(f"Saving segmented data to {SEGMENTED_CSV}")
    df.to_csv(SEGMENTED_CSV, index=False)
    print(f"Saved {len(df)} segmented records")
    print()

    return df


def split_pipeline(
    test_ratio: float = 0.2,
    random_seed: int = None,
    **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split pipeline: Load Segmented -> Split.

    Args:
        test_ratio: Fraction of data for test set.
        random_seed: Random seed for reproducible splits.
        **kwargs: Optional arguments:
            - force_resplit: Force re-run of splitting (even if train/test csv exist).
            - force_resegment: Force re-run of segmentation (also forces split).
            - force_repreprocess: Force re-run of preprocessing (also forces segmentation and split).
            - force_reload: Passed to load_pipeline.
            - min_segment_length: Passed to segment_pipeline for filtering.

    Returns:
        Tuple of (train_df, test_df).
    """
    from ml.training.train import segmented_train_test_split

    force_resplit = kwargs.get('force_resplit', False)
    force_resegment = kwargs.get('force_resegment', False)
    force_repreprocess = kwargs.get('force_repreprocess', False)

    # Check if we can just load the split data
    # Skip cache if any upstream step is forced
    if not (force_resplit or force_resegment or force_repreprocess) and TRAIN_CSV.exists() and TEST_CSV.exists():
        print(f"Loading cached train/test split from {CACHE_DIR}")
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)
        if 'timestamp' in train_df.columns:
            train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], format='mixed')
            test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], format='mixed')
        return train_df, test_df

    print("="*70)
    print("SPLIT PIPELINE: LOAD SEGMENTED → SPLIT")
    print("="*70)
    print()

    # Step 1: Get Segmented Data (already filtered by segment_pipeline)
    print("Step 1/2: Ensuring segmented data...")
    df = segment_pipeline(**kwargs)
    print(f"  ✓ Loaded {len(df)} segmented points")
    print(f"  ✓ Segments: {df['segment_id'].nunique()}")
    print()

    if len(df) == 0:
        raise ValueError("No valid segments found. Check segment_pipeline filters.")

    # Step 2: Train/test split
    print("Step 2/2: Splitting into train/test sets...")
    print(f"  Test ratio: {test_ratio:.1%}")
    print(f"  Random seed: {random_seed}")

    train_df, test_df = segmented_train_test_split(
        df,
        test_ratio=test_ratio,
        random_seed=random_seed,
        timestamp_column='timestamp',
        segment_column='segment_id'
    )

    # Save split data
    print(f"Saving split data to {CACHE_DIR}")
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print()
    print("="*70)
    print("SPLIT PIPELINE COMPLETE")
    print("="*70)
    print(f"  Train: {train_df['segment_id'].nunique()} segments, {len(train_df)} points")
    print(f"  Test:  {test_df['segment_id'].nunique()} segments, {len(test_df)} points")
    print(f"  Files saved: {TRAIN_CSV}, {TEST_CSV}")
    print()
    return train_df, test_df


def arima_pipeline(
    p: int = 5,
    d: int = 0,
    q: int = 5,
    value_column: str = 'speed_kmh',
    limit_segments: int = None,
    tune_hyperparams: bool = False,
    **kwargs
):
    """
    End-to-end ARIMA Model pipeline: Load -> Preprocess -> Split -> (Optional Tune) -> Evaluate.

    Fits a separate ARIMA(p, d, q) model to each test segment and evaluates performance.
    Optionally runs hyperparameter tuning before evaluation.

    Args:
        p: AR order (ignored if tune_hyperparams=True)
        d: Differencing order (ignored if tune_hyperparams=True)
        q: MA order (ignored if tune_hyperparams=True)
        value_column: Column to model
        limit_segments: Optional limit on number of segments for testing (for speed)
        tune_hyperparams: If True, run hyperparameter search before evaluation
        **kwargs: Arguments passed to split_pipeline (e.g. max_timedelta, force_recompute)

    Returns:
        dict: Results dictionary containing evaluation metrics
    """
    from ml.evaluation.evaluate import evaluate_arima
    from ml.validate.hyperparameters import arima_hyperparameters_segmented
    import pickle

    # 1. Load, Preprocess, and Split Data
    # Note: split_pipeline calls segment_pipeline internally
    train_df, test_df = split_pipeline(**kwargs)

    print("=" * 70)
    print("ARIMA MODEL PIPELINE")
    print("=" * 70)

    # Apply segment limit if requested
    train_df_limited = train_df
    test_df_limited = test_df
    if limit_segments:
        print(f"Limiting to {limit_segments} segments for training/testing...")
        train_segments = train_df['segment_id'].unique()[:limit_segments]
        train_df_limited = train_df[train_df['segment_id'].isin(train_segments)].copy()
        test_segments = test_df['segment_id'].unique()[:limit_segments]
        test_df_limited = test_df[test_df['segment_id'].isin(test_segments)].copy()

    # 2. Hyperparameter Tuning (optional)
    best_params_path = CACHE_DIR / "arima_best_params.pkl"

    if tune_hyperparams:
        print("\n" + "=" * 70)
        print("HYPERPARAMETER TUNING")
        print("=" * 70)

        try:
            tuning_result = arima_hyperparameters_segmented(
                train_df_limited,
                val_df=None,  # Use AIC for tuning
                value_column=value_column,
                p_range=(0, 5),
                d_range=(0, 2),
                q_range=(0, 5),
                metric='aic',
                n_segments_sample=min(10, train_df_limited['segment_id'].nunique()),
                max_combinations=100
            )

            # Update parameters with tuned values
            p = tuning_result['best_p']
            d = tuning_result['best_d']
            q = tuning_result['best_q']

            # Save best parameters
            print(f"\nSaving best parameters to {best_params_path}")
            with open(best_params_path, 'wb') as f:
                pickle.dump({
                    'p': p,
                    'd': d,
                    'q': q,
                    'best_score': tuning_result['best_score']
                }, f)

        except Exception as e:
            print(f"Hyperparameter tuning failed: {e}")
            print("Falling back to default parameters")
            import traceback
            traceback.print_exc()
    else:
        # Load cached parameters if available
        if best_params_path.exists():
            print(f"\nLoading cached best parameters from {best_params_path}")
            with open(best_params_path, 'rb') as f:
                cached_params = pickle.load(f)
                print(f"  Cached: ARIMA({cached_params['p']},{cached_params['d']},{cached_params['q']})")
                print(f"  (Using command-line parameters for this run)")

    # 3. Evaluate ARIMA on test segments
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"Evaluating ARIMA({p},{d},{q}) on test segments...")

    try:
        results = evaluate_arima(
            test_df_limited,
            p=p,
            d=d,
            q=q,
            value_column=value_column
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    return results


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run Shubble ML Pipelines")
    subparsers = parser.add_subparsers(dest="pipeline", help="Pipeline to run")

    # Load Pipeline Arguments
    load_parser = subparsers.add_parser("load", help="Load raw vehicle location data")
    load_parser.add_argument("--force-reload", action="store_true", help="Force fetching data from DB")

    # Preprocess Pipeline Arguments
    preprocess_parser = subparsers.add_parser("preprocess", help="Run preprocessing only")
    preprocess_parser.add_argument("--force-repreprocess", action="store_true", help="Force recomputation of preprocessing")
    preprocess_parser.add_argument("--force-reload", action="store_true", help="Force fetching data from DB")

    # Segment Pipeline Arguments
    segment_parser = subparsers.add_parser("segment", help="Run preprocessing, segmentation, speed calc")
    segment_parser.add_argument("--force-resegment", action="store_true", help="Force recomputation of segmentation")
    segment_parser.add_argument("--force-repreprocess", action="store_true", help="Force recomputation of preprocessing")
    segment_parser.add_argument("--force-reload", action="store_true", help="Force fetching data from DB")
    segment_parser.add_argument("--max-timedelta", type=float, default=15, help="Max time gap for segmentation (seconds)")
    segment_parser.add_argument("--max-distance", type=float, default=0.005, help="Max distance from route to split segment (km)")
    segment_parser.add_argument("--min-segment-length", type=int, default=3, help="Min points per segment")

    # Split Pipeline Arguments
    split_parser = subparsers.add_parser("split", help="Run full chain up to split")
    split_parser.add_argument("--force-resplit", action="store_true", help="Force recomputation of train/test split")
    split_parser.add_argument("--force-resegment", action="store_true", help="Force recomputation of segmentation")
    split_parser.add_argument("--force-repreprocess", action="store_true", help="Force recomputation of preprocessing")
    split_parser.add_argument("--force-reload", action="store_true", help="Force fetching data from DB")
    split_parser.add_argument("--max-timedelta", type=float, default=15, help="Max time gap for segmentation (seconds)")
    split_parser.add_argument("--max-distance", type=float, default=0.005, help="Max distance from route to split segment (km)")
    split_parser.add_argument("--min-segment-length", type=int, default=3, help="Min points per segment")
    split_parser.add_argument("--test-ratio", type=float, default=0.2, help="Test set fraction")
    split_parser.add_argument("--random-seed", type=int, default=42, help="Random seed")

    # ARIMA Pipeline Arguments
    arima_parser = subparsers.add_parser("arima", help="Run ARIMA Model pipeline (segment-wise evaluation)")
    arima_parser.add_argument("--force-resplit", action="store_true", help="Force recomputation of train/test split")
    arima_parser.add_argument("--force-resegment", action="store_true", help="Force recomputation of segmentation")
    arima_parser.add_argument("--force-repreprocess", action="store_true", help="Force recomputation of preprocessing")
    arima_parser.add_argument("--force-reload", action="store_true", help="Force fetching data from DB")
    arima_parser.add_argument("--max-timedelta", type=float, default=15, help="Max time gap for segmentation (seconds)")
    arima_parser.add_argument("--max-distance", type=float, default=0.005, help="Max distance from route to split segment (km)")
    arima_parser.add_argument("--min-segment-length", type=int, default=3, help="Min points per segment")
    arima_parser.add_argument("--test-ratio", type=float, default=0.2, help="Test set fraction")
    arima_parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    arima_parser.add_argument("--tune-hyperparams", action="store_true", help="Run hyperparameter search before evaluation")
    arima_parser.add_argument("--p", type=int, default=3, help="AR order (ignored if --tune-hyperparams)")
    arima_parser.add_argument("--d", type=int, default=0, help="Differencing order (ignored if --tune-hyperparams)")
    arima_parser.add_argument("--q", type=int, default=2, help="MA order (ignored if --tune-hyperparams)")
    arima_parser.add_argument("--value-column", type=str, default="speed_kmh", help="Column to model")
    arima_parser.add_argument("--limit-segments", type=int, default=None, help="Limit number of segments (for testing)")

    # Parse arguments
    args = parser.parse_args()
    if args.pipeline is None:
        parser.print_help()
        sys.exit(1)

    # Convert args to dictionary to pass as kwargs
    kwargs = {k: v for k, v in vars(args).items() if k != 'pipeline'}

    if args.pipeline == "load":
        print("Running Load Pipeline...")
        df = load_pipeline(**kwargs)
        print(f"\nLoaded {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSample data:")
        print(df.head())
    elif args.pipeline == "preprocess":
        print("Running Preprocessing Pipeline...")
        df = preprocess_pipeline(**kwargs)
        print(f"\nPreprocessed {len(df)} records")
    elif args.pipeline == "segment":
        print("Running Segment Pipeline...")
        df = segment_pipeline(**kwargs)
        print(f"\nSegmented into {df['segment_id'].nunique()} segments")
    elif args.pipeline == "split":
        print("Running Split Pipeline...")
        train_df, test_df = split_pipeline(**kwargs)
        print(f"\nSplit complete:")
        print(f"  Train: {len(train_df)} records")
        print(f"  Test: {len(test_df)} records")
    elif args.pipeline == "arima":
        print("Running ARIMA Model Pipeline...")
        results = arima_pipeline(**kwargs)
        if results:
            print(f"\n{'='*70}")
            print("ARIMA PIPELINE COMPLETE")
            print(f"{'='*70}")
            print(f"Overall MSE: {results['overall_mse']:.4f}")
            print(f"Overall RMSE: {results['overall_rmse']:.4f}")
