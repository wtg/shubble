"""
ML Pipeline Utilities

Complete end-to-end pipelines for preprocessing and train/test splitting.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# PIPELINE HIERARCHY
# ============================================================================

# Define the pipeline execution hierarchy
# If a stage is triggered, all subsequent stages are also triggered
PIPELINE_HIERARCHY = [
    'load',
    'preprocess',
    'segment',
    'stops',
    'eta',
    'split',
    'train',
    'fit'
]


def apply_pipeline_hierarchy(kwargs: dict) -> dict:
    """
    Apply pipeline hierarchy to kwargs.

    If a pipeline stage flag is set to True, all subsequent stages in the
    hierarchy are also set to True.

    Pipeline hierarchy order: load -> preprocess -> segment -> stops -> eta -> split -> train -> fit

    Args:
        kwargs: Dictionary of keyword arguments

    Returns:
        Modified kwargs with hierarchy applied

    Examples:
        >>> apply_pipeline_hierarchy({'segment': True})
        {'segment': True, 'stops': True, 'eta': True, 'split': True, 'train': True, 'fit': True}

        >>> apply_pipeline_hierarchy({'load': True})
        {'load': True, 'preprocess': True, 'segment': True, 'stops': True, 'eta': True, 'split': True, 'train': True, 'fit': True}
    """
    kwargs = kwargs.copy()

    # Find the earliest stage that is set to True
    trigger_index = None
    for i, stage in enumerate(PIPELINE_HIERARCHY):
        if kwargs.get(stage, False):
            if trigger_index is None:
                trigger_index = i
            break

    # If a stage was triggered, set all subsequent stages to True
    if trigger_index is not None:
        for stage in PIPELINE_HIERARCHY[trigger_index:]:
            kwargs[stage] = True

    return kwargs


# ============================================================================
# CACHE CONFIGURATION & HELPERS
# ============================================================================

from ml.cache import (
    get_cache_path, load_cached_csv, save_csv,
    ARIMA_CACHE_DIR, LSTM_CACHE_DIR,
    RAW_CSV, PREPROCESSED_CSV,
    get_polyline_dir
)


# ============================================================================
# BASIC PIPELINES
# ============================================================================

def load_pipeline(**kwargs) -> pd.DataFrame:
    """
    Load raw vehicle location data.

    Args:
        load: Re-load from database

    Note:
        Pipeline hierarchy is applied externally via apply_pipeline_hierarchy().
        This function only checks its own 'load' flag.

    Returns:
        DataFrame with raw vehicle location data
    """
    from ml.data.load import load_vehicle_locations

    load = kwargs.get('load', False)

    # Try to load from cache
    if not load:
        cached_df = load_cached_csv(RAW_CSV, "vehicle locations")
        if cached_df is not None:
            return cached_df

    # Fetch from database
    logger.info("Fetching vehicle locations from database...")
    df = load_vehicle_locations()
    save_csv(df, RAW_CSV, "fetched data")
    return df


def preprocess_pipeline(df: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
    """
    Run the preprocessing pipeline on vehicle location data.

    Steps:
    1. Load vehicle locations (using load_pipeline)
    2. Convert timestamps to epoch seconds
    3. Add closest route information

    Args:
        df: Optional DataFrame to process. If None, loads from load_pipeline.
        preprocess: Re-compute preprocessing
        load: Re-load from database
        additive: If True, only compute closest points for rows with NaN values (default: False)

    Note:
        Pipeline hierarchy is applied externally via apply_pipeline_hierarchy().
        This function only checks its own 'preprocess' flag.

    Returns:
        Preprocessed DataFrame
    """
    from ml.data.preprocess import to_epoch_seconds, add_closest_points

    preprocess = kwargs.get('preprocess', False)
    cache = kwargs.get('cache', True)
    additive = kwargs.get('additive', False)

    # Try to load from cache (only if no input df provided and caching is enabled)
    if not preprocess and cache:
        cached_df = load_cached_csv(PREPROCESSED_CSV, "preprocessed data")
        if cached_df is not None:
            return cached_df

    logger.info("="*70)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("="*70)

    # Load data if not provided
    if df is None:
        df = load_pipeline(**kwargs)

    # Add epoch seconds
    logger.info("Step 1/2: Converting timestamps to epoch seconds...")
    to_epoch_seconds(df, 'timestamp', 'epoch_seconds')

    # Add route information
    logger.info("Step 2/2: Adding closest route information...")
    add_closest_points(df, 'latitude', 'longitude', {
        'distance': 'dist_to_route',
        'route_name': 'route',
        'closest_point_lat': 'closest_lat',
        'closest_point_lon': 'closest_lon',
        'polyline_index': 'polyline_idx',
        'segment_index': 'segment_idx'
    }, additive=additive)

    # Save to cache if caching is enabled
    if cache:
        save_csv(df, PREPROCESSED_CSV, "preprocessed data")
    return df


def speed_pipeline(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Calculate distance and speed for segmented data.

    Args:
        df: DataFrame with 'segment_id' column

    Returns:
        DataFrame with added 'distance_km' and 'speed_kmh' columns
    """
    from ml.data.preprocess import distance_delta, speed

    logger.info("Calculating distance and speed...")

    # Calculate distance and speed
    df = (df
          .pipe(distance_delta, 'closest_lat', 'closest_lon', 'distance_km')
          .pipe(speed, 'distance_km', 'epoch_seconds', 'speed_kmh'))

    # Mark segment boundaries
    df['_new_segment'] = df['segment_id'] != df['segment_id'].shift(1)

    # Set distance and speed to NaN at segment boundaries
    df.loc[df['_new_segment'], 'distance_km'] = np.nan
    df.loc[df['_new_segment'], 'speed_kmh'] = np.nan

    # Clean up
    df = df.drop(columns=['_new_segment'])

    logger.info("  ✓ Calculated segment-local speeds")
    return df


def segment_pipeline(df: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
    """
    Segment pipeline: Preprocess -> Segment -> Clean Routes -> Speed -> Filter.

    Args (via kwargs):
        df: Optional DataFrame (preprocessed). If None, runs preprocess_pipeline.
        max_timedelta: Maximum time gap (seconds) for consecutive points (default: 15)
        max_distance: Maximum distance from route (km) to split segments (default: 0.020)
        min_segment_length: Minimum points required to keep a segment (default: 3)
        window_size: Window size for cleaning NaN route values (default: 5)
        require_majority_valid: If True, allows filling NaN endpoints without strict majority (default: False)
        segment: Re-run segmentation (default: False)
        preprocess: Re-run preprocessing (default: False)
        load: Re-load from database (default: False)

    Note:
        Pipeline hierarchy is applied externally via apply_pipeline_hierarchy().
        This function only checks its own 'segment' flag.

    Returns:
        Segmented DataFrame with speeds
    """
    from ml.data.preprocess import segment_by_consecutive, filter_segments_by_length, clean_closest_route, add_closest_points_educated

    # Extract parameters with defaults
    max_timedelta = kwargs.get('max_timedelta', 15)
    max_distance = kwargs.get('max_distance', 0.020)
    min_segment_length = kwargs.get('min_segment_length', 3)
    window_size = kwargs.get('window_size', 5)
    require_majority_valid = kwargs.get('require_majority_valid', False)
    segment = kwargs.get('segment', False)
    cache = kwargs.get('cache', True)

    # Get parameterized cache path
    cache_path = get_cache_path(
        'locations_segmented',
        max_timedelta=max_timedelta,
        max_distance=max_distance,
        min_segment_length=min_segment_length,
        window_size=window_size
    )

    # Try to load from cache (only if caching is enabled)
    if not segment and cache:
        cached_df = load_cached_csv(cache_path, "segmented data")
        if cached_df is not None:
            return cached_df

    logger.info("="*70)
    logger.info("SEGMENT PIPELINE")
    logger.info("="*70)

    # Step 1: Preprocess (if df not provided)
    if df is None:
        df = preprocess_pipeline(**kwargs)
    logger.info(f"Step 1/5: Preprocessed {len(df)} location points")

    # Step 2: Segment
    logger.info(f"Step 2/5: Segmenting (max gap: {max_timedelta}s, max distance: {max_distance} km)...")
    df = segment_by_consecutive(
        df,
        max_timedelta=max_timedelta,
        segment_column='segment_id',
        distance_column='dist_to_route',
        max_distance_to_route=max_distance
    )
    logger.info(f"  ✓ Created {df['segment_id'].nunique()} segments")

    # Step 3: Clean NaN route values using segment-aware windowing
    logger.info(f"Step 3/5: Cleaning NaN route values (window size: {window_size})...")
    df = clean_closest_route(df, route_column='route', polyline_idx_column='polyline_idx',
                            segment_column='segment_id', window_size=window_size,
                            require_majority_valid=require_majority_valid)

    # Step 3.5: Refine geometric points for inferred routes
    logger.info(f"Step 3.5/5: Refining geometric points for inferred routes...")
    add_closest_points_educated(
        df,
        lat_column='latitude',
        lon_column='longitude',
        route_column='route',
        polyline_idx_column='polyline_idx',
        output_columns={
            'distance': 'dist_to_route',
            'closest_point_lat': 'closest_lat',
            'closest_point_lon': 'closest_lon',
            'segment_index': 'segment_idx'
        }
    )

    # Step 4: Add speed
    logger.info("Step 4/5: Adding speed calculations...")
    df = speed_pipeline(df, **kwargs)

    # Step 5: Filter short segments
    logger.info(f"Step 5/5: Filtering segments < {min_segment_length} points...")
    initial_segments = df['segment_id'].nunique()
    df = filter_segments_by_length(df, 'segment_id', min_segment_length)
    final_segments = df['segment_id'].nunique()
    logger.info(f"  ✓ Kept {final_segments}/{initial_segments} segments")

    # Save to cache if caching is enabled
    if cache:
        save_csv(df, cache_path, "segmented data")
    return df


def stops_pipeline(df: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
    """
    Stops Pipeline: Segment -> Add Stops -> Add Polyline Distances.

    This pipeline prepares data for route matching by:
    1. Segmenting vehicle location data
    2. Adding stop information (detecting when vehicles are at stops)
    3. Calculating polyline distances (distance from start, distance to end)

    Args:
        df: Optional DataFrame (segmented). If None, runs segment_pipeline.
        stops: Re-compute stops preprocessing
        segment: Re-run segmentation
        preprocess: Re-run preprocessing
        load: Re-load from database

    Note:
        Pipeline hierarchy is applied externally via apply_pipeline_hierarchy().
        This function only checks its own 'stops' flag.

    Returns:
        DataFrame with added columns: 'stop_name', 'stop_route',
        'dist_from_start', 'dist_to_end', 'polyline_length'
    """
    from ml.data.preprocess import add_stops, add_polyline_distances

    stops = kwargs.get('stops', False)
    cache = kwargs.get('cache', True)

    # Extract segmentation parameters (with defaults matching segment_pipeline call below)
    max_timedelta = kwargs.get('max_timedelta', 30)
    max_distance = kwargs.get('max_distance', 0.005)
    min_segment_length = kwargs.get('min_segment_length', 3)
    window_size = kwargs.get('window_size', 5)

    # Get parameterized cache path
    cache_path = get_cache_path(
        'stops_data',
        max_timedelta=max_timedelta,
        max_distance=max_distance,
        min_segment_length=min_segment_length,
        window_size=window_size
    )

    # Try to load from cache (only if caching is enabled)
    if not stops and cache:
        cached_df = load_cached_csv(cache_path, "stops data")
        if cached_df is not None:
            return cached_df

    logger.info("="*70)
    logger.info("STOPS PIPELINE")
    logger.info("="*70)

    # Build complete kwargs with extracted parameters (ensures consistency with cache path)
    segment_kwargs = {
        **kwargs,
        'max_timedelta': max_timedelta,
        'max_distance': max_distance,
        'min_segment_length': min_segment_length,
        'window_size': window_size
    }

    # Step 1: Get segmented data
    if df is None:
        df = segment_pipeline(**segment_kwargs)

    # Step 2: Add stops
    logger.info("Step 1/2: Adding stop information...")
    add_stops(df, 'latitude', 'longitude', {
        'route_name': 'stop_route',
        'stop_name': 'stop_name'
    })

    # Step 3: Add polyline distances
    logger.info("Step 2/2: Calculating polyline distances...")
    add_polyline_distances(
        df, 'latitude', 'longitude',
        {
            'distance_from_start': 'dist_from_start',
            'distance_to_end': 'dist_to_end',
            'total_length': 'polyline_length'
        },
        distance_column='dist_to_route',
        closest_point_lat_column='closest_lat',
        closest_point_lon_column='closest_lon',
        route_column='route',
        polyline_index_column='polyline_idx',
        segment_index_column='segment_idx'
    )
    logger.info(f"  ✓ Calculated polyline distances for {len(df)} points")

    # Save to cache if caching is enabled
    if cache:
        save_csv(df, cache_path, "stops data")
    return df


def eta_pipeline(df: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
    """
    ETA Pipeline: Stops -> Filter Rows After Stop -> Add ETAs.

    This pipeline prepares data for ETA predictions by:
    1. Loading stops data (if not provided)
    2. Filtering rows after the last stop (also removes segments without stops)
    3. Calculating ETAs to the next stop

    Args:
        df: Optional DataFrame (with stops). If None, runs stops_pipeline.
        eta: Re-compute ETA preprocessing
        stops: Re-compute stops preprocessing
        segment: Re-run segmentation
        preprocess: Re-run preprocessing
        load: Re-load from database

    Note:
        Pipeline hierarchy is applied externally via apply_pipeline_hierarchy().
        This function only checks its own 'eta' flag.

    Returns:
        DataFrame with added column: 'eta_seconds'
    """
    from ml.data.preprocess import filter_rows_after_stop, add_eta

    eta = kwargs.get('eta', False)
    cache = kwargs.get('cache', True)

    # Extract segmentation parameters (with defaults)
    max_timedelta = kwargs.get('max_timedelta', 30)
    max_distance = kwargs.get('max_distance', 0.005)
    min_segment_length = kwargs.get('min_segment_length', 3)
    window_size = kwargs.get('window_size', 5)

    # Get parameterized cache path
    cache_path = get_cache_path(
        'eta_data',
        max_timedelta=max_timedelta,
        max_distance=max_distance,
        min_segment_length=min_segment_length,
        window_size=window_size
    )

    # Try to load from cache (only if caching is enabled)
    if not eta and cache:
        cached_df = load_cached_csv(cache_path, "ETA data")
        if cached_df is not None:
            return cached_df

    logger.info("="*70)
    logger.info("ETA PIPELINE")
    logger.info("="*70)

    # Step 1: Get stops data
    if df is None:
        df = stops_pipeline(**kwargs)

    # Step 2: Filter rows after last stop (also removes segments without stops)
    logger.info("Step 1/2: Filtering rows after last stop...")
    initial_points = len(df)
    initial_segments = df['segment_id'].nunique()
    df = filter_rows_after_stop(df, 'segment_id', 'stop_name')
    removed_points = initial_points - len(df)
    final_segments = df['segment_id'].nunique()
    removed_segments = initial_segments - final_segments
    logger.info(f"  ✓ Removed {removed_points} points ({removed_points/initial_points*100:.1f}%)")
    logger.info(f"  ✓ Removed {removed_segments} segments without stops")

    # Step 3: Add ETAs
    logger.info("Step 2/2: Calculating ETAs...")
    df = add_eta(df, 'stop_name', 'epoch_seconds', 'eta_seconds')
    logger.info(f"  ✓ Calculated ETAs for {len(df)} points")

    # Save to cache if caching is enabled
    if cache:
        save_csv(df, cache_path, "ETA data")
    return df


# ============================================================================
# SPLIT PIPELINE
# ============================================================================

def split_pipeline(
    test_ratio: float = 0.2,
    random_seed: int = None,
    mode: str = 'arima',
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split pipeline: Load Data -> Split into Train/Test.

    Args:
        test_ratio: Fraction of data for test set
        random_seed: Random seed for reproducible splits
        mode: Pipeline mode - 'arima' (speed data) or 'lstm' (ETA data)
        split: Re-run splitting
        stops: Re-compute stops (for lstm mode)
        segment: Re-run segmentation
        preprocess: Re-run preprocessing
        load: Re-load from database

    Note:
        Pipeline hierarchy is applied externally via apply_pipeline_hierarchy().
        This function only checks its own 'split' flag.

    Returns:
        Tuple of (train_df, test_df)
    """
    from ml.training.train import segmented_train_test_split

    # Validate mode
    if mode not in ['arima', 'lstm']:
        raise ValueError(f"mode must be 'arima' or 'lstm', got '{mode}'")

    use_eta_data = (mode == 'lstm')
    split = kwargs.get('split', False)
    segment = kwargs.get('segment', False)
    preprocess = kwargs.get('preprocess', False)

    # Get parameterized cache paths
    cache_dir = ARIMA_CACHE_DIR if mode == 'arima' else LSTM_CACHE_DIR
    train_csv = get_cache_path(
        f'{mode}_train',
        cache_dir=cache_dir,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    test_csv = get_cache_path(
        f'{mode}_test',
        cache_dir=cache_dir,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

    # Try to load from cache
    if not split:
        if train_csv.exists() and test_csv.exists():
            logger.info(f"Loading cached train/test split from {cache_dir}")
            train_df = load_cached_csv(train_csv, "train data")
            test_df = load_cached_csv(test_csv, "test data")
            if train_df is not None and test_df is not None:
                return train_df, test_df

    logger.info("="*70)
    logger.info(f"SPLIT PIPELINE ({mode.upper()} MODE)")
    logger.info("="*70)

    # Load data based on mode
    if use_eta_data:
        logger.info("Loading stops and ETA data...")
        df = stops_pipeline(**kwargs)
        df = eta_pipeline(df=df, **kwargs)
    else:
        logger.info("Loading segmented data...")
        df = segment_pipeline(**kwargs)

    logger.info(f"  ✓ Loaded {len(df)} points, {df['segment_id'].nunique()} segments")

    if len(df) == 0:
        raise ValueError("No valid segments found")

    # Split into train/test
    logger.info(f"Splitting into train/test (ratio: {test_ratio:.1%}, seed: {random_seed})...")
    train_df, test_df = segmented_train_test_split(
        df,
        test_ratio=test_ratio,
        random_seed=random_seed,
        timestamp_column='timestamp',
        segment_column='segment_id'
    )

    # Save split data
    logger.info(f"Saving split data to {cache_dir}")
    save_csv(train_df, train_csv, f"{mode} train data")
    save_csv(test_df, test_csv, f"{mode} test data")

    logger.info(f"  Train: {train_df['segment_id'].nunique()} segments, {len(train_df)} points")
    logger.info(f"  Test:  {test_df['segment_id'].nunique()} segments, {len(test_df)} points")
    return train_df, test_df


def split_by_polyline_pipeline(df: pd.DataFrame = None, **kwargs) -> dict[tuple[str, int], pd.DataFrame]:
    """
    Split preprocessed data by (route, polyline_index) for per-polyline training.

    This pipeline:
    1. Loads preprocessed data if not provided (with route and polyline_idx columns)
    2. Splits by unique (route, polyline_index) combinations
    3. Returns a dictionary mapping polyline keys to DataFrames

    Args:
        df: Optional DataFrame to split. If None, loads from preprocess_pipeline
        **kwargs: Passed to preprocess_pipeline if df is None (e.g., preprocess)

    Returns:
        Dictionary mapping (route_name, polyline_index) tuples to DataFrames

    Example:
        >>> polyline_dfs = split_by_polyline_pipeline()
        >>> for (route, idx), df in polyline_dfs.items():
        ...     logger.info(f"{route} segment {idx}: {len(df)} points")
    """
    from ml.data.preprocess import split_by_route_polyline_index

    logger.info("="*70)
    logger.info("SPLIT BY POLYLINE PIPELINE")
    logger.info("="*70)

    # Load preprocessed data if not provided
    if df is None:
        df = preprocess_pipeline(**kwargs)

    # Check required columns
    if 'route' not in df.columns or 'polyline_idx' not in df.columns:
        raise ValueError(
            "Data must have 'route' and 'polyline_idx' columns. "
            "Make sure preprocessing includes route matching."
        )

    # Split by polyline
    logger.info(f"Splitting {len(df)} points by route and polyline index...")
    polyline_dfs = split_by_route_polyline_index(
        df,
        route_column='route',
        polyline_index_column='polyline_idx'
    )

    logger.info(f"  ✓ Split into {len(polyline_dfs)} unique polylines:")
    for (route, idx), polyline_df in sorted(polyline_dfs.items()):
        logger.info(f"    - {route} segment {idx}: {len(polyline_df)} points")

    return polyline_dfs


# ============================================================================
# LSTM PIPELINE
# ============================================================================

def lstm_pipeline(
    input_columns: list[str] = ['latitude', 'longitude', 'speed_kmh'],
    output_columns: list[str] = ['eta_seconds'],
    sequence_length: int = 10,
    hidden_size: int = 50,
    num_layers: int = 2,
    dropout: float = 0.0,
    epochs: int = 20,
    batch_size: int = 64,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    verbose: bool = True,
    train: bool = True,
    limit_polylines: int = None,
    **kwargs
) -> dict[tuple[str, int], tuple]:
    """
    LSTM Pipeline: Stops Pipeline -> Split by Polyline -> Train per Polyline -> Evaluate.

    This pipeline:
    1. Runs stops pipeline (segment, add stops, add ETAs, add polyline distances, filter)
    2. Splits data by (route, polyline_index) combinations
    3. For each polyline: split train/test, train LSTM, evaluate
    4. Saves all data to ml/cache/lstm/<route>_<polyline_idx>/:
       - data.csv: Full polyline data
       - train.csv: Training split
       - test.csv: Test split
       - model.pth: Trained LSTM model

    Args:
        input_columns: Features to use for prediction
        output_columns: Targets to predict
        sequence_length: Length of input sequences
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: LSTM dropout
        epochs: Training epochs
        batch_size: Batch size
        test_ratio: Train/test split ratio
        random_seed: Random seed
        verbose: Whether to print progress
        train: Retrain models
        limit_polylines: Optional limit on number of polylines to process
        stops: Re-compute stops preprocessing
        segment: Re-run segmentation
        preprocess: Re-run preprocessing
        load: Re-load from database

    Note:
        Pipeline hierarchy is applied externally via apply_pipeline_hierarchy().
        This function only checks its own 'train' flag for model training.

    Returns:
        Dictionary mapping (route, polyline_idx) to (model, results) tuples
    """
    from ml.training.train import train_lstm, segmented_train_test_split
    from ml.evaluation.evaluate import evaluate_lstm
    from ml.models.lstm import LSTMModel

    logger.info("="*70)
    logger.info("LSTM PIPELINE (PER-POLYLINE TRAINING)")
    logger.info("="*70)

    # Step 1: Get stops data (segmented, stops added, polyline distances)
    stops_df = stops_pipeline(**kwargs)

    # Step 2: Add ETAs (filter rows after stop, calculate ETAs)
    eta_df = eta_pipeline(df=stops_df, **kwargs)

    # Step 3: Split by polyline
    polyline_dfs = split_by_polyline_pipeline(df=eta_df)

    # Limit polylines if requested
    if limit_polylines:
        logger.info(f"\nLimiting to {limit_polylines} polylines...")
        polyline_keys = sorted(polyline_dfs.keys())[:limit_polylines]
        polyline_dfs = {k: polyline_dfs[k] for k in polyline_keys}
        logger.info(f"  ✓ Processing {len(polyline_dfs)} polylines")

    # Store results for each polyline
    polyline_models = {}

    # Step 3: Process each polyline
    for polyline_key, df in sorted(polyline_dfs.items()):
        route_name, polyline_idx = polyline_key

        logger.info("\n" + "="*70)
        logger.info(f"POLYLINE: {route_name} - Segment {polyline_idx}")
        logger.info("="*70)
        logger.info(f"  Data points: {len(df)}")
        logger.info(f"  Segments: {df['segment_id'].nunique()}")
        logger.info(f"  Stops detected: {df['stop_name'].notna().sum()}")
        logger.info(f"  ETAs calculated: {df['eta_seconds'].notna().sum()}")

        # Check if we have enough segments
        if df['segment_id'].nunique() < 2:
            logger.info(f"  ⚠ Only {df['segment_id'].nunique()} segment(s) - skipping this polyline")
            continue

        # Create polyline-specific directory
        polyline_dir = get_polyline_dir(route_name, polyline_idx)
        polyline_dir.mkdir(parents=True, exist_ok=True)

        # Define paths for this polyline
        polyline_data_path = polyline_dir / "data.csv"
        train_path = polyline_dir / "train.csv"
        test_path = polyline_dir / "test.csv"
        model_path = polyline_dir / "model.pth"
        scaler_path = polyline_dir / "scaler.pkl"

        # Save polyline data
        logger.info(f"\nSaving polyline data to {polyline_dir}/")
        save_csv(df, polyline_data_path, "polyline data")

        # 1. Split into train/test
        logger.info(f"\n1. Splitting into train/test (ratio: {test_ratio:.1%})...")
        train_df, test_df = segmented_train_test_split(
            df,
            test_ratio=test_ratio,
            random_seed=random_seed,
            timestamp_column='timestamp',
            segment_column='segment_id'
        )
        logger.info(f"  ✓ Train: {train_df['segment_id'].nunique()} segments, {len(train_df)} points")
        logger.info(f"  ✓ Test:  {test_df['segment_id'].nunique()} segments, {len(test_df)} points")

        # Save train/test splits
        save_csv(train_df, train_path, "train data")
        save_csv(test_df, test_path, "test data")

        # 2. Clean data
        logger.info("\n2. Cleaning data...")
        def clean_df(df_input, name="data"):
            if 'speed_kmh' in df_input.columns:
                df_input['speed_kmh'] = df_input['speed_kmh'].fillna(0)
            req_cols = [c for c in input_columns + output_columns if c in df_input.columns]
            initial = len(df_input)
            df_output = df_input.dropna(subset=req_cols)
            if len(df_output) < initial:
                logger.info(f"  ✓ Dropped {initial - len(df_output)} rows with missing values in {name}")
            return df_output

        train_df = clean_df(train_df, "train set")
        test_df = clean_df(test_df, "test set")

        if len(train_df) == 0 or len(test_df) == 0:
            logger.info(f"  ⚠ Insufficient data after cleaning - skipping this polyline")
            continue

        # 3. Train or load model
        if train or not model_path.exists():
            logger.info("\n3. TRAINING MODEL")
            logger.info(f"  Architecture: {num_layers} layers, {hidden_size} hidden units, seq_len={sequence_length}")

            try:
                model = train_lstm(
                    train_df,
                    input_columns=input_columns,
                    output_columns=output_columns,
                    sequence_length=sequence_length,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    epochs=epochs,
                    batch_size=batch_size,
                    segment_column='segment_id',
                    verbose=verbose
                )

                logger.info(f"\n  Saving model to {model_path}...")
                model.save(model_path, scaler_path=str(scaler_path))
                logger.info("  ✓ Model trained and saved")
            except Exception as e:
                logger.info(f"  ✗ Training failed: {e}")
                continue
        else:
            logger.info(f"\n3. LOADING CACHED MODEL")
            logger.info(f"  Loading from {model_path}...")

            input_size = len(input_columns)
            output_size = len(output_columns)
            model = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout=dropout
            )
            model.load(model_path, scaler_path=str(scaler_path))
            logger.info(f"  ✓ Loaded LSTM model")

        # 4. Evaluate
        logger.info("\n4. MODEL EVALUATION")

        try:
            results = evaluate_lstm(
                model,
                test_df,
                input_columns=input_columns,
                output_columns=output_columns,
                sequence_length=sequence_length,
                segment_column='segment_id'
            )

            logger.info(f"  Test predictions: {results['num_predictions']}")
            logger.info(f"  Test MSE: {results['mse']:.4f}")
            logger.info(f"  Test RMSE: {results['rmse']:.4f}")
            logger.info(f"  Test MAE: {results['mae']:.4f}")

            # Store model and results
            polyline_models[polyline_key] = (model, results)
        except Exception as e:
            logger.info(f"  ✗ Evaluation failed: {e}")
            continue

    # Summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*70)
    logger.info(f"Successfully trained {len(polyline_models)} models:")
    for (route, idx), (model, results) in sorted(polyline_models.items()):
        polyline_dir = get_polyline_dir(route, idx)
        logger.info(f"  {route} seg {idx}: RMSE={results['rmse']:.4f}, MAE={results['mae']:.4f}")
        logger.info(f"    → {polyline_dir}/")

    return polyline_models


# ============================================================================
# ARIMA PIPELINE
# ============================================================================

def arima_pipeline(
    p: int = 5,
    d: int = 0,
    q: int = 5,
    value_column: str = 'speed_kmh',
    limit_segments: int = None,
    tune_hyperparams: bool = False,
    fit: bool = True,
    **kwargs
):
    """
    ARIMA Pipeline: Load -> Preprocess -> Split -> (Optional Tune) -> Train -> Evaluate.

    Args:
        p: AR order (ignored if tune_hyperparams=True)
        d: Differencing order (ignored if tune_hyperparams=True)
        q: MA order (ignored if tune_hyperparams=True)
        value_column: Column to model
        limit_segments: Optional limit on number of segments
        tune_hyperparams: If True, run hyperparameter search
        fit: Retrain model
        split: Re-run splitting
        segment: Re-run segmentation
        preprocess: Re-run preprocessing
        load: Re-load from database

    Note:
        Pipeline hierarchy is applied externally via apply_pipeline_hierarchy().
        This function only checks its own 'fit' flag.

    Returns:
        dict: Results dictionary containing evaluation metrics
    """
    from ml.evaluation.evaluate import evaluate_arima_new_segment
    from ml.validate.hyperparameters import arima_hyperparameters
    from ml.training.train import fit_arima
    import pickle

    logger.info("="*70)
    logger.info("ARIMA PIPELINE")
    logger.info("="*70)

    # Get parameterized cache paths
    best_params_path = get_cache_path(
        'arima_best_params',
        cache_dir=ARIMA_CACHE_DIR,
        extension='pkl',
        p=p, d=d, q=q,
        value_column=value_column
    )
    model_params_path = get_cache_path(
        'arima_model_params',
        cache_dir=ARIMA_CACHE_DIR,
        extension='pkl',
        p=p, d=d, q=q,
        value_column=value_column
    )

    # Load and split data
    train_df, test_df = split_pipeline(mode='arima', **kwargs)

    # Limit segments if requested
    if limit_segments:
        logger.info(f"Limiting to {limit_segments} segments...")
        train_segments = train_df['segment_id'].unique()[:limit_segments]
        train_df = train_df[train_df['segment_id'].isin(train_segments)].copy()
        test_segments = test_df['segment_id'].unique()[:limit_segments]
        test_df = test_df[test_df['segment_id'].isin(test_segments)].copy()

    # Hyperparameter tuning (optional)
    if tune_hyperparams:
        logger.info("\n" + "="*70)
        logger.info("HYPERPARAMETER TUNING")
        logger.info("="*70)

        try:
            tuning_result = arima_hyperparameters(
                train_df,
                n_segments=5,
                p_range=(0, 5),
                d_range=(0, 2),
                q_range=(0, 5),
                value_column=value_column,
                min_training_length=100,
                max_combinations=150,
                verbose=True
            )

            p = tuning_result['best_p']
            d = tuning_result['best_d']
            q = tuning_result['best_q']

            logger.info(f"\nSaving best parameters to {best_params_path}...")
            with open(best_params_path, 'wb') as f:
                pickle.dump({
                    'p': p, 'd': d, 'q': q,
                    'best_mse': tuning_result['best_mse'],
                    'best_rmse': tuning_result['best_rmse'],
                    'weights': tuning_result['weights']
                }, f)

        except Exception as e:
            logger.info(f"Hyperparameter tuning failed: {e}")
            logger.info("Falling back to default parameters")
    else:
        # Show cached parameters if available
        if best_params_path.exists():
            with open(best_params_path, 'rb') as f:
                cached_params = pickle.load(f)
                logger.info(f"\nCached best parameters: ARIMA({cached_params['p']},{cached_params['d']},{cached_params['q']})")
                logger.info(f"Using command-line parameters: ARIMA({p},{d},{q})")

    # Train model (or load cached)
    pretrained_params = None

    if fit or not model_params_path.exists():
        logger.info("\n" + "="*70)
        logger.info("MODEL TRAINING")
        logger.info("="*70)
        logger.info(f"Training ARIMA({p},{d},{q}) on largest training segment...")

        try:
            train_segments = train_df['segment_id'].unique()
            if len(train_segments) > 0:
                segment_sizes = train_df.groupby('segment_id').size()
                largest_segment_id = segment_sizes.idxmax()
                train_segment = train_df[train_df['segment_id'] == largest_segment_id].copy()

                logger.info(f"  Training segment: {largest_segment_id} ({len(train_segment)} points)")

                train_values = train_segment[value_column].dropna().values
                model = fit_arima(train_values, p=p, d=d, q=q)
                pretrained_params = model.results.params

                with open(model_params_path, 'wb') as f:
                    pickle.dump({
                        'p': p, 'd': d, 'q': q,
                        'params': pretrained_params,
                        'value_column': value_column
                    }, f)
                logger.info("  ✓ Model trained and saved")
        except Exception as e:
            logger.info(f"  Warning: Training failed: {e}")
            pretrained_params = None
    else:
        logger.info("\n" + "="*70)
        logger.info("LOADING CACHED MODEL")
        logger.info("="*70)

        try:
            with open(model_params_path, 'rb') as f:
                cached_model = pickle.load(f)

            if cached_model['p'] == p and cached_model['d'] == d and cached_model['q'] == q:
                pretrained_params = cached_model['params']
                logger.info(f"  ✓ Loaded ARIMA({p},{d},{q}) parameters")
            else:
                logger.info(f"  Warning: Cached model is ARIMA({cached_model['p']},{cached_model['d']},{cached_model['q']}) but requested ARIMA({p},{d},{q})")
                logger.info("  Continuing without warm-start")
        except Exception as e:
            logger.info(f"  Warning: Failed to load cached model: {e}")

    # Evaluate on test segments
    logger.info("\n" + "="*70)
    logger.info("MODEL EVALUATION")
    logger.info("="*70)

    test_segments = test_df['segment_id'].unique()
    logger.info(f"Evaluating ARIMA({p},{d},{q}) on {len(test_segments)} test segments...")

    all_predictions = []
    segment_results = []
    num_evaluated = 0
    num_failed = 0

    for i, segment_id in enumerate(test_segments, 1):
        test_segment = test_df[test_df['segment_id'] == segment_id].copy()

        try:
            result = evaluate_arima_new_segment(
                test_segment,
                p=p, d=d, q=q,
                value_column=value_column,
                min_training_length=100,
                start_params=pretrained_params
            )

            segment_results.append({
                'segment_id': segment_id,
                'mse': result['mse'],
                'rmse': result['rmse'],
                'mae': result['mae'],
                'num_predictions': result['num_predictions'],
                'num_successful': result['num_successful'],
                'success_rate': result['success_rate']
            })

            predictions_df = result['predictions_df']
            predictions_df['segment_id'] = segment_id
            all_predictions.append(predictions_df)
            num_evaluated += 1

            logger.info(f"RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}, Success: {result['success_rate']:.1%}")

        except Exception as e:
            num_failed += 1
            continue

    # Aggregate results
    if num_evaluated > 0:
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        segment_results_df = pd.DataFrame(segment_results)

        successful_preds = all_predictions_df[all_predictions_df['fit_success']]
        overall_mse = successful_preds['squared_error'].mean() if len(successful_preds) > 0 else np.nan
        overall_rmse = np.sqrt(overall_mse) if not np.isnan(overall_mse) else np.nan
        overall_mae = successful_preds['absolute_error'].mean() if len(successful_preds) > 0 else np.nan

        logger.info(f"\n{'='*70}")
        logger.info("OVERALL RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"  Segments evaluated: {num_evaluated}/{len(test_segments)}")
        logger.info(f"  Segments failed: {num_failed}/{len(test_segments)}")
        logger.info(f"  Overall MSE: {overall_mse:.4f}")
        logger.info(f"  Overall RMSE: {overall_rmse:.4f}")
        logger.info(f"  Overall MAE: {overall_mae:.4f}")

        return {
            'overall_mse': overall_mse,
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'all_predictions_df': all_predictions_df,
            'segment_results_df': segment_results_df,
            'num_segments': len(test_segments),
            'num_evaluated': num_evaluated,
            'num_failed': num_failed
        }
    else:
        logger.info(f"\n  No segments successfully evaluated!")
        return None


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    # Configure logging for CLI execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(description="Run Shubble ML Pipelines")
    subparsers = parser.add_subparsers(dest="pipeline", help="Pipeline to run")

    # Load Pipeline
    load_parser = subparsers.add_parser("load", help="Load raw vehicle location data")
    load_parser.add_argument("--load", action="store_true", help="Re-load data from database (triggers all downstream stages)")

    # Preprocess Pipeline
    preprocess_parser = subparsers.add_parser("preprocess", help="Run preprocessing only")
    preprocess_parser.add_argument("--preprocess", action="store_true", help="Re-run preprocessing (triggers all downstream stages)")
    preprocess_parser.add_argument("--load", action="store_true", help="Re-load data from database (triggers all stages)")

    # Stops Pipeline
    stops_parser = subparsers.add_parser("stops", help="Run stops pipeline (stops, ETAs, polyline distances)")
    stops_parser.add_argument("--stops", action="store_true", help="Re-run stops processing (triggers all downstream stages)")
    stops_parser.add_argument("--segment", action="store_true", help="Re-run segmentation (triggers stops + downstream stages)")
    stops_parser.add_argument("--preprocess", action="store_true", help="Re-run preprocessing (triggers all downstream stages)")
    stops_parser.add_argument("--window-size", type=int, default=5, help="Window size for cleaning NaN routes")

    # LSTM Pipeline
    lstm_parser = subparsers.add_parser("lstm", help="Run LSTM pipeline (per-polyline training)")
    lstm_parser.add_argument("--epochs", type=int, default=20)
    lstm_parser.add_argument("--batch-size", type=int, default=64)
    lstm_parser.add_argument("--hidden-size", type=int, default=50)
    lstm_parser.add_argument("--num-layers", type=int, default=2)
    lstm_parser.add_argument("--seq-len", type=int, default=10, dest="sequence_length")
    lstm_parser.add_argument("--train", action="store_true", help="Re-train models")
    lstm_parser.add_argument("--stops", action="store_true", help="Re-run stops processing (triggers train)")
    lstm_parser.add_argument("--segment", action="store_true", help="Re-run segmentation (triggers stops + train)")
    lstm_parser.add_argument("--preprocess", action="store_true", help="Re-run preprocessing (triggers all downstream stages)")
    lstm_parser.add_argument("--load", action="store_true", help="Re-load data from database")
    lstm_parser.add_argument("--limit-polylines", type=int, default=None)
    lstm_parser.add_argument("--window-size", type=int, default=5, help="Window size for cleaning NaN routes")

    # ARIMA Pipeline
    arima_parser = subparsers.add_parser("arima", help="Run ARIMA pipeline")
    arima_parser.add_argument("--split", action="store_true", help="Re-run train/test split (triggers train + fit)")
    arima_parser.add_argument("--segment", action="store_true", help="Re-run segmentation (triggers split + train + fit)")
    arima_parser.add_argument("--preprocess", action="store_true", help="Re-run preprocessing (triggers all downstream stages)")
    arima_parser.add_argument("--load", action="store_true", help="Re-load data from database")
    arima_parser.add_argument("--fit", action="store_true", help="Re-fit model")
    arima_parser.add_argument("--max-timedelta", type=float, default=15)
    arima_parser.add_argument("--max-distance", type=float, default=0.02)
    arima_parser.add_argument("--min-segment-length", type=int, default=3)
    arima_parser.add_argument("--window-size", type=int, default=5, help="Window size for cleaning NaN routes")
    arima_parser.add_argument("--test-ratio", type=float, default=0.2)
    arima_parser.add_argument("--random-seed", type=int, default=42)
    arima_parser.add_argument("--tune-hyperparams", action="store_true")
    arima_parser.add_argument("--p", type=int, default=3)
    arima_parser.add_argument("--d", type=int, default=0)
    arima_parser.add_argument("--q", type=int, default=2)
    arima_parser.add_argument("--value-column", type=str, default="speed_kmh")
    arima_parser.add_argument("--limit-segments", type=int, default=None)

    # Parse and execute
    args = parser.parse_args()
    if args.pipeline is None:
        parser.print_help()
        sys.exit(1)

    kwargs = {k: v for k, v in vars(args).items() if k != 'pipeline'}

    # Apply pipeline hierarchy: if a stage is triggered, all downstream stages are also triggered
    kwargs = apply_pipeline_hierarchy(kwargs)

    if args.pipeline == "load":
        df = load_pipeline(**kwargs)
        logger.info(f"\nLoaded {len(df)} records")
    elif args.pipeline == "preprocess":
        df = preprocess_pipeline(**kwargs)
        logger.info(f"\nPreprocessed {len(df)} records")
    elif args.pipeline == "stops":
        df = stops_pipeline(**kwargs)
        df = eta_pipeline(df=df, **kwargs)
        logger.info(f"\nGenerated stops data (stops, ETAs, polyline distances) for {len(df)} records")
    elif args.pipeline == "lstm":
        polyline_models = lstm_pipeline(**kwargs)
        if polyline_models:
            avg_rmse = sum(r['rmse'] for _, r in polyline_models.values()) / len(polyline_models)
            avg_mae = sum(r['mae'] for _, r in polyline_models.values()) / len(polyline_models)
            logger.info(f"\n✓ LSTM pipeline complete. Trained {len(polyline_models)} models.")
            logger.info(f"  Average RMSE: {avg_rmse:.4f}, Average MAE: {avg_mae:.4f}")
        else:
            logger.info("\n✗ No models were trained successfully.")
    elif args.pipeline == "arima":
        results = arima_pipeline(**kwargs)
        if results:
            logger.info(f"\n✓ ARIMA pipeline complete. Overall RMSE: {results['overall_rmse']:.4f}")
