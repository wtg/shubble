"""
ML Pipeline Utilities

Complete end-to-end pipelines for preprocessing and train/test splitting.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache subdirectories
SHARED_CACHE_DIR = CACHE_DIR / "shared"
ARIMA_CACHE_DIR = CACHE_DIR / "arima"
LSTM_CACHE_DIR = CACHE_DIR / "lstm"

# Create subdirectories
for cache_dir in [SHARED_CACHE_DIR, ARIMA_CACHE_DIR, LSTM_CACHE_DIR]:
    cache_dir.mkdir(parents=True, exist_ok=True)

# Shared pipeline caches (used by all pipelines)
RAW_CSV = SHARED_CACHE_DIR / "locations_raw.csv"
PREPROCESSED_CSV = SHARED_CACHE_DIR / "locations_preprocessed.csv"
SEGMENTED_CSV = SHARED_CACHE_DIR / "locations_segmented.csv"
STOPS_PREPROCESSED_CSV = SHARED_CACHE_DIR / "stops_preprocessed.csv"

# ARIMA-specific caches
ARIMA_TRAIN_CSV = ARIMA_CACHE_DIR / "arima_train.csv"
ARIMA_TEST_CSV = ARIMA_CACHE_DIR / "arima_test.csv"
ARIMA_BEST_PARAMS_PATH = ARIMA_CACHE_DIR / "arima_best_params.pkl"
ARIMA_MODEL_PARAMS_PATH = ARIMA_CACHE_DIR / "arima_model_params.pkl"

# LSTM-specific caches
LSTM_TRAIN_CSV = LSTM_CACHE_DIR / "lstm_train.csv"
LSTM_TEST_CSV = LSTM_CACHE_DIR / "lstm_test.csv"
LSTM_MODEL_PATH = LSTM_CACHE_DIR / "lstm_model.pth"


def get_cache_paths(mode: str) -> dict[str, Path]:
    """
    Get cache file paths for a specific pipeline mode.

    Args:
        mode: Pipeline mode ('arima' or 'lstm')

    Returns:
        Dictionary with keys: 'train', 'test'
    """
    if mode == 'arima':
        return {'train': ARIMA_TRAIN_CSV, 'test': ARIMA_TEST_CSV}
    elif mode == 'lstm':
        return {'train': LSTM_TRAIN_CSV, 'test': LSTM_TEST_CSV}
    else:
        # Fallback for other modes
        cache_dir = CACHE_DIR / mode
        cache_dir.mkdir(parents=True, exist_ok=True)
        return {
            'train': cache_dir / f"{mode}_train.csv",
            'test': cache_dir / f"{mode}_test.csv"
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _load_cached_csv(path: Path, description: str) -> Optional[pd.DataFrame]:
    """Load a cached CSV file with timestamp parsing."""
    if not path.exists():
        return None

    print(f"Loading {description} from {path}")
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    print(f"Loaded {len(df)} records from cache")
    return df


def _save_csv(df: pd.DataFrame, path: Path, description: str):
    """Save a DataFrame to CSV."""
    print(f"Saving {description} to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} records")


# ============================================================================
# BASIC PIPELINES
# ============================================================================

def load_pipeline(**kwargs) -> pd.DataFrame:
    """
    Load raw vehicle location data.

    Args:
        force_reload: If True, fetch from database even if cached file exists

    Returns:
        DataFrame with raw vehicle location data
    """
    from ml.data.load import load_vehicle_locations

    force_reload = kwargs.get('force_reload', False)

    # Try to load from cache
    if not force_reload:
        cached_df = _load_cached_csv(RAW_CSV, "vehicle locations")
        if cached_df is not None:
            return cached_df

    # Fetch from database
    print("Fetching vehicle locations from database...")
    df = load_vehicle_locations()
    _save_csv(df, RAW_CSV, "fetched data")
    return df


def preprocess_pipeline(**kwargs) -> pd.DataFrame:
    """
    Run the preprocessing pipeline on vehicle location data.

    Steps:
    1. Load vehicle locations (using load_pipeline)
    2. Convert timestamps to epoch seconds
    3. Add closest route information

    Args:
        force_repreprocess: If True, recompute preprocessing even if cached
        force_reload: Passed to load_pipeline

    Returns:
        Preprocessed DataFrame
    """
    from ml.data.preprocess import to_epoch_seconds, add_closest_points, clean_closest_route

    force_repreprocess = kwargs.get('force_repreprocess', False)
    window_size = kwargs.get('window_size', 5)

    # Try to load from cache
    if not force_repreprocess:
        cached_df = _load_cached_csv(PREPROCESSED_CSV, "preprocessed data")
        if cached_df is not None:
            return cached_df

    print("="*70)
    print("PREPROCESSING PIPELINE")
    print("="*70)

    # Load data
    df = load_pipeline(**kwargs)

    # Add epoch seconds
    print("Step 1/2: Converting timestamps to epoch seconds...")
    to_epoch_seconds(df, 'timestamp', 'epoch_seconds')

    # Add route information
    print("Step 2/3: Adding closest route information...")
    add_closest_points(df, 'latitude', 'longitude', {
        'distance': 'dist_to_route',
        'route_name': 'route',
        'closest_point_lat': 'closest_lat',
        'closest_point_lon': 'closest_lon',
        'polyline_index': 'polyline_idx'
    })

    # Clean NaN route values using surrounding context
    print("Step 3/3: Cleaning NaN route values...")
    df = clean_closest_route(df, route_column='route', polyline_idx_column='polyline_idx', window_size=window_size)

    _save_csv(df, PREPROCESSED_CSV, "preprocessed data")
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

    print("Calculating distance and speed...")

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

    print("  ✓ Calculated segment-local speeds")
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
        max_timedelta: Maximum time gap (seconds) for consecutive points
        max_distance: Maximum distance from route (km) to split segments
        min_segment_length: Minimum points required to keep a segment
        force_resegment: Force re-run of segmentation
        force_repreprocess: Force re-run of preprocessing
        force_reload: Passed to load_pipeline

    Returns:
        Segmented DataFrame with speeds
    """
    from ml.data.preprocess import segment_by_consecutive, filter_segments_by_length

    force_resegment = kwargs.get('force_resegment', False)
    force_repreprocess = kwargs.get('force_repreprocess', False)

    # Try to load from cache
    if not (force_resegment or force_repreprocess):
        cached_df = _load_cached_csv(SEGMENTED_CSV, "segmented data")
        if cached_df is not None:
            return cached_df

    print("="*70)
    print("SEGMENT PIPELINE")
    print("="*70)

    # Step 1: Preprocess
    df = preprocess_pipeline(**kwargs)
    print(f"Step 1/4: Preprocessed {len(df)} location points")

    # Step 2: Segment
    print(f"Step 2/4: Segmenting (max gap: {max_timedelta}s, max distance: {max_distance} km)...")
    df = segment_by_consecutive(
        df,
        max_timedelta=max_timedelta,
        segment_column='segment_id',
        distance_column='dist_to_route',
        max_distance_to_route=max_distance
    )
    print(f"  ✓ Created {df['segment_id'].nunique()} segments")

    # Step 3: Add speed
    print("Step 3/4: Adding speed calculations...")
    df = speed_pipeline(df, **kwargs)

    # Step 4: Filter short segments
    print(f"Step 4/4: Filtering segments < {min_segment_length} points...")
    initial_segments = df['segment_id'].nunique()
    df = filter_segments_by_length(df, 'segment_id', min_segment_length)
    final_segments = df['segment_id'].nunique()
    print(f"  ✓ Kept {final_segments}/{initial_segments} segments")

    _save_csv(df, SEGMENTED_CSV, "segmented data")
    return df


def stops_pipeline(**kwargs) -> pd.DataFrame:
    """
    Stops Pipeline: Segment -> Add Stops -> Filter Rows After Stop -> Add ETAs -> Add Polyline Distances.

    This pipeline prepares data for stop-based predictions by:
    1. Segmenting vehicle location data
    2. Adding stop information (detecting when vehicles are at stops)
    3. Filtering rows after the last stop (also removes segments without stops)
    4. Calculating ETAs to the next stop
    5. Calculating polyline distances (distance from start, distance to end)

    Args:
        force_restops: Force re-computation of stops preprocessing
        force_resegment: Force re-run of segmentation
        force_repreprocess: Force re-run of preprocessing

    Returns:
        DataFrame with added columns: 'stop_name', 'stop_route', 'eta_seconds',
        'dist_from_start', 'dist_to_end', 'polyline_length'
    """
    from ml.data.preprocess import (
        add_stops, filter_rows_after_stop, add_eta, add_polyline_distances
    )

    force_restops = kwargs.get('force_restops', False)
    force_resegment = kwargs.get('force_resegment', False)
    force_repreprocess = kwargs.get('force_repreprocess', False)

    # Try to load from cache
    if not (force_restops or force_resegment or force_repreprocess):
        cached_df = _load_cached_csv(STOPS_PREPROCESSED_CSV, "stops preprocessed data")
        if cached_df is not None:
            return cached_df

    print("="*70)
    print("STOPS PIPELINE")
    print("="*70)

    # Step 1: Get segmented data
    df = segment_pipeline(**kwargs)

    # Step 2: Add stops
    print("Step 1/4: Adding stop information...")
    add_stops(df, 'latitude', 'longitude', {
        'route_name': 'stop_route',
        'stop_name': 'stop_name'
    })

    # Step 3: Filter rows after last stop (also removes segments without stops)
    print("Step 2/4: Filtering rows after last stop...")
    initial_points = len(df)
    initial_segments = df['segment_id'].nunique()
    df = filter_rows_after_stop(df, 'segment_id', 'stop_name')
    removed_points = initial_points - len(df)
    final_segments = df['segment_id'].nunique()
    removed_segments = initial_segments - final_segments
    print(f"  ✓ Removed {removed_points} points ({removed_points/initial_points*100:.1f}%)")
    print(f"  ✓ Removed {removed_segments} segments without stops")

    # Step 4: Add ETAs
    print("Step 3/4: Calculating ETAs...")
    df = add_eta(df, 'stop_name', 'epoch_seconds', 'eta_seconds')
    print(f"  ✓ Calculated ETAs for {len(df)} points")

    # Step 5: Add polyline distances
    print("Step 4/4: Calculating polyline distances...")
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
    print(f"  ✓ Calculated polyline distances for {len(df)} points")

    _save_csv(df, STOPS_PREPROCESSED_CSV, "stops preprocessed data")
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
        force_resplit: Force re-run of splitting
        force_resegment: Force re-run of segmentation
        force_repreprocess: Force re-run of preprocessing

    Returns:
        Tuple of (train_df, test_df)
    """
    from ml.training.train import segmented_train_test_split

    # Validate mode
    if mode not in ['arima', 'lstm']:
        raise ValueError(f"mode must be 'arima' or 'lstm', got '{mode}'")

    use_eta_data = (mode == 'lstm')
    force_resplit = kwargs.get('force_resplit', False)
    force_resegment = kwargs.get('force_resegment', False)
    force_repreprocess = kwargs.get('force_repreprocess', False)

    # Get cache paths
    cache_paths = get_cache_paths(mode)
    train_csv, test_csv = cache_paths['train'], cache_paths['test']

    # Try to load from cache
    if not (force_resplit or force_resegment or force_repreprocess):
        if train_csv.exists() and test_csv.exists():
            cache_dir = ARIMA_CACHE_DIR if mode == 'arima' else LSTM_CACHE_DIR
            print(f"Loading cached train/test split from {cache_dir}")
            train_df = _load_cached_csv(train_csv, "train data")
            test_df = _load_cached_csv(test_csv, "test data")
            if train_df is not None and test_df is not None:
                return train_df, test_df

    print("="*70)
    print(f"SPLIT PIPELINE ({mode.upper()} MODE)")
    print("="*70)

    # Load data based on mode
    if use_eta_data:
        print("Loading stops preprocessed data...")
        df = stops_pipeline(**kwargs)
    else:
        print("Loading segmented data...")
        df = segment_pipeline(**kwargs)

    print(f"  ✓ Loaded {len(df)} points, {df['segment_id'].nunique()} segments")

    if len(df) == 0:
        raise ValueError("No valid segments found")

    # Split into train/test
    print(f"Splitting into train/test (ratio: {test_ratio:.1%}, seed: {random_seed})...")
    train_df, test_df = segmented_train_test_split(
        df,
        test_ratio=test_ratio,
        random_seed=random_seed,
        timestamp_column='timestamp',
        segment_column='segment_id'
    )

    # Save split data
    cache_dir = ARIMA_CACHE_DIR if mode == 'arima' else LSTM_CACHE_DIR
    print(f"Saving split data to {cache_dir}")
    _save_csv(train_df, train_csv, f"{mode} train data")
    _save_csv(test_df, test_csv, f"{mode} test data")

    print(f"  Train: {train_df['segment_id'].nunique()} segments, {len(train_df)} points")
    print(f"  Test:  {test_df['segment_id'].nunique()} segments, {len(test_df)} points")
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
        **kwargs: Passed to preprocess_pipeline if df is None (e.g., force_repreprocess)

    Returns:
        Dictionary mapping (route_name, polyline_index) tuples to DataFrames

    Example:
        >>> polyline_dfs = split_by_polyline_pipeline()
        >>> for (route, idx), df in polyline_dfs.items():
        ...     print(f"{route} segment {idx}: {len(df)} points")
    """
    from ml.data.preprocess import split_by_route_polyline_index

    print("="*70)
    print("SPLIT BY POLYLINE PIPELINE")
    print("="*70)

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
    print(f"Splitting {len(df)} points by route and polyline index...")
    polyline_dfs = split_by_route_polyline_index(
        df,
        route_column='route',
        polyline_index_column='polyline_idx'
    )

    print(f"  ✓ Split into {len(polyline_dfs)} unique polylines:")
    for (route, idx), polyline_df in sorted(polyline_dfs.items()):
        print(f"    - {route} segment {idx}: {len(polyline_df)} points")

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
    dropout: float = 0.1,
    epochs: int = 10,
    batch_size: int = 64,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    verbose: bool = True,
    force_retrain: bool = True,
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
        force_retrain: If True, retrain models even if cached
        limit_polylines: Optional limit on number of polylines to process
        **kwargs: Additional arguments passed to stops_pipeline and split_by_polyline_pipeline

    Returns:
        Dictionary mapping (route, polyline_idx) to (model, results) tuples
    """
    from ml.training.train import train_lstm, segmented_train_test_split
    from ml.evaluation.evaluate import evaluate_lstm
    from ml.models.lstm import LSTMModel

    print("="*70)
    print("LSTM PIPELINE (PER-POLYLINE TRAINING)")
    print("="*70)

    # Step 1: Get stops preprocessed data (segmented, stops added, ETAs calculated, polyline distances, filtered)
    stops_df = stops_pipeline(**kwargs)

    # Step 2: Split by polyline
    polyline_dfs = split_by_polyline_pipeline(df=stops_df)

    # Limit polylines if requested
    if limit_polylines:
        print(f"\nLimiting to {limit_polylines} polylines...")
        polyline_keys = sorted(polyline_dfs.keys())[:limit_polylines]
        polyline_dfs = {k: polyline_dfs[k] for k in polyline_keys}
        print(f"  ✓ Processing {len(polyline_dfs)} polylines")

    # Store results for each polyline
    polyline_models = {}

    # Step 3: Process each polyline
    for polyline_key, df in sorted(polyline_dfs.items()):
        route_name, polyline_idx = polyline_key

        print("\n" + "="*70)
        print(f"POLYLINE: {route_name} - Segment {polyline_idx}")
        print("="*70)
        print(f"  Data points: {len(df)}")
        print(f"  Segments: {df['segment_id'].nunique()}")
        print(f"  Stops detected: {df['stop_name'].notna().sum()}")
        print(f"  ETAs calculated: {df['eta_seconds'].notna().sum()}")

        # Check if we have enough segments
        if df['segment_id'].nunique() < 2:
            print(f"  ⚠ Only {df['segment_id'].nunique()} segment(s) - skipping this polyline")
            continue

        # Create polyline-specific directory
        safe_route = route_name.replace(' ', '_').replace('/', '_')
        polyline_dir = LSTM_CACHE_DIR / f"{safe_route}_{polyline_idx}"
        polyline_dir.mkdir(parents=True, exist_ok=True)

        # Define paths for this polyline
        polyline_data_path = polyline_dir / "data.csv"
        train_path = polyline_dir / "train.csv"
        test_path = polyline_dir / "test.csv"
        model_path = polyline_dir / "model.pth"

        # Save polyline data
        print(f"\nSaving polyline data to {polyline_dir}/")
        _save_csv(df, polyline_data_path, "polyline data")

        # 1. Split into train/test
        print(f"\n1. Splitting into train/test (ratio: {test_ratio:.1%})...")
        train_df, test_df = segmented_train_test_split(
            df,
            test_ratio=test_ratio,
            random_seed=random_seed,
            timestamp_column='timestamp',
            segment_column='segment_id'
        )
        print(f"  ✓ Train: {train_df['segment_id'].nunique()} segments, {len(train_df)} points")
        print(f"  ✓ Test:  {test_df['segment_id'].nunique()} segments, {len(test_df)} points")

        # Save train/test splits
        _save_csv(train_df, train_path, "train data")
        _save_csv(test_df, test_path, "test data")

        # 2. Clean data
        print("\n2. Cleaning data...")
        def clean_df(df_input, name="data"):
            if 'speed_kmh' in df_input.columns:
                df_input['speed_kmh'] = df_input['speed_kmh'].fillna(0)
            req_cols = [c for c in input_columns + output_columns if c in df_input.columns]
            initial = len(df_input)
            df_output = df_input.dropna(subset=req_cols)
            if len(df_output) < initial:
                print(f"  ✓ Dropped {initial - len(df_output)} rows with missing values in {name}")
            return df_output

        train_df = clean_df(train_df, "train set")
        test_df = clean_df(test_df, "test set")

        if len(train_df) == 0 or len(test_df) == 0:
            print(f"  ⚠ Insufficient data after cleaning - skipping this polyline")
            continue

        # 3. Train or load model
        if force_retrain or not model_path.exists():
            print("\n3. TRAINING MODEL")
            print(f"  Architecture: {num_layers} layers, {hidden_size} hidden units, seq_len={sequence_length}")

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

                print(f"\n  Saving model to {model_path}...")
                model.save(model_path)
                print("  ✓ Model trained and saved")
            except Exception as e:
                print(f"  ✗ Training failed: {e}")
                continue
        else:
            print(f"\n3. LOADING CACHED MODEL")
            print(f"  Loading from {model_path}...")

            input_size = len(input_columns)
            output_size = len(output_columns)
            model = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout=dropout
            )
            model.load(model_path)
            print(f"  ✓ Loaded LSTM model")

        # 4. Evaluate
        print("\n4. MODEL EVALUATION")

        try:
            results = evaluate_lstm(
                model,
                test_df,
                input_columns=input_columns,
                output_columns=output_columns,
                sequence_length=sequence_length,
                segment_column='segment_id'
            )

            print(f"  Test predictions: {results['num_predictions']}")
            print(f"  Test MSE: {results['mse']:.4f}")
            print(f"  Test RMSE: {results['rmse']:.4f}")
            print(f"  Test MAE: {results['mae']:.4f}")

            # Store model and results
            polyline_models[polyline_key] = (model, results)
        except Exception as e:
            print(f"  ✗ Evaluation failed: {e}")
            continue

    # Summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"Successfully trained {len(polyline_models)} models:")
    for (route, idx), (model, results) in sorted(polyline_models.items()):
        safe_route = route.replace(' ', '_').replace('/', '_')
        polyline_dir = LSTM_CACHE_DIR / f"{safe_route}_{idx}"
        print(f"  {route} seg {idx}: RMSE={results['rmse']:.4f}, MAE={results['mae']:.4f}")
        print(f"    → {polyline_dir}/")

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
    force_refit: bool = True,
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
        force_refit: If True, retrain model even if cached

    Returns:
        dict: Results dictionary containing evaluation metrics
    """
    from ml.evaluation.evaluate import evaluate_arima_new_segment
    from ml.validate.hyperparameters import arima_hyperparameters
    from ml.training.train import fit_arima
    import pickle

    print("="*70)
    print("ARIMA PIPELINE")
    print("="*70)

    # Load and split data
    train_df, test_df = split_pipeline(mode='arima', **kwargs)

    # Limit segments if requested
    if limit_segments:
        print(f"Limiting to {limit_segments} segments...")
        train_segments = train_df['segment_id'].unique()[:limit_segments]
        train_df = train_df[train_df['segment_id'].isin(train_segments)].copy()
        test_segments = test_df['segment_id'].unique()[:limit_segments]
        test_df = test_df[test_df['segment_id'].isin(test_segments)].copy()

    # Hyperparameter tuning (optional)
    if tune_hyperparams:
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING")
        print("="*70)

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

            print(f"\nSaving best parameters to {ARIMA_BEST_PARAMS_PATH}...")
            with open(ARIMA_BEST_PARAMS_PATH, 'wb') as f:
                pickle.dump({
                    'p': p, 'd': d, 'q': q,
                    'best_mse': tuning_result['best_mse'],
                    'best_rmse': tuning_result['best_rmse'],
                    'weights': tuning_result['weights']
                }, f)

        except Exception as e:
            print(f"Hyperparameter tuning failed: {e}")
            print("Falling back to default parameters")
    else:
        # Show cached parameters if available
        if ARIMA_BEST_PARAMS_PATH.exists():
            with open(ARIMA_BEST_PARAMS_PATH, 'rb') as f:
                cached_params = pickle.load(f)
                print(f"\nCached best parameters: ARIMA({cached_params['p']},{cached_params['d']},{cached_params['q']})")
                print(f"Using command-line parameters: ARIMA({p},{d},{q})")

    # Train model (or load cached)
    pretrained_params = None

    if force_refit or not ARIMA_MODEL_PARAMS_PATH.exists():
        print("\n" + "="*70)
        print("MODEL TRAINING")
        print("="*70)
        print(f"Training ARIMA({p},{d},{q}) on largest training segment...")

        try:
            train_segments = train_df['segment_id'].unique()
            if len(train_segments) > 0:
                segment_sizes = train_df.groupby('segment_id').size()
                largest_segment_id = segment_sizes.idxmax()
                train_segment = train_df[train_df['segment_id'] == largest_segment_id].copy()

                print(f"  Training segment: {largest_segment_id} ({len(train_segment)} points)")

                train_values = train_segment[value_column].dropna().values
                model = fit_arima(train_values, p=p, d=d, q=q)
                pretrained_params = model.results.params

                with open(ARIMA_MODEL_PARAMS_PATH, 'wb') as f:
                    pickle.dump({
                        'p': p, 'd': d, 'q': q,
                        'params': pretrained_params,
                        'value_column': value_column
                    }, f)
                print("  ✓ Model trained and saved")
        except Exception as e:
            print(f"  Warning: Training failed: {e}")
            pretrained_params = None
    else:
        print("\n" + "="*70)
        print("LOADING CACHED MODEL")
        print("="*70)

        try:
            with open(ARIMA_MODEL_PARAMS_PATH, 'rb') as f:
                cached_model = pickle.load(f)

            if cached_model['p'] == p and cached_model['d'] == d and cached_model['q'] == q:
                pretrained_params = cached_model['params']
                print(f"  ✓ Loaded ARIMA({p},{d},{q}) parameters")
            else:
                print(f"  Warning: Cached model is ARIMA({cached_model['p']},{cached_model['d']},{cached_model['q']}) but requested ARIMA({p},{d},{q})")
                print("  Continuing without warm-start")
        except Exception as e:
            print(f"  Warning: Failed to load cached model: {e}")

    # Evaluate on test segments
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    test_segments = test_df['segment_id'].unique()
    print(f"Evaluating ARIMA({p},{d},{q}) on {len(test_segments)} test segments...")

    all_predictions = []
    segment_results = []
    num_evaluated = 0
    num_failed = 0

    for i, segment_id in enumerate(test_segments, 1):
        print(f"\n  Segment {i}/{len(test_segments)} (ID: {segment_id})...", end=" ")
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

            print(f"RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}, Success: {result['success_rate']:.1%}")

        except Exception as e:
            print(f"Failed: {e}")
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

        print(f"\n{'='*70}")
        print("OVERALL RESULTS")
        print(f"{'='*70}")
        print(f"  Segments evaluated: {num_evaluated}/{len(test_segments)}")
        print(f"  Segments failed: {num_failed}/{len(test_segments)}")
        print(f"  Overall MSE: {overall_mse:.4f}")
        print(f"  Overall RMSE: {overall_rmse:.4f}")
        print(f"  Overall MAE: {overall_mae:.4f}")

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
        print(f"\n  No segments successfully evaluated!")
        return None


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run Shubble ML Pipelines")
    subparsers = parser.add_subparsers(dest="pipeline", help="Pipeline to run")

    # Load Pipeline
    load_parser = subparsers.add_parser("load", help="Load raw vehicle location data")
    load_parser.add_argument("--force-reload", action="store_true", help="Force fetching data from DB")

    # Preprocess Pipeline
    preprocess_parser = subparsers.add_parser("preprocess", help="Run preprocessing only")
    preprocess_parser.add_argument("--force-repreprocess", action="store_true")
    preprocess_parser.add_argument("--force-reload", action="store_true")

    # Stops Pipeline
    stops_parser = subparsers.add_parser("stops", help="Run stops pipeline (stops, ETAs, polyline distances)")
    stops_parser.add_argument("--force-restops", action="store_true")
    stops_parser.add_argument("--force-resegment", action="store_true")
    stops_parser.add_argument("--force-repreprocess", action="store_true")

    # LSTM Pipeline
    lstm_parser = subparsers.add_parser("lstm", help="Run LSTM pipeline (per-polyline training)")
    lstm_parser.add_argument("--epochs", type=int, default=10)
    lstm_parser.add_argument("--batch-size", type=int, default=64)
    lstm_parser.add_argument("--hidden-size", type=int, default=50)
    lstm_parser.add_argument("--num-layers", type=int, default=2)
    lstm_parser.add_argument("--seq-len", type=int, default=10, dest="sequence_length")
    lstm_parser.add_argument("--force-retrain", action="store_true")
    lstm_parser.add_argument("--force-restops", action="store_true")
    lstm_parser.add_argument("--force-resegment", action="store_true")
    lstm_parser.add_argument("--force-repreprocess", action="store_true")
    lstm_parser.add_argument("--force-reload", action="store_true")
    lstm_parser.add_argument("--limit-polylines", type=int, default=None)

    # ARIMA Pipeline
    arima_parser = subparsers.add_parser("arima", help="Run ARIMA pipeline")
    arima_parser.add_argument("--force-resplit", action="store_true")
    arima_parser.add_argument("--force-resegment", action="store_true")
    arima_parser.add_argument("--force-repreprocess", action="store_true")
    arima_parser.add_argument("--force-reload", action="store_true")
    arima_parser.add_argument("--force-refit", action="store_true")
    arima_parser.add_argument("--max-timedelta", type=float, default=15)
    arima_parser.add_argument("--max-distance", type=float, default=0.01)
    arima_parser.add_argument("--min-segment-length", type=int, default=3)
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

    if args.pipeline == "load":
        df = load_pipeline(**kwargs)
        print(f"\nLoaded {len(df)} records")
    elif args.pipeline == "preprocess":
        df = preprocess_pipeline(**kwargs)
        print(f"\nPreprocessed {len(df)} records")
    elif args.pipeline == "stops":
        df = stops_pipeline(**kwargs)
        print(f"\nGenerated stops data (stops, ETAs, polyline distances) for {len(df)} records")
    elif args.pipeline == "lstm":
        polyline_models = lstm_pipeline(**kwargs)
        if polyline_models:
            avg_rmse = sum(r['rmse'] for _, r in polyline_models.values()) / len(polyline_models)
            avg_mae = sum(r['mae'] for _, r in polyline_models.values()) / len(polyline_models)
            print(f"\n✓ LSTM pipeline complete. Trained {len(polyline_models)} models.")
            print(f"  Average RMSE: {avg_rmse:.4f}, Average MAE: {avg_mae:.4f}")
        else:
            print("\n✗ No models were trained successfully.")
    elif args.pipeline == "arima":
        results = arima_pipeline(**kwargs)
        if results:
            print(f"\n✓ ARIMA pipeline complete. Overall RMSE: {results['overall_rmse']:.4f}")
