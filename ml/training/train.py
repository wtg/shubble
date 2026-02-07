"""Training utilities for ML models."""
import numpy as np
import pandas as pd
import random
import warnings
from tqdm import tqdm
from typing import Tuple, Optional
from pathlib import Path


def filter_segmented(
    df: pd.DataFrame,
    segment_column: str,
    min_length: int
) -> pd.DataFrame:
    """
    Filter out segments with fewer than min_length points.

    Returns a new DataFrame containing only segments that have at least
    min_length data points. Does not modify the original DataFrame.

    Args:
        df: DataFrame with segment_column indicating trip segments
        segment_column: Name of column containing segment IDs
        min_length: Minimum number of points required for a segment to be kept

    Returns:
        Filtered DataFrame (copy) with only segments >= min_length points

    Raises:
        KeyError: If segment_column not found in DataFrame
        ValueError: If min_length < 1

    Example:
        >>> df = pd.DataFrame({
        ...     'segment_id': [1, 1, 2, 2, 2, 3],
        ...     'value': [1, 2, 3, 4, 5, 6]
        ... })
        >>> # Keep only segments with 2+ points
        >>> filtered = filter_segmented(df, 'segment_id', min_length=2)
        >>> filtered['segment_id'].unique()
        array([1, 2])  # Segment 3 removed (only 1 point)
    """
    # Validation
    if segment_column not in df.columns:
        raise KeyError(f"'{segment_column}' column not found in DataFrame")

    if min_length < 1:
        raise ValueError(f"min_length must be >= 1, got {min_length}")

    # Calculate segment sizes
    segment_sizes = df.groupby(segment_column).size()

    # Find segments that meet the minimum length requirement
    valid_segments = segment_sizes[segment_sizes >= min_length].index

    # Filter and return copy
    filtered_df = df[df[segment_column].isin(valid_segments)].copy()

    return filtered_df


def fit_arima(
    data: np.ndarray,
    p: int,
    d: int,
    q: int,
    start_params: Optional[np.ndarray] = None
):
    """
    Fit an ARIMA model to time series data.

    Simple utility function to train ARIMA on a single time series array.
    Suppresses convergence warnings for cleaner output.

    Args:
        data: 1D numpy array or array-like of time series values
        p: AR order (number of lag observations)
        d: Differencing order (number of times to difference)
        q: MA order (size of moving average window)
        start_params: Optional array of starting parameter values for warm-starting
                     the optimization. Should be the params from a previous fit.

    Returns:
        Trained ARIMAModel instance with fitted results

    Raises:
        RuntimeError: If ARIMA fitting fails
        ValueError: If data is empty or too short

    Example:
        >>> segment_values = segment_df['speed_kmh'].dropna().values
        >>> model = fit_arima(segment_values, p=3, d=0, q=2)
        >>> prediction = model.predict(n_periods=1)
        >>> # Warm-start a new fit with previous coefficients
        >>> model2 = fit_arima(new_data, p=3, d=0, q=2, start_params=model.results.params)
    """
    from ml.models.arima import ARIMAModel
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    # Validation
    if len(data) == 0:
        raise ValueError("Data array cannot be empty")

    min_length = max(p + d, q + d, 1)
    if len(data) < min_length:
        raise ValueError(
            f"Data too short: {len(data)} points (need at least {min_length} for ARIMA({p},{d},{q}))"
        )

    # Create and fit model
    model = ARIMAModel(p=p, d=d, q=q)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        try:
            # Pass start_params to fit if provided
            if start_params is not None:
                model.fit(data, start_params=start_params)
            else:
                model.fit(data)
        except Exception as e:
            raise RuntimeError(f"Failed to fit ARIMA({p},{d},{q}): {str(e)}")

    return model


def segmented_train_test_split(
    df: pd.DataFrame,
    timestamp_column: str,
    segment_column: str,
    test_ratio: float = 0.2,
    random_seed: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe with segment_id into train and test sets.

    Randomly selects entire segments (trips) for the test set until the total
    number of data points in the test set reaches or exceeds the desired ratio.
    This prevents data leakage by ensuring that all data points from a single
    trip are either in train or test, never split between both.

    Args:
        df: DataFrame with 'segment_id' column indicating trip segments
        test_ratio: Desired fraction of data points for test set (0 to 1)
        random_seed: Optional random seed for reproducibility
        timestamp_column: Name of column containing timestamps for temporal ordering
        segment_column: Name of column containing segment IDs

    Returns:
        Tuple of (train_df, test_df)

    Raises:
        ValueError: If test_ratio is not between 0 and 1
        ValueError: If df is empty
        KeyError: If 'segment_id' column not found

    Example:
        >>> df = pd.DataFrame({
        ...     'segment_id': [1, 1, 2, 2, 3, 3],
        ...     'value': [1, 2, 3, 4, 5, 6]
        ... })
        >>> train, test = segmented_train_test_split(df, test_ratio=0.3)
    """
    # Validation
    if not 0 <= test_ratio <= 1:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")

    if len(df) == 0:
        raise ValueError("DataFrame cannot be empty")

    if segment_column not in df.columns:
        raise KeyError(f"'{segment_column}' column not found in DataFrame")

    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Get segment sizes
    segment_sizes = df.groupby(segment_column).size()

    # Calculate total data points and target test size
    total_points = len(df)
    target_test_points = int(total_points * test_ratio)

    # Get list of unique segment IDs and shuffle
    segment_ids = segment_sizes.index.tolist()
    random.shuffle(segment_ids)

    # Greedily select segments for test set
    test_segment_ids = []
    test_points = 0

    for seg_id in segment_ids:
        segment_size = segment_sizes[seg_id]

        # Add this segment to test set
        test_segment_ids.append(seg_id)
        test_points += segment_size

        # Stop when we've reached or exceeded the target
        if test_points >= target_test_points:
            break

    # Split dataframe by segment_id membership
    test_segment_ids_set = set(test_segment_ids)
    train_df = df[~df['segment_id'].isin(test_segment_ids_set)].copy()
    test_df = df[df['segment_id'].isin(test_segment_ids_set)].copy()

    # Sort by segment_id and timestamp to preserve temporal ordering
    if timestamp_column in train_df.columns:
        train_df = train_df.sort_values(['segment_id', timestamp_column]).reset_index(drop=True)
        test_df = test_df.sort_values(['segment_id', timestamp_column]).reset_index(drop=True)

    return train_df, test_df


def train_lstm(
    df: pd.DataFrame,
    input_columns: list[str],
    output_columns: list[str],
    sequence_length: int = 10,
    hidden_size: int = 50,
    num_layers: int = 1,
    dropout: float = 0.0,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    device: str = "cpu",
    segment_column: Optional[str] = None,
    verbose: bool = True
):
    """
    Train an LSTM model on the provided DataFrame.

    Args:
        df: DataFrame containing the data
        input_columns: List of column names to use as input features
        output_columns: List of column names to predict
        sequence_length: Length of input sequences
        hidden_size: Number of features in the hidden state
        num_layers: Number of recurrent layers
        dropout: Dropout probability
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Number of epochs
        device: Device to train on ('cpu' or 'cuda')
        segment_column: Optional column to use for segmenting data. 
                        If provided, sequences won't cross segment boundaries.
        verbose: Whether to print training progress.

    Returns:
        Trained LSTMModel instance
    """
    from ml.models.lstm import LSTMModel

    # Prepare data
    X_list = []
    y_list = []
    
    # Define helper to process a single continuous block of data
    def process_block(block_df):
        data_in = block_df[input_columns].values
        data_out = block_df[output_columns].values
        
        if len(block_df) <= sequence_length:
            return

        for i in range(len(block_df) - sequence_length):
            X_list.append(data_in[i : i + sequence_length])
            y_list.append(data_out[i + sequence_length])

    if segment_column and segment_column in df.columns:
        for _, group in df.groupby(segment_column):
            process_block(group)
    else:
        process_block(df)
        
    if not X_list:
        raise ValueError("Not enough data to create sequences with length " + str(sequence_length))

    X = np.array(X_list)
    y = np.array(y_list)

    # Initialize model
    input_size = len(input_columns)
    output_size = len(output_columns)
    
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        device=device
    )

    # Fit model
    model.fit(X, y, verbose=verbose)

    return model

if __name__ == "__main__":
    from ml.pipelines import preprocess_pipeline
    from ml.data.segment import segment_by_consecutive

    print("="*70)
    print("Training Data Splitter Demo")
    print("="*70)
    print()

    # 1. Get data
    print("Step 1: Getting preprocessed data...")
    # Use the pipeline function to get data (cached if available)
    df = preprocess_pipeline(force_recompute=False)
    print(f"  ✓ Loaded {len(df)} records")
    print()

    # 2. Segment data (needed for the split functions to work)
    print("Step 2: Segmenting data for demonstration...")
    # Create segments based on time gaps
    max_timedelta = 300  # 5 minutes
    df = segment_by_consecutive(df, max_timedelta=max_timedelta, segment_column='segment_id')
    print(f"  ✓ Created {df['segment_id'].nunique()} segments")
    print()

    # 3. Filter short segments (using function from this file)
    print("Step 3: Filtering short segments (filter_segmented)...")
    min_length = 5
    print(f"  Removing segments with < {min_length} points")
    filtered_df = filter_segmented(df, 'segment_id', min_length=min_length)
    print(f"  ✓ Remaining segments: {filtered_df['segment_id'].nunique()}")
    print(f"  ✓ Remaining points: {len(filtered_df)}")
    print()

    # 4. Split data (using function from this file)
    print("Step 4: Splitting into train/test sets (segmented_train_test_split)...")
    test_ratio = 0.2
    print(f"  Test ratio: {test_ratio:.1%}")

    # Force resplit to show it working (even if cached files exist)
    # We use force_resplit=True to demonstrate the logic in this file actually running
    train_df, test_df = segmented_train_test_split(
        filtered_df,
        timestamp_column='timestamp',
        segment_column='segment_id',
        test_ratio=test_ratio,
        random_seed=42,
    )

    print()
    print("="*70)
    print("SPLIT RESULTS")
    print("="*70)

    train_points = len(train_df)
    test_points = len(test_df)
    total_points = train_points + test_points

    print(f"Train set: {train_df['segment_id'].nunique()} segments, {train_points} points ({train_points/total_points:.1%})")
    print(f"Test set:  {test_df['segment_id'].nunique()} segments, {test_points} points ({test_points/total_points:.1%})")
    print()

    print("First few rows of Training Data:")
    print(train_df[['vehicle_id', 'segment_id', 'timestamp']].head())
    print()

    print()
    print("="*70)
    print("DEMO COMPLETE")
    print("="*70)
