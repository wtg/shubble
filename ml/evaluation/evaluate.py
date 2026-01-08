"""Evaluation utilities for ML models."""
import numpy as np
import pandas as pd


def evaluate_arima_new_segment(
    test_segment_df: pd.DataFrame,
    p: int,
    d: int,
    q: int,
    value_column: str = 'speed_kmh',
    timestamp_column: str = 'timestamp',
    min_training_length: int = 10,
    start_params: np.ndarray = None
) -> dict:
    """
    Evaluate ARIMA on a new test segment using expanding window with pre-trained parameters.

    Uses the expanding window approach (like evaluate_arima_segment) but on a test segment
    with parameters that were optimized on training data. This evaluates how well the
    trained hyperparameters generalize to new, unseen segments.

    For the test segment:
    1. Fit ARIMA on first min_training_length points, predict next point
    2. Fit ARIMA on first min_training_length+1 points, predict next point
    3. Continue until fitting on first n-1 points and predicting point n
    4. Calculate metrics across all predictions

    If start_params (pretrained coefficients) are provided, they will be used to warm-start
    the first ARIMA fit, and each subsequent fit will be warm-started with the parameters
    from the previous fit. This can significantly speed up training and improve convergence.

    Args:
        test_segment_df: DataFrame containing a single test segment's time series data
        p: AR order (from trained/optimized model)
        d: Differencing order (from trained/optimized model)
        q: MA order (from trained/optimized model)
        value_column: Column containing values to predict (default: 'speed_kmh')
        timestamp_column: Column containing timestamps (default: 'timestamp')
        min_training_length: Minimum number of points for initial training (default: 10)
        start_params: Optional pretrained coefficients from training data to warm-start
                     the optimization. Should be the params from a trained ARIMA model.

    Returns:
        Dictionary containing:
            - 'predictions_df': DataFrame with predictions, actuals, and errors
            - 'mse': Mean squared error
            - 'rmse': Root mean squared error
            - 'mae': Mean absolute error
            - 'num_predictions': Total number of predictions made
            - 'num_successful': Number of successful predictions
            - 'success_rate': Fraction of successful predictions

    Raises:
        ValueError: If test segment is too short for evaluation

    Example:
        >>> # Train on a segment and get coefficients
        >>> train_segment = train_df[train_df['segment_id'] == 5].copy()
        >>> from ml.training.train import fit_arima
        >>> train_model = fit_arima(train_segment['speed_kmh'].values, p=3, d=0, q=2)
        >>> pretrained_params = train_model.results.params
        >>>
        >>> # Evaluate on test segment with warm-starting
        >>> test_segment = test_df[test_df['segment_id'] == 10].copy()
        >>> results = evaluate_arima_new_segment(
        ...     test_segment,
        ...     p=3, d=0, q=2,
        ...     min_training_length=15,
        ...     start_params=pretrained_params
        ... )
        >>> print(f"Test RMSE: {results['rmse']:.4f}")
        >>> print(f"Success rate: {results['success_rate']:.1%}")
    """
    # Use evaluate_arima_segment to perform expanding window evaluation with warm-start
    predictions_df = evaluate_arima_segment(
        test_segment_df,
        p=p,
        d=d,
        q=q,
        value_column=value_column,
        timestamp_column=timestamp_column,
        min_training_length=min_training_length,
        start_params=start_params
    )

    # Calculate metrics only on successful predictions
    successful_predictions = predictions_df[predictions_df['fit_success']]

    if len(successful_predictions) == 0:
        # No successful predictions
        return {
            'predictions_df': predictions_df,
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'num_predictions': len(predictions_df),
            'num_successful': 0,
            'success_rate': 0.0
        }

    # Calculate metrics
    mse = successful_predictions['squared_error'].mean()
    rmse = np.sqrt(mse)
    mae = successful_predictions['absolute_error'].mean()
    num_predictions = len(predictions_df)
    num_successful = len(successful_predictions)
    success_rate = num_successful / num_predictions

    return {
        'predictions_df': predictions_df,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'num_predictions': num_predictions,
        'num_successful': num_successful,
        'success_rate': success_rate
    }


def evaluate_arima_segment(
    segment_df: pd.DataFrame,
    p: int,
    d: int,
    q: int,
    value_column: str = 'speed_kmh',
    timestamp_column: str = 'timestamp',
    min_training_length: int = 10,
    start_params: np.ndarray = None
) -> pd.DataFrame:
    """
    Evaluate ARIMA using expanding window one-step-ahead predictions on a single segment.

    This function implements an expanding window approach:
    1. Fit ARIMA on first min_training_length points, predict point min_training_length + 1
    2. Fit ARIMA on first min_training_length + 1 points, predict point min_training_length + 2
    3. Continue until fitting on first n-1 points and predicting point n

    This evaluates one-step-ahead prediction performance where the model is retrained
    at each step with all available data up to that point. If start_params is provided,
    each fit will be warm-started with the parameters from the previous fit.

    Args:
        segment_df: DataFrame containing a single segment's time series data
        p: AR order (number of lag observations)
        d: Differencing order
        q: MA order
        value_column: Column containing values to predict (default: 'speed_kmh')
        timestamp_column: Column containing timestamps (default: 'timestamp')
        min_training_length: Minimum number of points to use for initial training (default: 10)
        start_params: Optional array of starting parameter values for the first fit.
                     After the first fit, each subsequent fit will be warm-started with
                     the parameters from the previous fit.

    Returns:
        DataFrame with columns:
            - 'position': Index of the predicted point in the original data
            - 'training_size': Number of points used for training
            - 'actual': Actual value
            - 'predicted': Predicted value
            - 'error': actual - predicted
            - 'squared_error': (actual - predicted)²
            - 'absolute_error': |actual - predicted|
            - 'timestamp': Timestamp of the predicted point (if available)
            - 'fit_success': Whether the ARIMA fit succeeded

    Raises:
        ValueError: If segment_df is too short (< min_training_length + 1)
        KeyError: If value_column not found

    Example:
        >>> segment_df = train_df[train_df['segment_id'] == 5].copy()
        >>> results_df = evaluate_arima_segment(
        ...     segment_df,
        ...     p=3, d=0, q=2,
        ...     min_training_length=10
        ... )
        >>> print(f"Mean squared error: {results_df['squared_error'].mean():.4f}")
        >>> print(f"Successful fits: {results_df['fit_success'].sum()}/{len(results_df)}")
    """
    from ml.training.train import fit_arima

    # Validation
    if value_column not in segment_df.columns:
        raise KeyError(f"Column '{value_column}' not found in DataFrame")

    # Sort by timestamp if available
    if timestamp_column in segment_df.columns:
        segment_df = segment_df.sort_values(timestamp_column).reset_index(drop=True)

    # Extract and clean values
    values = segment_df[value_column].dropna().values
    n_points = len(values)

    if n_points < min_training_length + 1:
        raise ValueError(
            f"Segment too short: {n_points} points (need at least {min_training_length + 1})"
        )

    # Prepare to collect results
    prediction_records = []

    # Track parameters from previous fit for warm-starting
    current_params = start_params

    # Expanding window predictions: train on [0:i], predict i
    for i in range(min_training_length, n_points):
        training_data = values[:i]
        actual_value = values[i]
        training_size = len(training_data)

        fit_success = False
        predicted_value = np.nan

        try:
            # Fit ARIMA on training window with warm-start if available
            model = fit_arima(training_data, p=p, d=d, q=q, start_params=current_params)

            # Predict next point (one-step-ahead)
            prediction = model.predict(n_periods=1)
            predicted_value = float(prediction[0])
            fit_success = True

            # Update parameters for next iteration (warm-start)
            current_params = model.results.params

        except Exception:
            # If fitting fails, record NaN prediction
            # Don't update current_params - keep previous successful params
            predicted_value = np.nan

        # Calculate errors
        if np.isnan(predicted_value):
            error = np.nan
            squared_error = np.nan
            absolute_error = np.nan
        else:
            error = actual_value - predicted_value
            squared_error = error ** 2
            absolute_error = abs(error)

        # Build record
        record = {
            'position': i,
            'training_size': training_size,
            'actual': actual_value,
            'predicted': predicted_value,
            'error': error,
            'squared_error': squared_error,
            'absolute_error': absolute_error,
            'fit_success': fit_success
        }

        # Add timestamp if available
        if timestamp_column in segment_df.columns:
            # Get timestamp from original dataframe
            # Need to map back to original indices (before dropna)
            valid_indices = segment_df[segment_df[value_column].notna()].index
            if i < len(valid_indices):
                orig_idx = valid_indices[i]
                record['timestamp'] = segment_df.loc[orig_idx, timestamp_column]

        prediction_records.append(record)

    # Create results DataFrame
    results_df = pd.DataFrame(prediction_records)

    return results_df


def evaluate_lstm(
    model,
    test_df: pd.DataFrame,
    input_columns: list[str],
    output_columns: list[str],
    sequence_length: int = 10,
    segment_column: str = 'segment_id'
) -> dict:
    """
    Evaluate LSTM model on test data.

    Generates sequences from test data, makes predictions, and calculates
    evaluation metrics (MSE, RMSE, MAE).

    Args:
        model: Trained LSTMModel instance
        test_df: Test DataFrame with input and output features
        input_columns: List of column names to use as input features
        output_columns: List of column names to predict
        sequence_length: Length of input sequences
        segment_column: Column name for segment IDs (sequences won't cross segments)

    Returns:
        Dictionary containing:
            - 'predictions': Array of predictions
            - 'actuals': Array of actual values
            - 'errors': Array of prediction errors
            - 'mse': Mean squared error
            - 'rmse': Root mean squared error
            - 'mae': Mean absolute error
            - 'num_predictions': Total number of predictions made

    Example:
        >>> from ml.models.lstm import LSTMModel
        >>> # Assume model is already trained
        >>> results = evaluate_lstm(
        ...     model,
        ...     test_df,
        ...     input_columns=['latitude', 'longitude', 'speed_kmh'],
        ...     output_columns=['eta_seconds'],
        ...     sequence_length=10
        ... )
        >>> print(f"Test MSE: {results['mse']:.4f}")
        >>> print(f"Test RMSE: {results['rmse']:.4f}")
    """
    # Prepare test sequences
    X_list = []
    y_list = []

    # Helper to process a single continuous block of data
    def process_block(block_df):
        data_in = block_df[input_columns].values
        data_out = block_df[output_columns].values

        if len(block_df) <= sequence_length:
            return

        for i in range(len(block_df) - sequence_length):
            X_list.append(data_in[i : i + sequence_length])
            y_list.append(data_out[i + sequence_length])

    # Group by segment to avoid crossing segment boundaries
    if segment_column and segment_column in test_df.columns:
        for _, group in test_df.groupby(segment_column):
            process_block(group)
    else:
        process_block(test_df)

    if not X_list:
        raise ValueError(
            f"Not enough data to create sequences with length {sequence_length}. "
            f"Test set has {len(test_df)} points."
        )

    X_test = np.array(X_list)
    y_test = np.array(y_list)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate errors
    errors = y_test - predictions
    squared_errors = errors ** 2
    absolute_errors = np.abs(errors)

    # Calculate metrics
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    mae = np.mean(absolute_errors)

    return {
        'predictions': predictions,
        'actuals': y_test,
        'errors': errors,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'num_predictions': len(predictions)
    }
