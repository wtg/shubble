"""Hyperparameter optimization for time series models."""
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm


def arima_hyperparameters(
    train_df: pd.DataFrame,
    n_segments: int = 5,
    p_range: tuple = (0, 5),
    d_range: tuple = (0, 2),
    q_range: tuple = (0, 5),
    value_column: str = 'speed_kmh',
    timestamp_column: str = 'timestamp',
    segment_column: str = 'segment_id',
    min_training_length: int = 10,
    max_combinations: int = None,
    verbose: bool = True
) -> dict:
    """
    Find optimal ARIMA hyperparameters across multiple training segments.

    Runs expanding window hyperparameter search on the n largest segments in the
    training set, then aggregates results using a weighted average (weighted by
    number of data points in each segment). This provides robust hyperparameter
    selection that works well across different trip patterns.

    Args:
        train_df: Training DataFrame with segment_column
        n_segments: Number of largest segments to evaluate (default: 5)
        p_range: Tuple of (min, max) for AR order p
        d_range: Tuple of (min, max) for differencing order d
        q_range: Tuple of (min, max) for MA order q
        value_column: Column containing values to predict (default: 'speed_kmh')
        timestamp_column: Column containing timestamps (default: 'timestamp')
        segment_column: Column containing segment IDs (default: 'segment_id')
        min_training_length: Minimum number of points for initial training (default: 10)
        max_combinations: Optional limit on number of combinations to try per segment
        verbose: If True, show progress and print results

    Returns:
        Dictionary containing:
            - 'best_p': Best AR order (weighted average across segments)
            - 'best_d': Best differencing order
            - 'best_q': Best MA order
            - 'best_mse': Best weighted average MSE
            - 'best_rmse': Best weighted average RMSE
            - 'aggregated_results': DataFrame with weighted results per (p,d,q)
            - 'segment_results': List of individual segment results
            - 'weights': Dictionary mapping segment_id to weight (number of predictions)

    Example:
        >>> from ml.pipelines import split_pipeline
        >>> train_df, _ = split_pipeline()
        >>> result = arima_hyperparameters(
        ...     train_df,
        ...     n_segments=5,
        ...     p_range=(0, 3),
        ...     d_range=(0, 1),
        ...     q_range=(0, 3),
        ...     max_combinations=50
        ... )
        >>> print(f"Best: ARIMA({result['best_p']},{result['best_d']},{result['best_q']})")
        >>> print(f"Weighted RMSE: {result['best_rmse']:.4f}")
    """
    # Select n largest segments
    segment_sizes = train_df.groupby(segment_column).size().sort_values(ascending=False)
    selected_segments = segment_sizes.head(n_segments).index.tolist()
    selected_sizes = segment_sizes.head(n_segments).to_dict()

    if verbose:
        print(f"{'='*70}")
        print(f"ARIMA HYPERPARAMETER SEARCH - Multiple Segments")
        print(f"{'='*70}")
        print(f"Selected {len(selected_segments)} largest segments:")
        for seg_id in selected_segments:
            print(f"  Segment {seg_id}: {selected_sizes[seg_id]} points")
        print()

    # Run hyperparameter search on each segment
    segment_results = []
    weights = {}

    for i, segment_id in enumerate(selected_segments, 1):
        if verbose:
            print(f"\n{'-'*70}")
            print(f"SEGMENT {i}/{len(selected_segments)} (ID: {segment_id})")
            print(f"{'-'*70}")

        # Extract segment
        segment_df = train_df[train_df[segment_column] == segment_id].copy()

        try:
            # Run expanding window hyperparameter search
            result = arima_hyperparameters_expanding_window(
                segment_df,
                p_range=p_range,
                d_range=d_range,
                q_range=q_range,
                value_column=value_column,
                timestamp_column=timestamp_column,
                min_training_length=min_training_length,
                max_combinations=max_combinations,
                verbose=verbose
            )

            # Store results with segment info
            result['segment_id'] = segment_id
            result['segment_size'] = selected_sizes[segment_id]

            # Extract results_df and add segment metadata
            results_df = result['results_df'].copy()
            results_df['segment_id'] = segment_id

            segment_results.append({
                'segment_id': segment_id,
                'segment_size': selected_sizes[segment_id],
                'results_df': results_df,
                'best_p': result['best_p'],
                'best_d': result['best_d'],
                'best_q': result['best_q'],
                'best_mse': result['best_mse'],
                'best_rmse': result['best_rmse']
            })

            # Weight is the number of successful predictions made
            # This is approximately segment_size - min_training_length
            num_predictions = len(result['best_predictions_df']) if 'best_predictions_df' in result else 0
            weights[segment_id] = num_predictions

        except Exception as e:
            if verbose:
                print(f"  Failed on segment {segment_id}: {e}")
            continue

    if not segment_results:
        raise ValueError("No segments successfully completed hyperparameter search")

    # Aggregate results using weighted average
    if verbose:
        print(f"\n{'='*70}")
        print("AGGREGATING RESULTS ACROSS SEGMENTS")
        print(f"{'='*70}")

    # Combine all results DataFrames
    all_results_dfs = [sr['results_df'] for sr in segment_results]
    combined_results = pd.concat(all_results_dfs, ignore_index=True)

    # For each (p, d, q) combination, compute weighted average MSE and RMSE
    param_cols = ['p', 'd', 'q']

    # Group by parameters
    grouped = combined_results.groupby(param_cols)

    aggregated_rows = []
    for params, group in grouped:
        p_val, d_val, q_val = params

        # Get MSE values and weights for this parameter combination
        mse_values = []
        rmse_values = []
        segment_weights = []

        for _, row in group.iterrows():
            seg_id = row['segment_id']
            if pd.notna(row['mse']) and seg_id in weights:
                mse_values.append(row['mse'])
                rmse_values.append(row['rmse'])
                segment_weights.append(weights[seg_id])

        if len(mse_values) == 0:
            # No valid results for this combination
            continue

        # Compute weighted averages
        total_weight = sum(segment_weights)
        weighted_mse = sum(m * w for m, w in zip(mse_values, segment_weights)) / total_weight
        weighted_rmse = sum(r * w for r, w in zip(rmse_values, segment_weights)) / total_weight

        # Also compute unweighted statistics
        mean_mse = np.mean(mse_values)
        std_mse = np.std(mse_values)
        mean_rmse = np.mean(rmse_values)

        aggregated_rows.append({
            'p': int(p_val),
            'd': int(d_val),
            'q': int(q_val),
            'weighted_mse': weighted_mse,
            'weighted_rmse': weighted_rmse,
            'mean_mse': mean_mse,
            'std_mse': std_mse,
            'mean_rmse': mean_rmse,
            'num_segments': len(mse_values),
            'total_weight': total_weight
        })

    # Create aggregated results DataFrame
    aggregated_df = pd.DataFrame(aggregated_rows)
    aggregated_df = aggregated_df.sort_values('weighted_mse')

    if len(aggregated_df) == 0:
        raise ValueError("No valid parameter combinations found")

    # Get best parameters
    best_row = aggregated_df.iloc[0]
    best_p = int(best_row['p'])
    best_d = int(best_row['d'])
    best_q = int(best_row['q'])
    best_weighted_mse = best_row['weighted_mse']
    best_weighted_rmse = best_row['weighted_rmse']

    if verbose:
        print(f"\nBest parameters (weighted by segment predictions):")
        print(f"  ARIMA(p={best_p}, d={best_d}, q={best_q})")
        print(f"  Weighted MSE: {best_weighted_mse:.4f}")
        print(f"  Weighted RMSE: {best_weighted_rmse:.4f}")
        print(f"  Mean MSE: {best_row['mean_mse']:.4f} (±{best_row['std_mse']:.4f})")
        print(f"  Evaluated on: {best_row['num_segments']} segments")

        print(f"\nTop 5 configurations (by weighted MSE):")
        display_cols = ['p', 'd', 'q', 'weighted_mse', 'weighted_rmse', 'mean_mse', 'num_segments']
        print(aggregated_df.head(5)[display_cols].to_string(index=False))

        print(f"\nSegment weights (number of predictions):")
        for seg_id, weight in weights.items():
            print(f"  Segment {seg_id}: {weight} predictions")

    return {
        'best_p': best_p,
        'best_d': best_d,
        'best_q': best_q,
        'best_mse': best_weighted_mse,
        'best_rmse': best_weighted_rmse,
        'aggregated_results': aggregated_df,
        'segment_results': segment_results,
        'weights': weights
    }


def arima_hyperparameters_expanding_window(
    segment_df: pd.DataFrame,
    p_range: tuple = (0, 5),
    d_range: tuple = (0, 2),
    q_range: tuple = (0, 5),
    value_column: str = 'speed_kmh',
    timestamp_column: str = 'timestamp',
    min_training_length: int = 10,
    max_combinations: int = None,
    verbose: bool = True
) -> dict:
    """
    Find optimal ARIMA hyperparameters using expanding window evaluation on a segment.

    Uses evaluate_arima_segment to perform realistic one-step-ahead predictions
    with expanding training windows. This provides a more realistic assessment
    of model performance than traditional train/test splits.

    For each parameter combination (p, d, q):
    - Trains ARIMA on first min_training_length points, predicts next point
    - Trains ARIMA on first min_training_length+1 points, predicts next point
    - Continues until the end of the segment
    - Calculates MSE across all predictions

    Args:
        segment_df: DataFrame containing a single segment's time series data
        p_range: Tuple of (min, max) for AR order p
        d_range: Tuple of (min, max) for differencing order d
        q_range: Tuple of (min, max) for MA order q
        value_column: Column containing values to predict (default: 'speed_kmh')
        timestamp_column: Column containing timestamps (default: 'timestamp')
        min_training_length: Minimum number of points for initial training (default: 10)
        max_combinations: Optional limit on number of combinations to try
        verbose: If True, show progress bar and print results

    Returns:
        Dictionary containing:
            - 'best_p': Best AR order
            - 'best_d': Best differencing order
            - 'best_q': Best MA order
            - 'best_mse': Best MSE achieved
            - 'best_rmse': Best RMSE achieved
            - 'results_df': DataFrame with all tried combinations and scores
            - 'best_predictions_df': Predictions DataFrame from best configuration

    Raises:
        ValueError: If segment_df is too short for evaluation

    Example:
        >>> segment_df = train_df[train_df['segment_id'] == 5].copy()
        >>> result = arima_hyperparameters_expanding_window(
        ...     segment_df,
        ...     p_range=(0, 3),
        ...     d_range=(0, 1),
        ...     q_range=(0, 3),
        ...     min_training_length=15
        ... )
        >>> print(f"Best: ARIMA({result['best_p']},{result['best_d']},{result['best_q']})")
        >>> print(f"Best RMSE: {result['best_rmse']:.4f}")
    """
    from ml.evaluation.evaluate import evaluate_arima_segment

    # Validation
    if value_column not in segment_df.columns:
        raise KeyError(f"Column '{value_column}' not found in DataFrame")

    # Check segment length
    values = segment_df[value_column].dropna().values
    if len(values) < min_training_length + 5:
        raise ValueError(
            f"Segment too short for hyperparameter search: {len(values)} points "
            f"(need at least {min_training_length + 5})"
        )

    # Generate parameter combinations
    p_values = range(p_range[0], p_range[1] + 1)
    d_values = range(d_range[0], d_range[1] + 1)
    q_values = range(q_range[0], q_range[1] + 1)

    # Create all combinations
    combinations = list(product(p_values, d_values, q_values))

    # Limit combinations if requested
    if max_combinations and len(combinations) > max_combinations:
        if verbose:
            print(f"Limiting search to {max_combinations} random combinations out of {len(combinations)} total")
        np.random.shuffle(combinations)
        combinations = combinations[:max_combinations]

    if verbose:
        print(f"Testing {len(combinations)} parameter combinations with expanding window...")
        print(f"Segment length: {len(values)} points")
        print(f"Min training length: {min_training_length}")
        print(f"Predictions per model: {len(values) - min_training_length}")

    best_mse = float('inf')
    best_rmse = float('inf')
    best_params = None
    best_predictions_df = None
    results = []

    iterator = tqdm(combinations, desc="Grid search") if verbose else combinations

    for p, d, q in iterator:
        try:
            # Skip invalid combinations
            if p == 0 and q == 0:
                continue

            # Evaluate using expanding window
            predictions_df = evaluate_arima_segment(
                segment_df,
                p=p,
                d=d,
                q=q,
                value_column=value_column,
                timestamp_column=timestamp_column,
                min_training_length=min_training_length
            )

            # Calculate metrics only on successful predictions
            successful_predictions = predictions_df[predictions_df['fit_success']]

            if len(successful_predictions) == 0:
                # No successful predictions
                results.append({
                    'p': p,
                    'd': d,
                    'q': q,
                    'mse': np.nan,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'num_predictions': 0,
                    'num_successful': 0,
                    'success_rate': 0.0
                })
                continue

            mse = successful_predictions['squared_error'].mean()
            rmse = np.sqrt(mse)
            mae = successful_predictions['absolute_error'].mean()
            num_predictions = len(predictions_df)
            num_successful = len(successful_predictions)
            success_rate = num_successful / num_predictions

            # Record result
            results.append({
                'p': p,
                'd': d,
                'q': q,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'num_predictions': num_predictions,
                'num_successful': num_successful,
                'success_rate': success_rate
            })

            # Update best if better
            if mse < best_mse:
                best_mse = mse
                best_rmse = rmse
                best_params = (p, d, q)
                best_predictions_df = predictions_df

        except Exception as e:
            # Record failed attempt
            if verbose:
                print(f"  Failed ARIMA({p},{d},{q}): {e}")
            results.append({
                'p': p,
                'd': d,
                'q': q,
                'mse': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'num_predictions': 0,
                'num_successful': 0,
                'success_rate': 0.0
            })
            continue

    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mse')

    if verbose and best_params:
        print(f"\n{'='*70}")
        print("HYPERPARAMETER SEARCH RESULTS (Expanding Window)")
        print(f"{'='*70}")
        print(f"Best parameters found:")
        print(f"  ARIMA(p={best_params[0]}, d={best_params[1]}, q={best_params[2]})")
        print(f"  MSE: {best_mse:.4f}")
        print(f"  RMSE: {best_rmse:.4f}")

        best_result = results_df.iloc[0]
        print(f"  MAE: {best_result['mae']:.4f}")
        print(f"  Success rate: {best_result['success_rate']:.1%}")
        print(f"  Successful predictions: {best_result['num_successful']}/{best_result['num_predictions']}")

        print(f"\nTop 5 configurations:")
        print(results_df.head(5)[['p', 'd', 'q', 'mse', 'rmse', 'mae', 'success_rate']])

    if best_params is None:
        raise ValueError("No valid parameter combinations found")

    return {
        'best_p': best_params[0],
        'best_d': best_params[1],
        'best_q': best_params[2],
        'best_mse': best_mse,
        'best_rmse': best_rmse,
        'results_df': results_df,
        'best_predictions_df': best_predictions_df
    }


def lstm_hyperparameters(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    input_columns: list[str],
    output_columns: list[str],
    sequence_length_range: list[int] = [10, 20],
    hidden_size_range: list[int] = [32, 64],
    num_layers_range: list[int] = [1, 2],
    dropout_range: list[float] = [0.0, 0.1],
    learning_rate_range: list[float] = [0.001],
    batch_size_range: list[int] = [32, 64],
    epochs: int = 10,
    max_combinations: int = 20,
    segment_column: str = 'segment_id',
    verbose: bool = True
) -> dict:
    """
    Find optimal LSTM hyperparameters using random search.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        input_columns: Input feature names
        output_columns: Output target names
        sequence_length_range: List of sequence lengths to try
        hidden_size_range: List of hidden sizes to try
        num_layers_range: List of layer counts to try
        dropout_range: List of dropout rates to try
        learning_rate_range: List of learning rates to try
        batch_size_range: List of batch sizes to try
        epochs: Number of epochs to train for each candidate
        max_combinations: Max number of random combinations to try
        segment_column: Column name for segments (to respect boundaries)
        verbose: Print progress

    Returns:
        Dictionary with best parameters and results
    """
    from ml.training.train import train_lstm
    import torch

    # Generate parameter grid
    param_grid = {
        'sequence_length': sequence_length_range,
        'hidden_size': hidden_size_range,
        'num_layers': num_layers_range,
        'dropout': dropout_range,
        'learning_rate': learning_rate_range,
        'batch_size': batch_size_range
    }

    # Generate combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    # Limit combinations
    if len(combinations) > max_combinations:
        if verbose:
            print(f"Limiting search to {max_combinations} random combinations out of {len(combinations)} total")
        np.random.shuffle(combinations)
        combinations = combinations[:max_combinations]

    best_mse = float('inf')
    best_params = None
    results = []
    
    # Helper to prepare validation data
    def prepare_val_data(df, input_cols, output_cols, seq_len):
        X_list = []
        y_list = []
        
        def process_block(block_df):
            data_in = block_df[input_cols].values
            data_out = block_df[output_cols].values
            
            if len(block_df) <= seq_len:
                return

            for i in range(len(block_df) - seq_len):
                X_list.append(data_in[i : i + seq_len])
                y_list.append(data_out[i + seq_len])

        if segment_column and segment_column in df.columns:
            for _, group in df.groupby(segment_column):
                process_block(group)
        else:
            process_block(df)
            
        if not X_list:
             return None, None
             
        return np.array(X_list), np.array(y_list)

    if verbose:
        print(f"Testing {len(combinations)} LSTM configurations...")
        iterator = tqdm(combinations)
    else:
        iterator = combinations

    for params in iterator:
        try:
            # Train model
            model = train_lstm(
                train_df,
                input_columns=input_columns,
                output_columns=output_columns,
                sequence_length=params['sequence_length'],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                learning_rate=params['learning_rate'],
                batch_size=params['batch_size'],
                epochs=epochs,
                segment_column=segment_column,
                device="cpu" # Force CPU for stability during simple search, or "cuda" if available
            )
            
            # Evaluate
            X_val, y_val = prepare_val_data(
                val_df, 
                input_columns, 
                output_columns, 
                params['sequence_length']
            )
            
            if X_val is None:
                continue
                
            y_pred = model.predict(X_val)
            
            mse = np.mean((y_val - y_pred) ** 2)
            rmse = np.sqrt(mse)
            
            results.append({
                **params,
                'mse': mse,
                'rmse': rmse
            })
            
            if mse < best_mse:
                best_mse = mse
                best_params = params
                
        except Exception as e:
            if verbose:
                print(f"Failed configuration {params}: {e}")
            continue

    results_df = pd.DataFrame(results).sort_values('mse')
    
    if verbose:
        print("\nBest LSTM Parameters:")
        print(best_params)
        print(f"Best RMSE: {np.sqrt(best_mse):.4f}")
        print("\nTop 5 Results:")
        print(results_df.head(5))
        
    return {
        'best_params': best_params,
        'best_mse': best_mse,
        'best_rmse': np.sqrt(best_mse),
        'results_df': results_df
    }
