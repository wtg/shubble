"""Hyperparameter optimization for time series models."""
import numpy as np
import pandas as pd
import warnings
from itertools import product
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from ml.models.arima import ARIMAModel


def arima_hyperparameters(
    train_data: np.ndarray,
    val_data: np.ndarray = None,
    p_range: tuple = (0, 5),
    d_range: tuple = (0, 2),
    q_range: tuple = (0, 5),
    metric: str = 'aic',
    max_combinations: int = None,
    verbose: bool = True
) -> dict:
    """
    Find optimal ARIMA hyperparameters using grid search.

    Searches over combinations of p, d, q to find the best model parameters.

    Args:
        train_data: Training time series data (1D numpy array)
        val_data: Optional validation data for computing validation MSE.
                  If None, uses AIC/BIC on training data.
        p_range: Tuple of (min, max) for AR order p
        d_range: Tuple of (min, max) for differencing order d
        q_range: Tuple of (min, max) for MA order q
        metric: Optimization metric - 'aic', 'bic', or 'mse' (requires val_data)
        max_combinations: Optional limit on number of combinations to try
        verbose: If True, show progress bar and print results

    Returns:
        Dictionary containing:
            - 'best_p': Best AR order
            - 'best_d': Best differencing order
            - 'best_q': Best MA order
            - 'best_score': Best metric value achieved
            - 'best_model': Fitted ARIMAModel with best parameters
            - 'results_df': DataFrame with all tried combinations and scores

    Raises:
        ValueError: If metric='mse' but val_data is None

    Example:
        >>> train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> val = np.array([7.0, 8.0, 9.0])
        >>> result = arima_hyperparameters(
        ...     train, val,
        ...     p_range=(0, 2),
        ...     q_range=(0, 2),
        ...     metric='mse'
        ... )
        >>> print(f"Best ARIMA({result['best_p']},{result['best_d']},{result['best_q']})")
    """
    if metric == 'mse' and val_data is None:
        raise ValueError("metric='mse' requires val_data to be provided")

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
        print(f"Testing {len(combinations)} parameter combinations...")
        print(f"Optimization metric: {metric.upper()}")

    best_score = float('inf')
    best_params = None
    best_model = None
    results = []

    # Suppress warnings during grid search
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        iterator = tqdm(combinations, desc="Grid search") if verbose else combinations

        for p, d, q in iterator:
            try:
                # Skip invalid combinations
                if p == 0 and q == 0:
                    continue

                # Fit model
                model = ARIMAModel(p=p, d=d, q=q)
                model.fit(train_data)

                # Calculate score based on metric
                if metric == 'aic':
                    score = model.aic
                elif metric == 'bic':
                    score = model.bic
                elif metric == 'mse':
                    # One-step-ahead predictions on validation data
                    # Re-fit on combined train+val to get predictions
                    combined_data = np.concatenate([train_data, val_data])
                    val_model = ARIMAModel(p=p, d=d, q=q)
                    val_model.fit(combined_data)
                    predictions = val_model.get_one_step_predictions()

                    # Get predictions for validation period
                    n_train = len(train_data)
                    val_predictions = predictions[n_train:]
                    val_actuals = val_data[:len(val_predictions)]

                    score = mean_squared_error(val_actuals, val_predictions)
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                # Record result
                results.append({
                    'p': p,
                    'd': d,
                    'q': q,
                    metric: score,
                    'converged': True
                })

                # Update best if better
                if score < best_score:
                    best_score = score
                    best_params = (p, d, q)
                    best_model = model

            except Exception:
                # Record failed attempt
                results.append({
                    'p': p,
                    'd': d,
                    'q': q,
                    metric: np.nan,
                    'converged': False
                })
                continue

    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(metric)

    if verbose:
        print(f"\nBest parameters found:")
        print(f"  ARIMA(p={best_params[0]}, d={best_params[1]}, q={best_params[2]})")
        print(f"  {metric.upper()}: {best_score:.4f}")
        print(f"\nConverged: {results_df['converged'].sum()}/{len(results_df)}")
        print(f"\nTop 5 configurations:")
        print(results_df.head(5)[['p', 'd', 'q', metric, 'converged']])

    return {
        'best_p': best_params[0],
        'best_d': best_params[1],
        'best_q': best_params[2],
        'best_score': best_score,
        'best_model': best_model,
        'results_df': results_df
    }


def arima_hyperparameters_segmented(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame = None,
    value_column: str = 'speed_kmh',
    segment_column: str = 'segment_id',
    min_segment_length: int = 10,
    n_segments_sample: int = None,
    **kwargs
) -> dict:
    """
    Find optimal ARIMA hyperparameters across multiple segments.

    Performs hyperparameter search on multiple segments and aggregates results
    to find parameters that work well across different time series.

    Args:
        train_df: Training dataframe with segment_id column
        val_df: Optional validation dataframe
        value_column: Column name containing values to model
        segment_column: Column name containing segment IDs
        min_segment_length: Minimum length for segments to include
        n_segments_sample: Optional number of segments to sample for search
        **kwargs: Additional arguments passed to arima_hyperparameters()

    Returns:
        Dictionary containing best parameters and aggregated results

    Example:
        >>> result = arima_hyperparameters_segmented(
        ...     train_df,
        ...     val_df,
        ...     p_range=(0, 3),
        ...     q_range=(0, 3),
        ...     metric='mse'
        ... )
    """
    # Get segments
    train_segments = []
    val_segments = []

    for seg_id, group in train_df.groupby(segment_column):
        values = group[value_column].values
        valid_mask = ~np.isnan(values)
        values_clean = values[valid_mask]

        if len(values_clean) >= min_segment_length:
            train_segments.append((seg_id, values_clean))

    if val_df is not None:
        for seg_id, group in val_df.groupby(segment_column):
            values = group[value_column].values
            valid_mask = ~np.isnan(values)
            values_clean = values[valid_mask]

            if len(values_clean) >= min_segment_length:
                val_segments.append((seg_id, values_clean))

    # Sample segments if requested
    if n_segments_sample and len(train_segments) > n_segments_sample:
        import random
        train_segments = random.sample(train_segments, n_segments_sample)

    print(f"Running hyperparameter search on {len(train_segments)} segments...")

    # Aggregate scores across segments
    all_results = []

    for i, (seg_id, train_data) in enumerate(train_segments):
        print(f"\nSegment {i+1}/{len(train_segments)} (ID: {seg_id})")

        # Find matching validation segment if available
        val_data = None
        if val_df is not None:
            val_match = [v for s, v in val_segments if s == seg_id]
            if val_match:
                val_data = val_match[0]

        # Run hyperparameter search
        result = arima_hyperparameters(
            train_data,
            val_data,
            verbose=False,
            **kwargs
        )

        all_results.append(result['results_df'])

    # Aggregate results across segments
    combined_df = pd.concat(all_results, ignore_index=True)
    metric = kwargs.get('metric', 'aic')

    # Group by parameters and compute mean score
    param_cols = ['p', 'd', 'q']
    aggregated = combined_df.groupby(param_cols)[metric].agg(['mean', 'std', 'count']).reset_index()
    aggregated = aggregated.sort_values('mean')

    best_row = aggregated.iloc[0]

    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS ACROSS SEGMENTS")
    print(f"{'='*70}")
    print(f"Best parameters (mean {metric.upper()} across segments):")
    print(f"  ARIMA(p={int(best_row['p'])}, d={int(best_row['d'])}, q={int(best_row['q'])})")
    print(f"  Mean {metric.upper()}: {best_row['mean']:.4f} (std: {best_row['std']:.4f})")
    print(f"\nTop 5 configurations:")
    print(aggregated.head(5))

    return {
        'best_p': int(best_row['p']),
        'best_d': int(best_row['d']),
        'best_q': int(best_row['q']),
        'best_score': best_row['mean'],
        'aggregated_results': aggregated,
        'all_results': combined_df
    }
