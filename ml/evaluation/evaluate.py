"""Evaluation utilities for ML models."""
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from ml.models.arima import ARIMAModel


def evaluate_arima(
    test_df: pd.DataFrame,
    value_column: str = 'speed_kmh',
    p: int = 1,
    d: int = 0,
    q: int = 1,
    min_train_size: int = 5
) -> dict:
    """
    Evaluate ARIMA model on test data by fitting a separate model to each segment.

    For each segment in the test data:
    1. Fit an ARIMA(p, d, q) model to the segment
    2. Generate one-step-ahead predictions
    3. Calculate MSE for the segment

    Args:
        test_df: DataFrame containing test data with 'segment_id'
        value_column: Column to evaluate on
        p: AR order
        d: Differencing order
        q: MA order
        min_train_size: Minimum segment length to attempt fitting

    Returns:
        Dictionary containing:
            - 'overall_mse': Overall MSE across all segments
            - 'overall_rmse': Overall RMSE across all segments
            - 'segment_mses': List of MSE values per segment
            - 'num_segments': Total number of segments
            - 'num_fitted': Number of segments successfully fitted
            - 'num_failed': Number of segments that failed to fit
    """
    num_test_segments = test_df['segment_id'].nunique()
    print(f"Evaluating ARIMA({p},{d},{q}) on {num_test_segments} test segments...")

    # Sort by segment and timestamp to ensure correct temporal order
    if 'timestamp' in test_df.columns:
        test_df = test_df.sort_values(['segment_id', 'timestamp'])
    else:
        test_df = test_df.sort_values('segment_id')

    all_predictions = []
    all_actuals = []
    segment_mses = []
    num_fitted = 0
    num_failed = 0
    num_skipped = 0

    print("  Fitting ARIMA models to each segment...")
    for segment_id, group in tqdm(
        test_df.groupby('segment_id', sort=False),
        total=num_test_segments,
        desc="Processing segments"
    ):
        # Extract values and remove NaNs
        values = group[value_column].values
        valid_mask = ~np.isnan(values)
        values_clean = values[valid_mask]

        # Skip if segment too short
        if len(values_clean) < min_train_size:
            num_skipped += 1
            continue

        try:
            # Fit ARIMA model to this segment
            model = ARIMAModel(p=p, d=d, q=q)

            # Suppress statsmodels warnings about starting parameters
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model.fit(values_clean)

            # Get one-step-ahead predictions
            predictions = model.get_one_step_predictions()

            # Align predictions with actuals (predictions may be shorter due to initialization)
            # fittedvalues typically starts after max(p, q, d) observations
            n_pred = len(predictions)
            actuals_aligned = values_clean[-n_pred:]

            # Calculate segment MSE
            segment_mse = mean_squared_error(actuals_aligned, predictions)
            segment_mses.append(segment_mse)

            # Accumulate for overall metrics
            all_predictions.extend(predictions)
            all_actuals.extend(actuals_aligned)

            num_fitted += 1

        except Exception as e:
            # ARIMA fitting can fail for various reasons (non-stationary, etc.)
            print(f"  Segment {segment_id} failed: {e}")
            num_failed += 1
            continue

    # Calculate overall metrics
    if len(all_predictions) > 0:
        overall_mse = mean_squared_error(all_actuals, all_predictions)
        overall_rmse = np.sqrt(overall_mse)
    else:
        overall_mse = float('nan')
        overall_rmse = float('nan')

    print(f"\n  Results:")
    print(f"    Segments fitted: {num_fitted}/{num_test_segments}")
    print(f"    Segments skipped: {num_skipped}/{num_test_segments}")
    print(f"    Segments failed: {num_failed}/{num_test_segments}")
    print(f"    Overall MSE: {overall_mse:.4f}")
    print(f"    Overall RMSE: {overall_rmse:.4f}")
    if segment_mses:
        print(f"    Mean segment MSE: {np.mean(segment_mses):.4f}")
        print(f"    Median segment MSE: {np.median(segment_mses):.4f}")

    return {
        'overall_mse': overall_mse,
        'overall_rmse': overall_rmse,
        'segment_mses': segment_mses,
        'num_segments': num_test_segments,
        'num_fitted': num_fitted,
        'num_skipped': num_skipped,
        'num_failed': num_failed
    }
