"""ARIMA model deployment utilities."""
import pickle
from pathlib import Path
from typing import Optional
import numpy as np

from ml.cache import get_cache_path, ARIMA_CACHE_DIR

# Get cache directory
CACHE_DIR = ARIMA_CACHE_DIR


def load_arima(
    p: int = 3,
    d: int = 0,
    q: int = 2,
    value_column: str = 'speed_kmh'
) -> dict:
    """
    Load trained ARIMA model parameters from cache.

    This function loads the cached model parameters that were saved during training.
    The returned dictionary contains the ARIMA order (p, d, q), the fitted coefficients,
    and the value column that was used for training.

    These parameters can be used as warm-start parameters for making predictions on
    new data segments using fit_arima with start_params.

    Args:
        p: AR order (default: 3)
        d: Differencing order (default: 0)
        q: MA order (default: 2)
        value_column: Column that was modeled (default: 'speed_kmh')

    Returns:
        Dictionary containing:
            - 'p': AR order
            - 'd': Differencing order
            - 'q': MA order
            - 'params': Fitted model coefficients (numpy array)
            - 'value_column': Name of the modeled column

    Raises:
        FileNotFoundError: If no cached model exists for the given parameters
        ValueError: If cached model parameters don't match requested parameters

    Example:
        >>> # Load model trained on speed data
        >>> model_info = load_arima(p=3, d=0, q=2, value_column='speed_kmh')
        >>> print(f"Loaded ARIMA({model_info['p']},{model_info['d']},{model_info['q']})")
        >>>
        >>> # Use parameters for prediction
        >>> from ml.training.train import fit_arima
        >>> new_model = fit_arima(
        ...     new_data,
        ...     p=model_info['p'],
        ...     d=model_info['d'],
        ...     q=model_info['q'],
        ...     start_params=model_info['params']
        ... )
        >>> predictions = new_model.predict(n_periods=5)
    """
    # Get parameterized cache path
    model_params_path = get_cache_path(
        'arima_model_params',
        cache_dir=CACHE_DIR,
        extension='pkl',
        p=p, d=d, q=q,
        value_column=value_column
    )

    # Check if cache file exists
    if not model_params_path.exists():
        raise FileNotFoundError(
            f"No cached ARIMA model found at {model_params_path}. "
            f"Please train a model with ARIMA({p},{d},{q}) on '{value_column}' first."
        )

    # Load cached parameters
    with open(model_params_path, 'rb') as f:
        cached_model = pickle.load(f)

    # Validate parameters match
    if cached_model['p'] != p or cached_model['d'] != d or cached_model['q'] != q:
        raise ValueError(
            f"Cached model is ARIMA({cached_model['p']},{cached_model['d']},{cached_model['q']}) "
            f"but requested ARIMA({p},{d},{q})"
        )

    if cached_model['value_column'] != value_column:
        raise ValueError(
            f"Cached model was trained on '{cached_model['value_column']}' "
            f"but requested '{value_column}'"
        )

    return cached_model


def list_cached_models() -> list[dict]:
    """
    List all cached ARIMA models.

    Returns:
        List of dictionaries, each containing model metadata:
            - 'path': Path to the cached model file
            - 'p': AR order
            - 'd': Differencing order
            - 'q': MA order
            - 'value_column': Name of the modeled column

    Example:
        >>> models = list_cached_models()
        >>> for model in models:
        ...     print(f"ARIMA({model['p']},{model['d']},{model['q']}) on {model['value_column']}")
    """
    if not CACHE_DIR.exists():
        return []

    models = []
    for pkl_file in CACHE_DIR.glob('arima_model_params*.pkl'):
        try:
            with open(pkl_file, 'rb') as f:
                cached_model = pickle.load(f)

            models.append({
                'path': pkl_file,
                'p': cached_model.get('p'),
                'd': cached_model.get('d'),
                'q': cached_model.get('q'),
                'value_column': cached_model.get('value_column'),
            })
        except Exception:
            # Skip files that can't be loaded
            continue

    return models
