"""
Cache Utilities for ML Pipelines and Deployment.
"""
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

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


def get_cache_path(base_name: str, cache_dir: Path = SHARED_CACHE_DIR, extension: str = 'csv', **params) -> Path:
    """
    Get cache file path with parameters encoded in the filename.

    Args:
        base_name: Base filename (e.g., 'locations_segmented', 'stops_preprocessed')
        cache_dir: Directory to store cache file (default: SHARED_CACHE_DIR)
        extension: File extension without dot (default: 'csv')
        **params: Parameters to include in filename (will be sorted alphabetically)

    Returns:
        Path to cache file with parameters

    Example:
        >>> get_cache_path('locations_segmented', max_timedelta=15, max_distance=0.005)
        Path('ml/cache/shared/locations_segmented_max_distance0.005_max_timedelta15.csv')
        >>> get_cache_path('arima_best_params', ARIMA_CACHE_DIR, 'pkl', p=3, d=0, q=2)
        Path('ml/cache/arima/arima_best_params_d0_p3_q2.pkl')
    """
    if not params:
        return cache_dir / f"{base_name}.{extension}"

    # Build param string from sorted params (for consistency)
    # Format numbers to avoid excessive decimal places
    param_parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            # Format floats to remove trailing zeros
            v_str = f"{v:g}"
        else:
            v_str = str(v)
        # Sanitize the value string for filename safety
        v_str = v_str.replace('_', '').replace('.', 'p')
        param_parts.append(f"{k}{v_str}")

    param_str = "_".join(param_parts)
    return cache_dir / f"{base_name}_{param_str}.{extension}"


def load_cached_csv(path: Path, description: str) -> Optional[pd.DataFrame]:
    """Load a cached CSV file with timestamp parsing."""
    if not path.exists():
        return None

    logger.info(f"Loading {description} from {path}")
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    logger.info(f"Loaded {len(df)} records from cache")
    return df


def save_csv(df: pd.DataFrame, path: Path, description: str):
    """Save a DataFrame to CSV."""
    logger.info(f"Saving {description} to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} records")
