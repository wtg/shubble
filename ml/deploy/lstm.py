"""LSTM model deployment utilities."""
from pathlib import Path
from typing import Optional

from ml.cache import LSTM_CACHE_DIR

# Get cache directory
CACHE_DIR = LSTM_CACHE_DIR


def load_lstm(
    input_columns: list[str] = ['latitude', 'longitude', 'speed_kmh'],
    output_columns: list[str] = ['eta_seconds'],
    hidden_size: int = 50,
    num_layers: int = 2,
    dropout: float = 0.1
) -> dict[tuple[str, int], 'LSTMModel']:
    """
    Load all trained LSTM models from cache.

    This function scans the LSTM cache directory and loads all trained models
    for each route polyline segment. Each model is stored in a subdirectory
    named <route>_<polyline_idx>/ containing a model.pth file.

    The returned dictionary maps (route_name, polyline_idx) tuples to loaded
    LSTMModel instances ready for making predictions.

    Args:
        input_columns: Feature columns used during training (default: ['latitude', 'longitude', 'speed_kmh'])
        output_columns: Target columns predicted by the model (default: ['eta_seconds'])
        hidden_size: LSTM hidden size (default: 50)
        num_layers: Number of LSTM layers (default: 2)
        dropout: LSTM dropout rate (default: 0.1)

    Returns:
        Dictionary mapping (route_name, polyline_idx) to LSTMModel instances

    Raises:
        FileNotFoundError: If LSTM cache directory doesn't exist
        ValueError: If no models are found in the cache directory

    Example:
        >>> # Load all trained LSTM models
        >>> models = load_lstm()
        >>> print(f"Loaded {len(models)} models")
        >>>
        >>> # Make prediction for a specific route
        >>> model = models[('East Route', 0)]
        >>> prediction = model.predict(input_sequence)
        >>>
        >>> # List all available routes
        >>> for (route, idx) in models.keys():
        ...     print(f"{route} - Segment {idx}")
    """
    from ml.models.lstm import LSTMModel

    if not CACHE_DIR.exists():
        raise FileNotFoundError(
            f"LSTM cache directory not found at {CACHE_DIR}. "
            "Please train LSTM models first using the lstm pipeline."
        )

    models = {}
    input_size = len(input_columns)
    output_size = len(output_columns)

    # Scan cache directory for polyline subdirectories
    for polyline_dir in sorted(CACHE_DIR.iterdir()):
        if not polyline_dir.is_dir():
            continue

        model_path = polyline_dir / "model.pth"
        if not model_path.exists():
            # Skip directories without model.pth
            continue

        # Parse directory name: <route>_<polyline_idx>
        dir_name = polyline_dir.name
        parts = dir_name.rsplit('_', 1)

        if len(parts) != 2:
            # Skip directories that don't match the expected format
            continue

        route_name_safe, polyline_idx_str = parts

        # Convert safe route name back (undo replacement of spaces and slashes)
        route_name = route_name_safe.replace('_', ' ')

        try:
            polyline_idx = int(polyline_idx_str)
        except ValueError:
            # Skip if polyline index is not an integer
            continue

        # Create model instance with same architecture as training
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )

        # Load trained weights
        try:
            model.load(model_path)
            models[(route_name, polyline_idx)] = model
        except Exception as e:
            # Log error but continue loading other models
            print(f"Warning: Failed to load model for {route_name} segment {polyline_idx}: {e}")
            continue

    if not models:
        raise ValueError(
            f"No LSTM models found in {CACHE_DIR}. "
            "Please train models first using the lstm pipeline."
        )

    return models


def load_lstm_for_route(
    route_name: str,
    polyline_idx: int,
    input_columns: list[str] = ['latitude', 'longitude', 'speed_kmh'],
    output_columns: list[str] = ['eta_seconds'],
    hidden_size: int = 50,
    num_layers: int = 2,
    dropout: float = 0.1
) -> 'LSTMModel':
    """
    Load a single LSTM model for a specific route and polyline segment.

    Args:
        route_name: Name of the route (e.g., 'East Route')
        polyline_idx: Index of the polyline segment
        input_columns: Feature columns used during training (default: ['latitude', 'longitude', 'speed_kmh'])
        output_columns: Target columns predicted by the model (default: ['eta_seconds'])
        hidden_size: LSTM hidden size (default: 50)
        num_layers: Number of LSTM layers (default: 2)
        dropout: LSTM dropout rate (default: 0.1)

    Returns:
        Loaded LSTMModel instance

    Raises:
        FileNotFoundError: If model file doesn't exist for the specified route/segment

    Example:
        >>> # Load model for specific route segment
        >>> model = load_lstm_for_route('East Route', 0)
        >>> prediction = model.predict(input_sequence)
    """
    from ml.models.lstm import LSTMModel

    # Convert route name to safe directory name
    safe_route = route_name.replace(' ', '_').replace('/', '_')
    polyline_dir = CACHE_DIR / f"{safe_route}_{polyline_idx}"
    model_path = polyline_dir / "model.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found for route '{route_name}' segment {polyline_idx} at {model_path}. "
            "Please train this model first using the lstm pipeline."
        )

    # Create model instance
    input_size = len(input_columns)
    output_size = len(output_columns)
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout
    )

    # Load trained weights
    model.load(model_path)
    return model


def list_cached_models() -> list[dict]:
    """
    List all cached LSTM models with metadata.

    Returns:
        List of dictionaries, each containing model metadata:
            - 'route_name': Name of the route
            - 'polyline_idx': Polyline segment index
            - 'path': Path to the model file
            - 'dir': Path to the polyline directory

    Example:
        >>> models = list_cached_models()
        >>> for model in models:
        ...     print(f"{model['route_name']} - Segment {model['polyline_idx']}")
    """
    if not CACHE_DIR.exists():
        return []

    models = []
    for polyline_dir in sorted(CACHE_DIR.iterdir()):
        if not polyline_dir.is_dir():
            continue

        model_path = polyline_dir / "model.pth"
        if not model_path.exists():
            continue

        # Parse directory name
        dir_name = polyline_dir.name
        parts = dir_name.rsplit('_', 1)

        if len(parts) != 2:
            continue

        route_name_safe, polyline_idx_str = parts
        route_name = route_name_safe.replace('_', ' ')

        try:
            polyline_idx = int(polyline_idx_str)
        except ValueError:
            continue

        models.append({
            'route_name': route_name,
            'polyline_idx': polyline_idx,
            'path': model_path,
            'dir': polyline_dir
        })

    return models
