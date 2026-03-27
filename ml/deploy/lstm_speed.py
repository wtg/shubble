"""Velocity LSTM model deployment utilities."""
import logging
from pathlib import Path
from typing import Optional

from ml.cache import LSTM_VELOCITY_CACHE_DIR, get_polyline_dir_velocity
from ml.models.lstm import LSTMModel

logger = logging.getLogger(__name__)

# Get cache directory
CACHE_DIR = LSTM_VELOCITY_CACHE_DIR


def load_lstm_velocity(
    input_columns: list[str] = ['latitude', 'longitude', 'speed_kmh'],
    output_columns: list[str] = ['next_speed_kmh'],
    hidden_size: int = 50,
    num_layers: int = 2,
    dropout: float = 0.1
) -> dict[tuple[str, int], LSTMModel]:
    """
    """
    from ml.models.lstm import LSTMModel

    return None