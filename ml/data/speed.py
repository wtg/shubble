"""Speed calculation functions for vehicle location data."""
import pandas as pd
import numpy as np


def distance_delta(
    df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    output_column: str
) -> pd.DataFrame:
    """
    Compute the distance traveled between consecutive GPS points.

    For each point, computes the great-circle distance from the previous point
    using the Haversine formula. The first row will have NaN since there's no
    previous point.

    Modifies the dataframe in place by adding a new column with distances in kilometers.

    Args:
        df: Pandas DataFrame to modify
        lat_column: Name of the latitude column
        lon_column: Name of the longitude column
        output_column: Name of the new column to create (default: 'distance_delta')

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # Import here to avoid circular imports
    from shared.stops import haversine_vectorized

    # Validation
    if lat_column not in df.columns:
        raise KeyError(f"Column '{lat_column}' not found in dataframe")
    if lon_column not in df.columns:
        raise KeyError(f"Column '{lon_column}' not found in dataframe")

    if len(df) == 0:
        df[output_column] = []
        return df

    # Create coordinate arrays: [lat, lon]
    coords = df[[lat_column, lon_column]].values

    # Shift coordinates to get previous position
    coords_prev = np.roll(coords, 1, axis=0)

    # Compute distances between consecutive points
    distances = haversine_vectorized(coords_prev, coords)

    # Set first distance to NaN (no previous point)
    distances[0] = np.nan

    # Add to dataframe
    df[output_column] = distances
    return df


def speed(
    df: pd.DataFrame,
    distance_column: str,
    time_column: str,
    output_column: str,
) -> pd.DataFrame:
    """
    Compute speed from distance and time deltas, with outlier filtering.

    Calculates speed by dividing distance by the time difference between consecutive
    points.

    Speed = distance / time_delta

    Modifies the dataframe in place by adding a new column with speed values.

    Args:
        df: Pandas DataFrame to modify
        distance_column: Name of the column containing distances (e.g., from distance_delta)
        time_column: Name of the column containing time values in seconds
        output_column: Name of the new column to create with speed values

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # Validation
    if distance_column not in df.columns:
        raise KeyError(f"Column '{distance_column}' not found in dataframe")
    if time_column not in df.columns:
        raise KeyError(f"Column '{time_column}' not found in dataframe")

    if len(df) == 0:
        df[output_column] = []
        return df

    # Get time values and compute deltas
    time_values = df[time_column].values
    time_deltas = np.diff(time_values, prepend=np.nan)

    # Get distance values
    distances = df[distance_column].values

    # Compute speed: distance / time_delta
    # Handle division by zero by setting to NaN where time_delta is 0
    with np.errstate(divide='ignore', invalid='ignore'):
        speeds = distances / time_deltas
        # Convert km/s to km/h by multiplying by 3600
        speeds = speeds * 3600

    # Set speed to NaN where time_delta is zero or negative
    speeds = np.where((time_deltas <= 0) | np.isnan(time_deltas), np.nan, speeds)

    # Add to dataframe
    df[output_column] = speeds
    return df
