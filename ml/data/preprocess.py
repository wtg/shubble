"""Basic preprocessing functions for vehicle location data."""
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()

# Epoch offset: 2025-01-01 00:00:00 UTC in Unix seconds
EPOCH_2025_OFFSET = int(datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())


def to_epoch_seconds(
    df: pd.DataFrame,
    input_column: str,
    output_column: str
) -> None:
    """
    Convert a datetime column to epoch seconds since 2025-01-01 00:00:00 UTC.

    Modifies the dataframe in place by adding a new column with epoch timestamps
    relative to the start of 2025. This produces smaller numbers that are easier
    to work with in machine learning models.

    Args:
        df: Pandas DataFrame to modify
        input_column: Name of the datetime column to convert
        output_column: Name of the new column to create with epoch seconds

    Raises:
        KeyError: If input_column doesn't exist in the dataframe
        TypeError: If input_column is not a datetime type

    Example:
        >>> df = pd.DataFrame({'timestamp': pd.to_datetime(['2025-01-01', '2025-01-02'])})
        >>> to_epoch_seconds(df, 'timestamp', 'epoch')
        >>> print(df)
                  timestamp     epoch
        0 2025-01-01            0.0
        1 2025-01-02        86400.0
    """
    if input_column not in df.columns:
        raise KeyError(f"Column '{input_column}' not found in dataframe")

    if not pd.api.types.is_datetime64_any_dtype(df[input_column]):
        raise TypeError(
            f"Column '{input_column}' must be datetime type, "
            f"got {df[input_column].dtype}"
        )

    # Convert datetime to Unix epoch seconds, then subtract 2025 offset
    df[output_column] = (df[input_column].astype('int64') / 1e9) - EPOCH_2025_OFFSET


def add_closest_points(
    df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    output_columns: dict[str, str],
    additive: bool = False
) -> None:
    """
    Add route information by finding the closest point on route polylines.

    For each GPS coordinate, uses the Stops.get_closest_point() function to find
    the nearest route polyline and adds the requested information as new columns.

    The get_closest_point function returns:
    - distance: Distance to the closest point on any route (km)
    - closest_point: The closest point coordinates [lat, lon]
    - route_name: Name of the closest route
    - polyline_index: Index of the polyline within that route
    - segment_index: Index of the segment within the polyline

    Modifies the dataframe in place by adding new columns based on output_columns mapping.

    Args:
        df: Pandas DataFrame to modify
        lat_column: Name of the latitude column
        lon_column: Name of the longitude column
        output_columns: Dictionary mapping return value names to output column names.
                       Valid keys: 'distance', 'closest_point_lat', 'closest_point_lon',
                                  'route_name', 'polyline_index'
                       Only specified keys will be added as columns.
        additive: If True, only process rows where output columns have NaN values.
                 Useful for incremental updates where some rows already have closest points.
                 (default: False)

    Raises:
        KeyError: If lat_column or lon_column doesn't exist in the dataframe

    Example:
        >>> df = pd.DataFrame({
        ...     'latitude': [42.7284, 42.7295],
        ...     'longitude': [-73.6788, -73.6799]
        ... })
        >>> add_closest_points(df, 'latitude', 'longitude', {
        ...     'distance': 'dist_to_route',
        ...     'route_name': 'route',
        ...     'closest_point_lat': 'closest_lat'
        ... })
        >>> # df now has columns: dist_to_route, route, closest_lat
    """
    # Import here to avoid circular imports
    from shared.stops import Stops

    # Validation
    if lat_column not in df.columns:
        raise KeyError(f"Column '{lat_column}' not found in dataframe")
    if lon_column not in df.columns:
        raise KeyError(f"Column '{lon_column}' not found in dataframe")

    if not output_columns:
        return  # Nothing to do

    # If additive mode, initialize columns if they don't exist and determine which rows need processing
    if additive:
        # Initialize output columns with NaN if they don't exist
        for output_col in output_columns.values():
            if output_col not in df.columns:
                df[output_col] = np.nan

        # Determine which rows need processing (any row with NaN in any output column)
        route_col = output_columns['route_name']
        rows_to_process_mask = df[route_col].isna()
        rows_to_process = df[rows_to_process_mask].copy()

        if len(rows_to_process) == 0:
            print(f'Additive mode: All {len(df)} rows already have closest points. Skipping.')
            return

        print(f'Additive mode: Processing {len(rows_to_process)}/{len(df)} rows with missing closest points')
    else:
        rows_to_process = df
        rows_to_process_mask = pd.Series([True] * len(df), index=df.index)

    def process_row(row):
        """Process a single row and return closest point information."""
        lat = row[lat_column]
        lon = row[lon_column]

        # Return NaN/None for all outputs if coordinates are invalid
        if pd.isna(lat) or pd.isna(lon):
            result = {}
            for key, output_col in output_columns.items():
                if key == 'route_name':
                    result[output_col] = None
                else:
                    result[output_col] = np.nan
            return pd.Series(result)

        # Get closest point information
        distance, closest_point, route_name, polyline_index, segment_index = Stops.get_closest_point((lat, lon))

        # Build result dictionary with requested values
        value_map = {
            'distance': distance,
            'closest_point_lat': closest_point[0] if closest_point is not None else np.nan,
            'closest_point_lon': closest_point[1] if closest_point is not None else np.nan,
            'route_name': route_name,
            'polyline_index': polyline_index,
            'segment_index': segment_index
        }
        result = {output_col: value_map[key] for key, output_col in output_columns.items()}

        return pd.Series(result)

    # Apply the function to rows that need processing
    result_df = rows_to_process.progress_apply(process_row, axis=1)
    print('Done! Adding columns to dataframe.')

    # Assign results back to the original dataframe for the processed rows
    for output_col in output_columns.values():
        df.loc[rows_to_process_mask, output_col] = result_df[output_col].values
