"""ETA-related functions for vehicle location data."""
import pandas as pd
import numpy as np


def filter_rows_after_stop(
    df: pd.DataFrame,
    segment_column: str,
    stop_column: str
) -> pd.DataFrame:
    """
    Filter out rows that occur after and including the last stop in each segment.

    Rows after and including the last stop cannot have accurate ETAs because there is no
    "next stop" to calculate the ETA to. This function removes those rows
    to ensure all remaining rows have valid ETA targets.

    Note: This function also implicitly removes segments that have no stops at all,
    since all rows in such segments are considered "after the last stop" (which
    doesn't exist).

    Args:
        df: DataFrame with segment and stop data
        segment_column: Name of the column containing segment IDs
        stop_column: Name of the column containing stop names (None/NaN when not at stop)

    Returns:
        DataFrame with:
        - Rows after the last stop in each segment removed
        - Segments without any stops completely removed

    Raises:
        KeyError: If segment_column or stop_column doesn't exist in the dataframe

    Example:
        >>> df = pd.DataFrame({
        ...     'segment_id': [1, 1, 1, 1, 1],
        ...     'stop': [None, 'Stop A', None, 'Stop B', None],
        ...     'value': [10, 20, 30, 40, 50]
        ... })
        >>> result = filter_rows_after_stop(df, 'segment_id', 'stop')
        >>> # Last row is removed (after Stop B)
        >>> result[['segment_id', 'stop', 'value']]
           segment_id     stop  value
        0           1     None     10
        1           1  Stop A     20
        2           1     None     30
        3           1  Stop B     40
    """
    # Validation
    if segment_column not in df.columns:
        raise KeyError(f"Column '{segment_column}' not found in dataframe")
    if stop_column not in df.columns:
        raise KeyError(f"Column '{stop_column}' not found in dataframe")

    # 1. Identify indices of rows where a stop occurs
    # We use the dataframe's index directly
    stops_mask = df[stop_column].notna()

    # 2. Find the max index (last stop) for each segment
    # Group the indices of stops by segment and find the max
    # We use df.index.to_series() to treat the index as values we can aggregate
    last_stop_indices = df.index.to_series()[stops_mask].groupby(df.loc[stops_mask, segment_column]).max()

    # 3. Map the last stop index back to the original dataframe
    # This creates a Series aligned with df, containing the cut-off index for each row's segment
    # Segments with no stops will have NaN, which correctly results in False during comparison
    limit_indices = df[segment_column].map(last_stop_indices)

    # 4. Filter: Keep rows where the current index is <= the last stop index for that segment
    # This vectorizes the "rows after last stop" check
    keep_mask = df.index <= limit_indices

    # Filter dataframe
    filtered_df = df[keep_mask].copy()

    return filtered_df


def add_eta(
    df: pd.DataFrame,
    stop_column: str,
    time_column: str,
    output_column: str
) -> pd.DataFrame:
    """
    Add ETA (estimated time of arrival) column showing time until next stop arrival.

    For each data point, calculates the time (in seconds) until the vehicle arrives
    at the next stop. Properly handles multiple vehicles by grouping by vehicle_id.

    Uses efficient pandas built-in functions for vectorized operations.

    Args:
        df: DataFrame with vehicle location data
        stop_column: Name of the column containing stop names (None/NaN when not at stop)
        time_column: Name of the column containing time values (in seconds, e.g., epoch_seconds)
        output_column: Name of the new column to create with ETA values

    Returns:
        DataFrame with added ETA column (NaN if no next stop exists)

    Raises:
        KeyError: If required columns don't exist in the dataframe

    Example:
        >>> df = pd.DataFrame({
        ...     'vehicle_id': [1, 1, 1, 1, 2, 2, 2],
        ...     'epoch_seconds': [0, 10, 20, 30, 0, 15, 30],
        ...     'stop': [None, None, 'Stop A', None, None, 'Stop B', None]
        ... })
        >>> result = add_eta(df, 'stop', 'epoch_seconds', 'eta_seconds')
        >>> # Row 0 (t=0) has eta=20 (arrives at Stop A at t=20)
        >>> # Row 1 (t=10) has eta=10 (arrives at Stop A at t=20)
        >>> # Row 2 (t=20) is at stop, eta=20 (next stop at t=40)
        >>> # Row 3 (t=30) has eta=10 (arrives at next stop at t=40)
    """
    # Validation
    if 'vehicle_id' not in df.columns:
        raise KeyError("Column 'vehicle_id' not found in dataframe")
    if stop_column not in df.columns:
        raise KeyError(f"Column '{stop_column}' not found in dataframe")
    if time_column not in df.columns:
        raise KeyError(f"Column '{time_column}' not found in dataframe")

    # Work with a copy
    df_work = df.copy()

    # Initialize ETA column with NaN
    df_work[output_column] = np.nan

    # Create a helper column to identify stops
    df_work['_is_at_stop'] = df_work[stop_column].notna()

    def calculate_eta_for_vehicle(group):
        """Calculate ETA for a single vehicle using vectorized operations."""
        # Get indices where vehicle is at stops
        stop_mask = group['_is_at_stop']

        if not stop_mask.any():
            # No stops, return group with NaN ETAs
            return group

        # Get stop times as a sorted array
        stop_times = group.loc[stop_mask, time_column].values
        stop_times_sorted = np.sort(stop_times)

        # For each row, find the next stop time using searchsorted
        current_times = group[time_column].values

        # searchsorted with side='right' gives us the index of the first element > current_time
        next_stop_indices = np.searchsorted(stop_times_sorted, current_times, side='right')

        # Get the next stop times (NaN if no next stop exists)
        next_stop_time_values = np.full(len(group), np.nan)
        valid_mask = next_stop_indices < len(stop_times_sorted)
        next_stop_time_values[valid_mask] = stop_times_sorted[next_stop_indices[valid_mask]]

        # Calculate ETA: next_stop_time - current_time
        group[output_column] = next_stop_time_values - current_times

        return group

    # Apply to each vehicle, segment group
    df_work = df_work.groupby(['vehicle_id', 'segment_id'], group_keys=False).apply(
        calculate_eta_for_vehicle
    )

    # Clean up helper column
    df_work = df_work.drop(columns=['_is_at_stop'])

    return df_work
