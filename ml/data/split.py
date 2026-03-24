"""Split functions for vehicle location data."""
import pandas as pd


def split_by_route_polyline_index(
    df: pd.DataFrame,
    route_column: str = 'route',
    polyline_index_column: str = 'polyline_idx'
) -> dict[tuple[str, int], pd.DataFrame]:
    """
    Split a dataframe into multiple dataframes, one for each unique (route, polyline_index).

    Args:
        df: DataFrame with route and polyline index data
        route_column: Name of the column containing route names
        polyline_index_column: Name of the column containing polyline indices

    Returns:
        Dictionary mapping (route, polyline_index) tuples to their corresponding DataFrames

    Raises:
        KeyError: If required columns don't exist in the dataframe
    """
    # Validation
    if route_column not in df.columns:
        raise KeyError(f"Column '{route_column}' not found in dataframe")
    if polyline_index_column not in df.columns:
        raise KeyError(f"Column '{polyline_index_column}' not found in dataframe")

    # Group by route and polyline index
    grouped = df.groupby([route_column, polyline_index_column])

    # Create a dictionary of dataframes
    split_dfs = {name: group.copy() for name, group in grouped}

    return split_dfs
