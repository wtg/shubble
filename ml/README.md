# ML Pipeline for Shuttle Tracker

Complete machine learning data pipeline for processing vehicle location data, with multi-level caching and train/test splitting that prevents data leakage.

## Overview

This pipeline transforms raw vehicle GPS data into ML-ready datasets by:

1. **Loading** vehicle location data from PostgreSQL
2. **Preprocessing** with feature engineering (routes, speeds, distances)
3. **Segmenting** into consecutive trip segments
4. **Filtering** out short segments (< minimum length)
5. **Splitting** into train/test sets without data leakage

## Quick Start

```python
from ml.pipelines import preprocess_and_split_pipeline

# Run complete pipeline with defaults
train_df, test_df = preprocess_and_split_pipeline(
    force_recompute=False,   # Use cached preprocessing
    force_resplit=False,      # Use cached train/test split
    max_timedelta=600,        # 10 minutes max gap between points
    min_segment_length=10,    # Keep segments with 10+ points
    test_ratio=0.2,           # 20% test data
    random_seed=42            # Reproducibility
)

# Both train_df and test_df have segment_id column preserved
print(f"Training: {train_df['segment_id'].nunique()} segments")
print(f"Testing: {test_df['segment_id'].nunique()} segments")
```

## Architecture

### Three-Level Caching

The pipeline uses multi-level caching for performance:

```
┌─────────────────────────────────────────────────────────┐
│ Level 1: Raw Data                                       │
│ File: ml/data/vehicle_locations.csv                     │
│ Function: load_vehicle_locations()                      │
│ Columns: id, vehicle_id, timestamp, lat, lon, heading   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Level 2: Preprocessed Data                              │
│ File: ml/data/preprocessed_vehicle_locations.csv        │
│ Function: preprocess_pipeline()                         │
│ Adds: epoch_seconds, route, distance_km, speed_kmh,     │
│       dist_to_route, closest_lat, closest_lon           │
└─────────────────────────────────────────────────────────┘
                          ↓
     ┌────────────────────────────────────────┐
     │ segment_by_consecutive()               │
     │ Adds: segment_id column                │
     └────────────────────────────────────────┘
                          ↓
     ┌────────────────────────────────────────┐
     │ filter_segmented() [optional]          │
     │ Removes short segments                 │
     └────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Level 3: Train/Test Split                               │
│ Files: ml/training/train.csv, ml/training/test.csv      │
│ Function: segmented_train_test_split()                  │
│ Preserves: segment_id column                            │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```python
# Step 1: Load raw data (with caching)
from ml.data.load import load_vehicle_locations
df = load_vehicle_locations(force_reload=False)
# Returns: vehicle locations with timestamp, lat, lon, heading

# Step 2: Preprocess (with caching)
from ml.pipelines import preprocess_pipeline
df = preprocess_pipeline(force_recompute=False)
# Adds: epoch_seconds, route, distance_km, speed_kmh, dist_to_route

# Step 3: Segment into trips
from ml.data.preprocess import segment_by_consecutive
df_segmented = segment_by_consecutive(df, max_timedelta=600, segment_column='segment_id')
# Returns: DataFrame with added segment_id column

# Step 4: Train/test split (with caching)
from ml.training.train import segmented_train_test_split, filter_segmented
# Optional: Filter out short segments
df_filtered = filter_segmented(df_segmented, 'segment_id', min_length=10)
# Split into train/test
train_df, test_df = segmented_train_test_split(
    df_filtered,
    timestamp_column='timestamp',
    segment_column='segment_id',
    test_ratio=0.2,
    random_seed=42,
    force_resplit=False
)
# Returns: (train_df, test_df) both with segment_id column

# OR: Use complete pipeline (includes filtering)
from ml.pipelines import preprocess_and_split_pipeline
train_df, test_df = preprocess_and_split_pipeline(
    max_timedelta=600,
    min_segment_length=10,
    test_ratio=0.2,
    random_seed=42
)
```

## Core Functions

### 1. Data Loading (`ml/data/load.py`)

**`load_vehicle_locations(force_reload=False)`**
- Loads vehicle location data from PostgreSQL database
- Caches to `ml/data/vehicle_locations.csv`
- Uses async SQLAlchemy for database queries
- Returns DataFrame with columns: id, vehicle_id, timestamp, latitude, longitude, heading

### 2. Preprocessing Functions (`ml/data/preprocess.py`)

**Individual Preprocessing Functions (all modify in-place):**

**`to_epoch_seconds(df, input_column, output_column)`**
- Converts timestamps to seconds since 2025-01-01
- Uses `EPOCH_2025_OFFSET` for smaller integer values (57x reduction)

**`distance_delta(df, lat_column, lon_column, output_column)`**
- Calculates Haversine distance between consecutive GPS points
- Returns distance in kilometers
- First row is NaN (no previous point)

**`speed(df, distance_column, time_column, output_column)`**
- Calculates speed from distance and time deltas
- Returns speed in km/h
- Handles division by zero (sets to NaN)

**`add_closest_points(df, lat_column, lon_column, output_columns)`**
- Finds nearest route polyline for each GPS point
- Uses `Stops.get_closest_point()` from `shared/stops.py`
- Returns: distance to route, closest lat/lon, route name, polyline index
- Uses pandas `apply()` for performance (~6ms per row)

**`segment_by_consecutive(df, max_timedelta)`**
- Splits DataFrame into list of consecutive trip segments
- Segments break when:
  - Vehicle ID changes
  - Time gap exceeds `max_timedelta` seconds
- Returns list of DataFrames

### 3. Pipeline Functions (`ml/pipelines.py`)

**`preprocess_pipeline(force_recompute=False)`**
- Runs complete preprocessing using pandas `pipe()`
- Caches to `ml/data/preprocessed_vehicle_locations.csv`
- Pipeline order:
  1. Add epoch_seconds
  2. Add route info (closest points)
  3. Add distance deltas
  4. Add speed

**`preprocess_and_split_pipeline(...)`**
- Complete end-to-end pipeline
- Combines preprocessing, segmentation, filtering, and train/test split
- Parameters:
  - `force_recompute`: Rerun preprocessing
  - `force_resplit`: Recompute train/test split
  - `max_timedelta`: Max gap for consecutive points (default 300s)
  - `min_segment_length`: Minimum points to keep a segment (default 3)
  - `test_ratio`: Fraction for test set (default 0.2)
  - `random_seed`: For reproducibility (default None)

### 4. Train/Test Split (`ml/training/train.py`)

**`filter_segmented(df, segment_column, min_length)`**
- Filters out segments with fewer than min_length points
- Returns a new DataFrame (does not modify in place)
- Useful for removing very short trips before training
- Example:
  ```python
  # Keep only segments with 10+ points
  df_filtered = filter_segmented(df, 'segment_id', min_length=10)
  ```

**`segmented_train_test_split(df, timestamp_column, segment_column, test_ratio=0.2, ...)`**
- Splits DataFrame with segment_id into train/test sets
- **Prevents data leakage**: Entire trips go to train OR test, never split
- Algorithm:
  1. Randomly shuffle segment IDs
  2. Greedily add segments to test set until reaching target ratio
  3. Remainder goes to train set
- Caches to `train.csv` and `test.csv` with `segment_id` column preserved
- Returns DataFrames (not lists)
- Parameters:
  - `df`: DataFrame with segment_column
  - `timestamp_column`: Column name for temporal ordering
  - `segment_column`: Column name for segment IDs
  - `test_ratio`: Target fraction for test set
  - `random_seed`: For reproducibility
  - `force_resplit`: Ignore cache and recompute
  - `output_dir`: Where to save train.csv/test.csv

## Features

### Epoch Time Optimization

Uses epoch since 2025-01-01 instead of Unix epoch:
- Unix epoch: `1735862400` (10 digits)
- 2025 epoch: `172800` (6 digits)
- **57x smaller integers** → better memory efficiency

```python
EPOCH_2025_OFFSET = 1735689600  # 2025-01-01 00:00:00 UTC
```

### Route Matching

Uses Haversine distance to find closest route polyline:
- Vectorized calculations for performance
- Returns route name if unambiguous
- Returns None if multiple routes too close (ambiguous)
- Integrated with `shared/stops.py` route definitions

### Data Leakage Prevention

Train/test split operates on **entire trip segments**:
- A trip is all consecutive points from the same vehicle
- Each trip goes entirely to train OR entirely to test
- No mixing of points from the same trip across sets
- Critical for time-series prediction tasks

### Segment Preservation

When saving to CSV, adds `segment_id` column:
- Tracks which segment each row belongs to
- Preserves temporal ordering within segments
- Can reconstruct original segments when loading from cache

```python
# Saved CSV has segment_id
train_df.to_csv('train.csv')  # Contains segment_id column

# Loading reconstructs segments
train_segments = [
    train_df[train_df['segment_id'] == seg_id].drop(columns=['segment_id'])
    for seg_id in train_df['segment_id'].unique()
]
```

## Column Reference

### Raw Data (after `load_vehicle_locations`)
- `id`: Primary key
- `vehicle_id`: Vehicle identifier
- `timestamp`: Datetime of GPS reading
- `latitude`: GPS latitude
- `longitude`: GPS longitude
- `heading`: Vehicle heading in degrees

### Preprocessed Data (after `preprocess_pipeline`)
All above columns plus:
- `epoch_seconds`: Seconds since 2025-01-01
- `route`: Matched route name (or None if ambiguous)
- `closest_lat`: Latitude of closest polyline point
- `closest_lon`: Longitude of closest polyline point
- `polyline_idx`: Index in route polyline
- `dist_to_route`: Distance to route in km
- `distance_km`: Distance traveled since last point
- `speed_kmh`: Speed in km/h

### Split Data (after `segmented_train_test_split`)
All preprocessed columns plus:
- `segment_id`: Trip segment identifier

## Performance

### Benchmarks (approximate)

- **Raw data load**: ~2-5 seconds (1.8M rows from database)
- **Preprocessing**: ~3 hours first run, <1 second cached
  - Bottleneck: Route matching at ~6ms per row
- **Segmentation**: ~1-2 seconds (1.8M rows)
- **Train/test split**: ~0.5 seconds first run, <0.1 second cached

### Optimization Tips

1. **Always use caching**: Set `force_recompute=False` and `force_resplit=False`
2. **Route matching is expensive**: Only reprocess when routes change
3. **Work with segments**: More efficient than full DataFrame for sequence models
4. **Vectorized operations**: All functions use numpy/pandas vectorization

## Examples

### Example 1: Basic Usage

```python
from ml.pipelines import preprocess_and_split_pipeline
import pandas as pd

# Run complete pipeline
train_df, test_df = preprocess_and_split_pipeline(
    max_timedelta=600,       # 10 minutes
    min_segment_length=10,   # Keep segments with 10+ points
    test_ratio=0.2,
    random_seed=42
)

# Analyze first 5 segments from training set
for segment_id in train_df['segment_id'].unique()[:5]:
    segment = train_df[train_df['segment_id'] == segment_id]

    print(f"Segment {segment_id}:")
    print(f"  Vehicle: {segment['vehicle_id'].iloc[0]}")
    print(f"  Route: {segment['route'].mode()[0] if not segment['route'].isna().all() else 'Unknown'}")
    print(f"  Points: {len(segment)}")
    print(f"  Duration: {segment['epoch_seconds'].iloc[-1] - segment['epoch_seconds'].iloc[0]:.0f}s")
    print()
```

### Example 2: Custom Preprocessing

```python
from ml.data.load import load_vehicle_locations
from ml.data.preprocess import (
    to_epoch_seconds,
    add_closest_points,
    distance_delta,
    speed
)

# Load and preprocess manually
df = load_vehicle_locations()

# Add custom features
to_epoch_seconds(df, 'timestamp', 'epoch_seconds')
add_closest_points(df, 'latitude', 'longitude', {
    'distance': 'dist_to_route',
    'route_name': 'route'
})
distance_delta(df, 'latitude', 'longitude', 'distance_km')
speed(df, 'distance_km', 'epoch_seconds', 'speed_kmh')

# Custom feature: acceleration
df['acceleration'] = df.groupby('vehicle_id')['speed_kmh'].diff() / df.groupby('vehicle_id')['epoch_seconds'].diff()
```

### Example 3: Filtering Segments

```python
from ml.pipelines import preprocess_pipeline
from ml.data.preprocess import segment_by_consecutive
from ml.training.train import filter_segmented

# Load and preprocess
df = preprocess_pipeline()

# Add segment IDs
df_segmented = segment_by_consecutive(df, max_timedelta=600, segment_column='segment_id')

print(f"Total segments: {df_segmented['segment_id'].nunique()}")
print(f"Total points: {len(df_segmented)}")

# Filter to keep only segments with 20+ points
df_filtered = filter_segmented(df_segmented, 'segment_id', min_length=20)

print(f"\nAfter filtering (min_length=20):")
print(f"  Remaining segments: {df_filtered['segment_id'].nunique()}")
print(f"  Remaining points: {len(df_filtered)}")

# Analyze segment sizes
segment_sizes = df_filtered.groupby('segment_id').size()
print(f"\nSegment size stats:")
print(f"  Min: {segment_sizes.min()}")
print(f"  Mean: {segment_sizes.mean():.1f}")
print(f"  Max: {segment_sizes.max()}")
```

### Example 4: Working with Segmented DataFrames

```python
from ml.pipelines import preprocess_pipeline
from ml.data.preprocess import segment_by_consecutive
from ml.training.train import segmented_train_test_split
import pandas as pd

# Get segmented data
df = preprocess_pipeline()
df_segmented = segment_by_consecutive(df, max_timedelta=600, segment_column='segment_id')

# Split into train/test
train_df, test_df = segmented_train_test_split(
    df_segmented,
    timestamp_column='timestamp',
    segment_column='segment_id',
    test_ratio=0.2,
    random_seed=42
)

# Work with individual segments from train set
for segment_id in train_df['segment_id'].unique()[:5]:  # First 5 segments
    segment = train_df[train_df['segment_id'] == segment_id]

    print(f"Segment {segment_id}:")
    print(f"  Vehicle: {segment['vehicle_id'].iloc[0]}")
    print(f"  Route: {segment['route'].mode()[0] if not segment['route'].isna().all() else 'Unknown'}")
    print(f"  Points: {len(segment)}")
    print(f"  Duration: {segment['epoch_seconds'].iloc[-1] - segment['epoch_seconds'].iloc[0]:.0f}s")
    print()
```

### Example 5: Force Reprocessing

```python
from ml.pipelines import preprocess_and_split_pipeline

# Force complete reprocessing (e.g., after route changes or data updates)
train_df, test_df = preprocess_and_split_pipeline(
    force_recompute=True,    # Rerun preprocessing
    force_resplit=True,       # Recompute train/test split
    max_timedelta=600,
    min_segment_length=10,
    test_ratio=0.2,
    random_seed=42
)
```

## Testing

Run the complete pipeline:

```bash
cd /Users/joel/eclipse-workspace/shuttletracker-new
python -m ml.pipelines
```

Run individual preprocessing functions demo:

```bash
python -m ml.data.preprocess
```

## File Structure

```
ml/
├── README.md                           # This file
├── requirements.txt                    # ML dependencies
├── pipelines.py                        # Complete preprocessing pipelines
│
├── data/
│   ├── __init__.py
│   ├── load.py                         # Database loading + caching
│   ├── preprocess.py                   # Individual preprocessing functions
│   ├── vehicle_locations.csv           # Raw data cache
│   └── preprocessed_vehicle_locations.csv  # Preprocessed cache
│
├── training/
│   ├── __init__.py
│   ├── train.py                        # Train/test split utilities
│   ├── train.csv                       # Training set cache
│   └── test.csv                        # Test set cache
│
├── models/                             # Model implementations
│   └── __init__.py
│
├── evaluation/                         # Evaluation utilities (future)
└── scripts/                            # Existing analysis scripts
```

## Dependencies

Install ML dependencies:

```bash
pip install -r ml/requirements.txt
```

Required packages:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scipy` - Scientific computing
- `scikit-learn` - ML utilities
- `SQLAlchemy[asyncio]` - Database ORM
- `asyncpg` - Async PostgreSQL driver
- `matplotlib`, `seaborn` - Visualization
- `statsmodels` - Statistical models

## Notes

### Timezone Handling
- Database stores timestamps in UTC
- `epoch_seconds` is calculated from UTC time
- Use `backend/time_utils.py` for timezone conversions

### Memory Considerations
- Full dataset (~1.8M rows) uses ~500MB RAM when loaded
- Segments use less memory (only active trip in memory)
- Cached CSVs are ~200MB total

### Data Quality
- First point in each segment has NaN distance and speed
- Points with ambiguous routes have None for route name
- Very short segments (< 3 points) are filtered out

## Models

### ARIMA Model (`ml/models/arima.py`)

Wrapper around `statsmodels.tsa.arima.model.ARIMA` for segment-based time series forecasting.

**Features:**
- Supports ARIMA(p,d,q) specification
- Fits separate models per trip segment
- Validates stationarity and handles fitting errors

## Future Enhancements

Potential improvements:
- [ ] Add data validation and quality checks
- [ ] Implement feature engineering for stop detection
- [ ] Add data augmentation for training
- [ ] Create visualization utilities for segments
- [ ] Add more time series models (SARIMA, state space with exogenous variables)
- [ ] Add evaluation metrics in `ml/evaluation/`
- [ ] Support for real-time inference pipeline
