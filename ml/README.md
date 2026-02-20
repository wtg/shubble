# ML Pipelines for Shuttle Tracker

Complete machine learning pipelines for processing vehicle location data and training predictive models.

## Overview

This package provides end-to-end ML pipelines for:

1. **Data Processing** - Load, preprocess, segment, and split vehicle GPS data
2. **Model Training** - Train ARIMA (speed) and LSTM (ETA) models
3. **Model Deployment** - Load trained models for inference
4. **Evaluation** - Assess model performance on test data

## Usage

```bash
# Run complete ARIMA pipeline
uv run python -m ml.pipelines arima --segment --p 3 --d 0 --q 2

# Run complete LSTM pipeline
uv run python -m ml.pipelines lstm --stops --train --epochs 20

# Load trained models for inference
uv run python -c "
from ml.deploy import load_arima, load_lstm
arima_model = load_arima(p=3, d=0, q=2)
lstm_models = load_lstm()
"
```

## Pipeline Hierarchy

Pipelines use an automatic hierarchy system where triggering an upstream stage automatically triggers all downstream stages:

```
load → preprocess → segment → stops → split → train → fit
```

### Examples

```bash
# --segment triggers: segment + stops + split + train + fit
uv run python -m ml.pipelines arima --segment

# --preprocess triggers: preprocess + segment + stops + split + train + fit
uv run python -m ml.pipelines lstm --preprocess

# --load triggers all stages
uv run python -m ml.pipelines arima --load
```

## Architecture

### Pipeline Stages

1. **Load** (`load_pipeline`)
   - Fetches vehicle locations from PostgreSQL
   - Caches to `ml/cache/shared/locations_raw.csv`

2. **Preprocess** (`preprocess_pipeline`)
   - Adds epoch seconds, route matching, closest points
   - Caches to `ml/cache/shared/locations_preprocessed.csv`

3. **Segment** (`segment_pipeline`)
   - Segments into consecutive trips
   - Cleans route data, calculates speeds
   - Filters short segments
   - Caches to parameterized path based on settings

4. **Stops** (`stops_pipeline`)
   - Detects stop locations
   - Calculates ETAs to next stop
   - Adds polyline distance metrics
   - Filters rows after last stop
   - Caches to parameterized path

5. **Split** (`split_pipeline`)
   - Splits into train/test sets by segment
   - Prevents data leakage
   - Caches to `ml/cache/arima/` or `ml/cache/lstm/`

6. **Train** (LSTM) or **Fit** (ARIMA)
   - Trains models and saves parameters
   - LSTM: Per-polyline models in `ml/cache/lstm/<route>_<idx>/`
   - ARIMA: Global parameters in `ml/cache/arima/`

### Parameterized Caching

Cache files include parameters in their filenames to prevent using cached data with wrong settings:

```
locations_segmented_max_distance0p005_max_timedelta15_min_segment_length3_window_size5.csv
stops_preprocessed_max_distance0p005_max_timedelta30_min_segment_length3_window_size5.csv
arima_train_random_seed42_test_ratio0p2.csv
```

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│ PostgreSQL Database                                      │
│ - vehicle_locations table                                │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│ Load Pipeline                                            │
│ - Async SQLAlchemy queries                               │
│ - Cache: locations_raw.csv                               │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│ Preprocess Pipeline                                      │
│ - Add epoch_seconds, route matching, closest points      │
│ - Cache: locations_preprocessed.csv                      │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│ Segment Pipeline                                         │
│ - Segment by time/distance gaps                          │
│ - Clean NaN routes with windowing                        │
│ - Calculate speeds                                        │
│ - Filter short segments                                   │
│ - Cache: locations_segmented_<params>.csv                │
└─────────────────────┬───────────────────────────────────┘
                      ↓
             ┌────────┴────────┐
             ↓                 ↓
┌──────────────────────┐   ┌──────────────────────┐
│ Stops Pipeline       │   │ Split Pipeline       │
│ (for LSTM)           │   │ (for ARIMA)          │
│                      │   │                      │
│ - Detect stops       │   │ - Train/test split   │
│ - Calculate ETAs     │   │ - By segment         │
│ - Polyline distances │   │ - Cache: train.csv   │
│ - Filter after stop  │   │         test.csv     │
│ - Cache: stops_*.csv │   │                      │
└──────────┬───────────┘   └──────────┬───────────┘
           ↓                          ↓
┌──────────────────────┐   ┌──────────────────────┐
│ LSTM Training        │   │ ARIMA Training       │
│ - Per-polyline       │   │ - Per-segment        │
│ - Cache: model.pth   │   │ - Cache: params.pkl  │
└──────────────────────┘   └──────────────────────┘
```

## Core Modules

### `ml/cache.py` - Cache Utilities

```python
from ml.cache import (
    get_cache_path,      # Generate parameterized cache paths
    load_cached_csv,     # Load cached CSV with timestamp parsing
    save_csv,            # Save CSV with logging
    SHARED_CACHE_DIR,    # ml/cache/shared/
    ARIMA_CACHE_DIR,     # ml/cache/arima/
    LSTM_CACHE_DIR,      # ml/cache/lstm/
)

# Example: Get parameterized path
path = get_cache_path(
    'locations_segmented',
    max_timedelta=15,
    max_distance=0.005,
    min_segment_length=3
)
# Returns: ml/cache/shared/locations_segmented_max_distance0p005_max_timedelta15_min_segment_length3.csv
```

### `ml/pipelines.py` - Pipeline Functions

```python
from ml.pipelines import (
    load_pipeline,           # Load from database
    preprocess_pipeline,     # Add features
    segment_pipeline,        # Segment trips
    stops_pipeline,          # Add stops/ETAs/distances
    split_pipeline,          # Train/test split
    arima_pipeline,          # Complete ARIMA workflow
    lstm_pipeline,           # Complete LSTM workflow
    apply_pipeline_hierarchy # Apply automatic hierarchy
)

# Run individual stage
df = segment_pipeline(
    max_timedelta=15,
    max_distance=0.005,
    min_segment_length=3,
    window_size=5
)

# Run complete pipeline
results = arima_pipeline(
    p=3, d=0, q=2,
    segment=True,  # Triggers: segment + split + fit
    test_ratio=0.2,
    random_seed=42
)
```

### `ml/deploy/` - Model Deployment

```python
from ml.deploy import (
    load_arima,              # Load ARIMA parameters
    load_lstm,               # Load all LSTM models
    load_lstm_for_route,     # Load single LSTM model
    list_arima_models,       # List available ARIMA models
    list_lstm_models,        # List available LSTM models
)

# Load ARIMA model parameters
arima_params = load_arima(p=3, d=0, q=2, value_column='speed_kmh')
# Returns: {'p': 3, 'd': 0, 'q': 2, 'params': array([...]), 'value_column': 'speed_kmh'}

# Use for warm-start prediction
from ml.training.train import fit_arima
model = fit_arima(new_data, p=3, d=0, q=2, start_params=arima_params['params'])

# Load all LSTM models
lstm_models = load_lstm()
# Returns: {('East Route', 0): LSTMModel, ('West Route', 1): LSTMModel, ...}

# Make prediction
model = lstm_models[('East Route', 0)]
prediction = model.predict(input_sequence)
```

### `ml/data/` - Preprocessing Functions

Functions are organized by pipeline stage. All functions modify DataFrames in-place:

```python
# Basic preprocessing
from ml.data.preprocess import to_epoch_seconds, add_closest_points

# Speed calculations
from ml.data.speed import distance_delta, speed

# Segmentation
from ml.data.segment import (
    segment_by_consecutive,
    filter_segments_by_length,
    clean_closest_route,
    add_closest_points_educated
)

# Stop detection
from ml.data.stops import add_stops, add_polyline_distances

# ETA calculations
from ml.data.eta import filter_rows_after_stop, add_eta

# Data splitting
from ml.data.split import split_by_route_polyline_index

# Example: Manual preprocessing
df = load_pipeline()
to_epoch_seconds(df, 'timestamp', 'epoch_seconds')
add_closest_points(df, 'latitude', 'longitude', {
    'distance': 'dist_to_route',
    'route_name': 'route',
    'closest_point_lat': 'closest_lat',
    'closest_point_lon': 'closest_lon',
    'polyline_index': 'polyline_idx',
    'segment_index': 'segment_idx'
})
```

### `ml/models/` - Model Implementations

```python
from ml.models.lstm import LSTMModel
from ml.training.train import fit_arima

# LSTM
model = LSTMModel(
    input_size=3,
    hidden_size=50,
    num_layers=2,
    output_size=1,
    dropout=0.1
)
model.train_model(X_train, y_train, epochs=20)
model.save('model.pth')

# ARIMA
model = fit_arima(data, p=3, d=0, q=2, start_params=None)
predictions = model.predict(n_periods=5)
```

## Logging

All pipelines use Python's logging module with hierarchical loggers:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Loggers:
# - ml.pipelines    - Pipeline execution
# - ml.cache        - Cache operations
# - ml.deploy.lstm  - LSTM deployment
# - ml.deploy.arima - ARIMA deployment
```

## CLI Usage

### ARIMA Pipeline

```bash
# Basic usage
uv run python -m ml.pipelines arima

# With hyperparameters
uv run python -m ml.pipelines arima --p 5 --d 1 --q 3

# Force re-segmentation (triggers segment + split + fit)
uv run python -m ml.pipelines arima --segment

# Tune hyperparameters
uv run python -m ml.pipelines arima --tune-hyperparams

# Custom parameters
uv run python -m ml.pipelines arima \
  --segment \
  --max-timedelta 15 \
  --max-distance 0.01 \
  --min-segment-length 5 \
  --test-ratio 0.2 \
  --random-seed 42
```

### LSTM Pipeline

```bash
# Basic usage
uv run python -m ml.pipelines lstm

# Force re-training
uv run python -m ml.pipelines lstm --train

# Force stops preprocessing (triggers stops + train)
uv run python -m ml.pipelines lstm --stops

# Custom parameters
uv run python -m ml.pipelines lstm \
  --stops \
  --epochs 50 \
  --batch-size 128 \
  --hidden-size 100 \
  --num-layers 3 \
  --seq-len 15 \
  --limit-polylines 5
```

### Other Pipelines

```bash
# Run just preprocessing
uv run python -m ml.pipelines preprocess

# Run just stops pipeline
uv run python -m ml.pipelines stops

# Run load pipeline
uv run python -m ml.pipelines load
```

## Column Reference

### After Load Pipeline
- `id`, `vehicle_id`, `timestamp`, `latitude`, `longitude`, `heading_degrees`, `speed_mph`

### After Preprocess Pipeline
Adds:
- `epoch_seconds` - Seconds since 2025-01-01
- `route` - Matched route name
- `closest_lat`, `closest_lon` - Closest polyline point
- `polyline_idx` - Polyline index
- `segment_idx` - Segment index within polyline
- `dist_to_route` - Distance to route (km)

### After Segment Pipeline
Adds:
- `segment_id` - Trip segment identifier
- `distance_km` - Distance from previous point
- `speed_kmh` - Calculated speed

### After Stops Pipeline
Adds:
- `stop_name` - Name of detected stop
- `stop_route` - Route for stop
- `eta_seconds` - Seconds to next stop
- `dist_from_start` - Distance from polyline start (km)
- `dist_to_end` - Distance to polyline end (km)
- `polyline_length` - Total polyline length (km)

## Performance

### Benchmarks (approximate)

- **Load**: ~2-5 seconds (1.8M rows from database)
- **Preprocess**: ~3 hours first run, <1 second cached
- **Segment**: ~1-2 seconds (1.8M rows)
- **Stops**: ~5-10 seconds (adds ETA calculations)
- **Split**: ~0.5 seconds first run, <0.1 second cached
- **ARIMA Training**: ~5-10 minutes per segment
- **LSTM Training**: ~10-30 minutes per polyline (20 epochs)

### Optimization Tips

1. **Use caching**: Don't use force flags unless data/code changed
2. **Limit polylines for LSTM**: Use `--limit-polylines` during development
3. **Parallel processing**: LSTM trains one model per polyline
4. **Warm-start ARIMA**: Use cached parameters for faster convergence

## File Structure

```
ml/
├── __init__.py                  # Package logger configuration
├── README.md                    # This file
├── pipelines.py                 # Complete pipeline workflows
├── cache.py                     # Cache utilities
│
├── cache/                       # Cached data and models
│   ├── shared/                  # Cross-pipeline caches
│   ├── arima/                   # ARIMA-specific caches
│   └── lstm/                    # LSTM per-polyline models
│       └── <route>_<idx>/       # Each polyline gets a directory
│           ├── data.csv
│           ├── train.csv
│           ├── test.csv
│           └── model.pth
│
├── data/
│   ├── load.py                  # Database loading
│   └── preprocess.py            # Preprocessing functions
│
├── models/
│   ├── lstm.py                  # LSTM implementation
│   └── arima.py                 # ARIMA wrapper
│
├── training/
│   └── train.py                 # Training utilities
│
├── evaluation/
│   └── evaluate.py              # Evaluation functions
│
├── validate/
│   └── hyperparameters.py       # Hyperparameter tuning
│
└── deploy/
    ├── __init__.py
    ├── arima.py                 # ARIMA deployment
    └── lstm.py                  # LSTM deployment
```

## Examples

### Example 1: Load Trained Models

```python
from ml.deploy import load_arima, load_lstm, list_lstm_models

# List available models
arima_models = list_arima_models()
lstm_models = list_lstm_models()

print(f"Available ARIMA models: {len(arima_models)}")
for model in lstm_models:
    print(f"  LSTM: {model['route_name']} segment {model['polyline_idx']}")

# Load specific models
arima_params = load_arima(p=3, d=0, q=2)
all_lstm_models = load_lstm()

# Use LSTM model
model = all_lstm_models[('East Route', 0)]
prediction = model.predict(input_sequence)
```

### Example 2: Custom Pipeline

```python
from ml.pipelines import (
    load_pipeline,
    preprocess_pipeline,
    segment_pipeline,
    stops_pipeline
)

# Load and preprocess
df = load_pipeline(load=False)  # Use cache
df = preprocess_pipeline(df=df, preprocess=False)

# Custom segmentation
df = segment_pipeline(
    df=df,
    max_timedelta=20,
    max_distance=0.01,
    min_segment_length=5,
    window_size=7
)

# Add stops
df = stops_pipeline(
    df=df,
    max_timedelta=30,
    max_distance=0.005
)

print(f"Processed {len(df)} points in {df['segment_id'].nunique()} segments")
```

### Example 3: Train with Custom Parameters

```python
from ml.pipelines import arima_pipeline

# Train ARIMA with custom settings
results = arima_pipeline(
    p=5, d=1, q=3,
    segment=True,
    max_timedelta=20,
    test_ratio=0.3,
    random_seed=123,
    limit_segments=10
)

print(f"RMSE: {results['overall_rmse']:.4f}")
print(f"MAE: {results['overall_mae']:.4f}")
```

## Future Enhancements

- [ ] Real-time inference API endpoint
- [ ] Model versioning and A/B testing
- [ ] AutoML for hyperparameter search
- [ ] Ensemble methods (ARIMA + LSTM)
- [ ] Feature importance analysis
- [ ] Data quality monitoring
- [ ] Prediction confidence intervals
- [ ] Online learning for model updates
