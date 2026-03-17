# Online KNN – Time to Known Stops

Predicts **time (seconds) to 2 known location stops** from raw GPS points using KNN. No route definitions are used; labels are derived from trajectories (time from each point until the vehicle first reaches each stop).

## Data source

- **Testing**: Raw locations from CSV (`ml/cache/shared/locations_raw.csv` by default). Same schema as the DB export: `vehicle_id`, `latitude`, `longitude`, `timestamp`.
- **Real-time (future)**: Implementation is left open. Use a custom `RawLocationsLoader` that returns a DataFrame (e.g. from PostgreSQL for “today”, or an incremental batch). The rest of the pipeline (labels → fit → predict) stays the same.

## Pipeline

1. **Load** raw locations (CSV or via `RawLocationsLoader`).
2. **Label**: For each point, compute `eta_seconds_stop_0` and `eta_seconds_stop_1` by looking forward in the same vehicle’s trajectory until it enters the stop radius.
3. **Fit** `TimeToStopKNN` on (lat, lon) with ETA columns as targets.
4. **Predict**: For a (lat, lon) query, return median (or mean) of k neighbors’ ETAs per stop.

## Stops and route filter

- **Stops**: Only **Student Union** and **Colonie**. Coordinates are hardcoded in `ml/online_knn/stops_config.py`. We do **not** read `shared/routes.json` or use any other stops; scripts that compute "ETA to next stop" over all route stops are not used here.
- **West shuttles only**: The learning curve uses `ROUTE_FILTER = "WEST"`, so it loads the **preprocessed** CSV (which has a `route` column from polyline matching) and keeps only rows where `route == "WEST"`. Run `python -m ml.pipelines preprocess` once so `ml/cache/shared/locations_preprocessed.csv` exists. To use all shuttles instead, set `ROUTE_FILTER = None` in `run_learning_curve.py` (then raw CSV is used).

## How to run the test

From the project root (where `ml/` lives):

```bash
python -m ml.online_knn
```

This uses `ml/cache/shared/locations_raw.csv`, fits the model on up to 20k rows for the date in the script, and prints a sample prediction. Edit `ml/online_knn/__main__.py` to change `STOPS`, `DATE_FILTER`, `MAX_ROWS`, or the query point.

**Requirement:** The CSV must exist and contain `vehicle_id`, `latitude`, `longitude`, `timestamp`. If you don’t have it, run the load pipeline once: `python -m ml.pipelines load` (with DB configured), or place a CSV with those columns at `ml/cache/shared/locations_raw.csv`.

### Learning curve (error vs amount of data)

Start with 100 points, add 2 every step, refit KNN, and plot error (MAE in seconds) on a fixed test set:

```bash
python -m ml.online_knn.run_learning_curve
```

Plot is saved to `ml/cache/shared/knn_error_vs_data.png` (X = training set size, Y = mean absolute error). Requires `matplotlib`.

## Usage (CSV testing)

```python
from pathlib import Path
from ml.online_knn import load_raw_locations_csv, compute_trajectory_etas, TimeToStopKNN
from ml.online_knn.pipeline import run_from_csv

# Define 2 stops (lat, lon)
stops = [
    (42.730, -73.676),  # stop 0
    (42.724, -73.681),  # stop 1
]

# Fit from CSV (optional: csv_path=Path("ml/cache/shared/locations_raw.csv"))
model = run_from_csv(stops, n_neighbors=5, date_filter="2025-07-31")

# Predict seconds to each stop for one point
eta0, eta1 = model.predict(42.727, -73.684)
print(f"Seconds to stop 0: {eta0:.0f}, stop 1: {eta1:.0f}")
```

## Real-time extension

- **Data**: Implement `RawLocationsLoader.load()` to return today’s (or latest) raw locations, e.g. from PostgreSQL with `timestamp >= get_campus_start_of_day()`, or an incremental batch.
- **Update**: Call `run_with_loader(loader, stops)` periodically (e.g. every 1–5 minutes) to refit on the current day’s data; or add an incremental update API that appends new rows to the stored points and refits the KNN index.
- **Prediction**: Existing `model.predict(lat, lon)` and `model.predict_batch(df)` stay unchanged; no real-time-specific API required.
