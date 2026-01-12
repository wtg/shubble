# ML Scripts

Collection of utility scripts for ML model training and data analysis.

## Scripts

### `average_polyline_eta.py`

Calculates the average time it takes a vehicle to travel each polyline segment based on stop-to-stop observations.

**What it does:**
1. Loads data from the stops pipeline
2. Finds segments where the shuttle stops at more than one stop
3. Calculates the time difference between consecutive stops within each segment
4. Aggregates the average travel time for each polyline
5. Saves results to `lstm/<route>_<idx>/average_travel_time.csv`

**Usage:**
```bash
python ml/scripts/average_polyline_eta.py
```

**Output:**
Creates `average_travel_time.csv` in each polyline directory with columns:
- `route`: Route name (e.g., "NORTH", "WEST")
- `polyline_idx`: Polyline index
- `avg_travel_time_seconds`: Average time to traverse this polyline
- `std_travel_time_seconds`: Standard deviation of travel times
- `sample_count`: Number of observations
- `min_time`: Minimum observed time
- `max_time`: Maximum observed time

**Example output:**
```csv
route,polyline_idx,avg_travel_time_seconds,std_travel_time_seconds,sample_count,min_time,max_time
WEST,6.0,14.63,18.70,27064,4.0,286.0
```

This means polyline 6 on the WEST route takes an average of 14.6 seconds to traverse, based on 27,064 observations.

**Use cases:**
- ETA prediction
- Route optimization
- Performance monitoring
- Identifying slow segments
