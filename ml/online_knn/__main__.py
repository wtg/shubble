"""
Run the online KNN pipeline from CSV (for testing).

  python -m ml.online_knn

Uses only Student Union and Colonie (fixed coords; routes.json not used).
Data: locations_raw.csv, or set route filter in run_learning_curve to use preprocessed.
"""
from ml.online_knn import run_from_csv
from ml.online_knn.stops_config import STOPS_STUDENT_UNION_COLONIE

STOPS = STOPS_STUDENT_UNION_COLONIE

# Use only this many rows from CSV (None = use all). Set e.g. 20_000 for fast test.
MAX_ROWS = 20_000

# Keep only this date (must match timestamps in CSV). None = use all dates.
DATE_FILTER = "2025-07-31"

# Query point for a sample prediction
QUERY_LAT, QUERY_LON = 42.727, -73.684


def main():
    print("Loading CSV, computing ETAs, fitting KNN...")
    model = run_from_csv(
        STOPS,
        n_neighbors=5,
        date_filter=DATE_FILTER,
        max_rows=MAX_ROWS,
    )
    eta0, eta1 = model.predict(QUERY_LAT, QUERY_LON)
    print(f"Sample prediction at ({QUERY_LAT}, {QUERY_LON}):")
    print(f"  Seconds to stop 0: {eta0:.0f}" if eta0 == eta0 else "  Seconds to stop 0: (no neighbors)")
    print(f"  Seconds to stop 1: {eta1:.0f}" if eta1 == eta1 else "  Seconds to stop 1: (no neighbors)")
    print("Done.")


if __name__ == "__main__":
    main()
