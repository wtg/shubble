"""
Benchmark: CPU / time impact of ARIMA vs no-ARIMA in worker prediction path.

Run both modes and compare total time (and optionally process CPU):

  uv run python test/bench_worker_arima.py

Or run a single mode:

  uv run python test/bench_worker_arima.py --with-arima
  uv run python test/bench_worker_arima.py --no-arima

For "with-arima" to actually run ARIMA fits, you need cached ARIMA params:
  uv run python -m ml.pipelines arima

Otherwise the with-arima run exits early (no timing). Use --no-arima to see
the fast path baseline.
"""
import argparse
import asyncio
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

# Build a minimal DataFrame so predict_next_state has something to work on
# (no DB/Redis required)
def _make_synthetic_df(num_vehicles: int = 5, points_per_vehicle: int = 100):
    import pandas as pd
    import numpy as np
    rows = []
    base = datetime.now(timezone.utc) - timedelta(minutes=points_per_vehicle)
    for v in range(num_vehicles):
        vehicle_id = f"bench-vehicle-{v}"
        for i in range(points_per_vehicle):
            ts = base + timedelta(seconds=i * 3)
            # speed_kmh: required for predict_next_state
            speed = 20.0 + np.random.randn() * 5
            rows.append({
                "vehicle_id": vehicle_id,
                "timestamp": ts,
                "speed_kmh": max(0.0, speed),
            })
    df = pd.DataFrame(rows)
    return df, [r["vehicle_id"] for r in rows[:num_vehicles]]


def _run_benchmark(arima_enabled: bool, n_calls: int = 5) -> float:
    """Run predict_next_state multiple times; return total elapsed seconds."""
    # Set env before any backend.worker.data import (so config reads it)
    os.environ["ARIMA_ENABLED"] = "true" if arima_enabled else "false"

    from backend.worker.data import predict_next_state

    df, vehicle_ids = _make_synthetic_df(num_vehicles=5, points_per_vehicle=100)
    start = time.perf_counter()
    for _ in range(n_calls):
        asyncio.run(predict_next_state(vehicle_ids, df=df.copy()))
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark ARIMA vs no-ARIMA in worker prediction path")
    parser.add_argument("--with-arima", action="store_true", help="Run only with ARIMA enabled")
    parser.add_argument("--no-arima", action="store_true", help="Run only with ARIMA disabled")
    parser.add_argument("--calls", type=int, default=5, help="Number of predict_next_state calls per run (default 5)")
    args = parser.parse_args()

    run_both = not args.with_arima and not args.no_arima
    if run_both:
        # Run in two subprocesses so each gets a clean env and config
        results = {}
        for label, arima in [("ARIMA enabled", True), ("ARIMA disabled", False)]:
            env = os.environ.copy()
            env["ARIMA_ENABLED"] = "true" if arima else "false"
            code = """
import asyncio
import os
import time
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

os.environ["ARIMA_ENABLED"] = """ + ("'true'" if arima else "'false'") + """
from backend.worker.data import predict_next_state

def make_df():
    rows = []
    base = datetime.now(timezone.utc) - timedelta(minutes=100)
    for v in range(5):
        for i in range(100):
            ts = base + timedelta(seconds=i * 3)
            speed = 20.0 + np.random.randn() * 5
            rows.append({"vehicle_id": f"bench-{v}", "timestamp": ts, "speed_kmh": max(0.0, speed)})
    return pd.DataFrame(rows), [f"bench-{v}" for v in range(5)]

df, vehicle_ids = make_df()
n = """ + str(args.calls) + """
start = time.perf_counter()
for _ in range(n):
    asyncio.run(predict_next_state(vehicle_ids, df=df.copy()))
elapsed = time.perf_counter() - start
print(f"ELAPSED:{elapsed:.3f}")
"""
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            try:
                out = subprocess.run(
                    [sys.executable, "-c", code],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=root,
                )
                if out.returncode != 0:
                    results[label] = None
                    print(f"{label}: subprocess failed\n{out.stderr}", file=sys.stderr)
                    continue
                for line in out.stdout.strip().splitlines():
                    if line.startswith("ELAPSED:"):
                        results[label] = float(line.split(":", 1)[1])
                        break
                else:
                    results[label] = None
            except subprocess.TimeoutExpired:
                results[label] = None
                print(f"{label}: timed out", file=sys.stderr)

        print("\n--- Benchmark (predict_next_state x {} calls, 5 vehicles x 100 points) ---".format(args.calls))
        for label, elapsed in results.items():
            if elapsed is not None:
                print(f"  {label}: {elapsed:.3f}s")
            else:
                print(f"  {label}: (failed or no cached ARIMA params)")
        if results.get("ARIMA enabled") and results.get("ARIMA disabled"):
            ratio = results["ARIMA enabled"] / results["ARIMA disabled"]
            print(f"  → ARIMA-on is ~{ratio:.1f}x slower")
        return 0

    # Single mode
    arima = args.with_arima
    os.environ["ARIMA_ENABLED"] = "true" if arima else "false"
    label = "ARIMA enabled" if arima else "ARIMA disabled"
    try:
        elapsed = _run_benchmark(arima, n_calls=args.calls)
        print(f"{label}: {elapsed:.3f}s (predict_next_state x {args.calls} calls)")
    except FileNotFoundError as e:
        if arima:
            print(f"{label}: skipped (no cached ARIMA params). Run: uv run python -m ml.pipelines arima", file=sys.stderr)
        else:
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
