"""
Benchmark: LSTM + ARIMA vs LSTM + cheap velocity (full generate_and_save_predictions).

Times the full prediction pipeline (get_all_stop_times + predict_next_state + save)
with ARIMA enabled vs disabled. Requires Redis (and DB for save_predictions) to be
running and a warm cache (run the worker or API once so get_today_dataframe has data).

  uv run python test/bench_full_predictions.py

Or run a single mode:

  uv run python test/bench_full_predictions.py --with-arima
  uv run python test/bench_full_predictions.py --no-arima

Prereqs: Redis + PostgreSQL running; processed dataframe in Redis (e.g. start
worker or hit API so cache is warm). Optional: set REDIS_URL, DATABASE_URL in .env.
"""
import argparse
import asyncio
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LSTM+ARIMA vs LSTM+cheap velocity (full prediction pipeline)"
    )
    parser.add_argument("--with-arima", action="store_true", help="Run only with ARIMA enabled")
    parser.add_argument("--no-arima", action="store_true", help="Run only with ARIMA disabled")
    parser.add_argument(
        "--calls",
        type=int,
        default=2,
        help="Number of generate_and_save_predictions calls per run (default 2)",
    )
    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=10,
        help="Max vehicle IDs to use from cache (default 10)",
    )
    args = parser.parse_args()

    run_both = not args.with_arima and not args.no_arima
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # So subprocess can import backend when run with -c
    subprocess_env = os.environ.copy()
    subprocess_env["PYTHONPATH"] = root

    # Placeholder replaced per run so each subprocess gets correct ARIMA_ENABLED
    _arima_placeholder = "__ARIMA_ENABLED_VALUE__"
    inline_code = """
import asyncio
import os
import time

os.environ["ARIMA_ENABLED"] = """ + _arima_placeholder + """

async def run():
    from backend.config import settings
    from backend.cache import init_cache
    from backend.database import create_async_db_engine, create_session_factory
    from backend.cache_dataframe import get_today_dataframe
    from backend.worker.data import generate_and_save_predictions

    await init_cache(settings.REDIS_URL)
    # Session factory is required for get_today_dataframe() when loading from DB
    engine = create_async_db_engine(settings.DATABASE_URL, echo=False)
    create_session_factory(engine)
    df = await get_today_dataframe()
    if df is None or df.empty:
        print("NO_DATA", flush=True)
        return
    vids = df["vehicle_id"].astype(str).unique().tolist()[:""" + str(args.max_vehicles) + """]
    n = """ + str(args.calls) + """
    start = time.perf_counter()
    for _ in range(n):
        await generate_and_save_predictions(vids)
    elapsed = time.perf_counter() - start
    print(f"ELAPSED:{elapsed:.3f}", flush=True)

asyncio.run(run())
"""

    if run_both:
        results = {}
        for label, arima in [("LSTM + ARIMA", True), ("LSTM + cheap velocity", False)]:
            env = {**subprocess_env, "ARIMA_ENABLED": "true" if arima else "false"}
            code = inline_code.replace(_arima_placeholder, "'true'" if arima else "'false'", 1)
            try:
                out = subprocess.run(
                    [sys.executable, "-c", code],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=root,
                )
                if out.returncode != 0:
                    results[label] = None
                    print(f"{label}: subprocess failed", file=sys.stderr)
                    if out.stderr:
                        print(out.stderr, file=sys.stderr)
                    continue
                if "NO_DATA" in (out.stdout or ""):
                    results[label] = None
                    print(f"{label}: no data (see steps below)", file=sys.stderr)
                    continue
                for line in (out.stdout or "").strip().splitlines():
                    if line.startswith("ELAPSED:"):
                        results[label] = float(line.split(":", 1)[1])
                        break
                else:
                    results[label] = None
            except subprocess.TimeoutExpired:
                results[label] = None
                print(f"{label}: timed out", file=sys.stderr)

        no_data = all(v is None for v in results.values())
        if no_data:
            print(
                "\nNo data. Do this first (then run the benchmark again):",
                file=sys.stderr,
            )
            print("  1. Start Postgres + Redis:  docker-compose --profile backend up -d postgres redis", file=sys.stderr)
            print("  2. Start test server:      uv run uvicorn test.server.server:app --port 4000", file=sys.stderr)
            print("  3. Set SAMSARA_BASE_URL=http://127.0.0.1:4000 (and API_KEY if needed), then start worker:", file=sys.stderr)
            print("     uv run python -m backend.worker", file=sys.stderr)
            print("  4. Wait 30–60s for locations to be written, then run this benchmark again.", file=sys.stderr)
        print("\n--- Full pipeline (generate_and_save_predictions x {} calls) ---".format(args.calls))
        for label, elapsed in results.items():
            if elapsed is not None:
                print(f"  {label}: {elapsed:.3f}s")
            else:
                print(f"  {label}: (failed or no data)")
        if results.get("LSTM + ARIMA") and results.get("LSTM + cheap velocity"):
            ratio = results["LSTM + ARIMA"] / results["LSTM + cheap velocity"]
            print(f"  → LSTM+ARIMA is ~{ratio:.1f}x slower")
        return 0

    # Single mode
    arima = args.with_arima
    env = {**subprocess_env, "ARIMA_ENABLED": "true" if arima else "false"}
    code = inline_code.replace(_arima_placeholder, "'true'" if arima else "'false'", 1)
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=root,
        )
        if out.returncode != 0:
            print("Subprocess failed", file=sys.stderr)
            if out.stderr:
                print(out.stderr, file=sys.stderr)
            return 1
        if "NO_DATA" in (out.stdout or ""):
            print("No data. Start Postgres + Redis, then start the test server and worker; wait 30–60s and re-run.", file=sys.stderr)
            return 1
        for line in (out.stdout or "").strip().splitlines():
            if line.startswith("ELAPSED:"):
                elapsed = float(line.split(":", 1)[1])
                label = "LSTM + ARIMA" if arima else "LSTM + cheap velocity"
                print(f"{label}: {elapsed:.3f}s (generate_and_save_predictions x {args.calls} calls)")
                return 0
        return 1
    except subprocess.TimeoutExpired:
        print("Timed out", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
