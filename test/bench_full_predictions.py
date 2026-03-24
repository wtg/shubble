"""
Benchmark: generate_and_save_predictions with ARIMA on vs off.

Default: LSTM ETAs + velocity (ARIMA vs cheap). Use --velocity-only to skip LSTM
and time only predict_next_state + save (ARIMA vs cheap baseline).

Use --minimal for a single timing with no LSTM and no ARIMA (ultimate cheap baseline).
Use --triple to compare three modes: LSTM+ARIMA, LSTM+cheap, and no-LSTM+cheap.

  uv run python test/bench_full_predictions.py
  uv run python test/bench_full_predictions.py --velocity-only
  uv run python test/bench_full_predictions.py --minimal
  uv run python test/bench_full_predictions.py --triple

Prereqs: Redis + PostgreSQL; warm cache (worker or API). Optional: REDIS_URL, DATABASE_URL in .env.
"""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark prediction pipeline: ARIMA vs cheap velocity (optional LSTM)"
    )
    parser.add_argument("--with-arima", action="store_true", help="Run only with ARIMA enabled")
    parser.add_argument("--no-arima", action="store_true", help="Run only with ARIMA disabled")
    parser.add_argument(
        "--velocity-only",
        action="store_true",
        help="Skip LSTM ETAs (velocity + save only); isolates ARIMA vs cheap",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Single run: no LSTM, no ARIMA (cheap velocity only); absolute baseline",
    )
    parser.add_argument(
        "--triple",
        action="store_true",
        help="Compare three modes: LSTM+ARIMA, LSTM+cheap, no-LSTM+cheap (omit --velocity-only)",
    )
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

    if args.minimal and (args.with_arima or args.no_arima):
        parser.error("--minimal cannot be combined with --with-arima or --no-arima")
    if args.triple and args.velocity_only:
        parser.error("--triple compares full pipeline modes; omit --velocity-only")
    if args.triple and args.minimal:
        parser.error("--triple already includes the minimal baseline; omit --minimal")
    if args.triple and (args.with_arima or args.no_arima):
        parser.error("--triple runs all three modes; omit --with-arima / --no-arima")

    run_both = not args.with_arima and not args.no_arima and not args.minimal
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess_env = os.environ.copy()
    subprocess_env["PYTHONPATH"] = root

    _arima_placeholder = "__ARIMA_ENABLED_VALUE__"
    _lstm_placeholder = "__LSTM_PREDICTIONS_ENABLED_VALUE__"
    inline_code = """
import asyncio
import os
import time

os.environ["ARIMA_ENABLED"] = """ + _arima_placeholder + """
os.environ["LSTM_PREDICTIONS_ENABLED"] = """ + _lstm_placeholder + """

async def run():
    from backend.config import settings
    from backend.cache import init_cache
    from backend.database import create_async_db_engine, create_session_factory
    from backend.cache_dataframe import get_today_dataframe
    from backend.worker.data import generate_and_save_predictions

    await init_cache(settings.REDIS_URL)
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

    def _run_subprocess(*, lstm: bool, arima: bool) -> subprocess.CompletedProcess:
        env = {
            **subprocess_env,
            "ARIMA_ENABLED": "true" if arima else "false",
            "LSTM_PREDICTIONS_ENABLED": "true" if lstm else "false",
        }
        code = (
            inline_code.replace(_arima_placeholder, "'true'" if arima else "'false'", 1)
            .replace(_lstm_placeholder, "'true'" if lstm else "'false'", 1)
        )
        return subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=root,
        )

    def _labels():
        if args.velocity_only:
            return ("Velocity + ARIMA", "Velocity + cheap velocity")
        return ("LSTM + ARIMA", "LSTM + cheap velocity")

    def _parse_elapsed(out: subprocess.CompletedProcess) -> float | None:
        for line in (out.stdout or "").strip().splitlines():
            if line.startswith("ELAPSED:"):
                return float(line.split(":", 1)[1])
        return None

    def _print_no_data_help():
        print(
            "\nNo data. Do this first (then run the benchmark again):",
            file=sys.stderr,
        )
        print("  1. Start Postgres + Redis:  docker-compose --profile backend up -d postgres redis", file=sys.stderr)
        print("  2. Start test server:      uv run uvicorn test.server.server:app --port 4000", file=sys.stderr)
        print("  3. Set SAMSARA_BASE_URL=http://127.0.0.1:4000 (and API_KEY if needed), then start worker:", file=sys.stderr)
        print("     uv run python -m backend.worker", file=sys.stderr)
        print("  4. Wait 30–60s for locations to be written, then run this benchmark again.", file=sys.stderr)

    if args.minimal:
        try:
            out = _run_subprocess(lstm=False, arima=False)
        except subprocess.TimeoutExpired:
            print("Timed out", file=sys.stderr)
            return 1
        if out.returncode != 0:
            print("Subprocess failed", file=sys.stderr)
            if out.stderr:
                print(out.stderr, file=sys.stderr)
            return 1
        if "NO_DATA" in (out.stdout or ""):
            _print_no_data_help()
            return 1
        elapsed = _parse_elapsed(out)
        if elapsed is None:
            return 1
        print(
            f"\n--- Minimal baseline (no LSTM, no ARIMA; generate_and_save_predictions x {args.calls} calls) ---"
        )
        print(f"  No LSTM + cheap velocity: {elapsed:.3f}s")
        return 0

    if run_both and args.triple:
        triple_runs = [
            ("LSTM + ARIMA", True, True),
            ("LSTM + cheap velocity", True, False),
            ("No LSTM + cheap velocity", False, False),
        ]
        results: dict[str, float | None] = {}
        for label, lstm, arima in triple_runs:
            try:
                out = _run_subprocess(lstm=lstm, arima=arima)
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
                results[label] = _parse_elapsed(out)
            except subprocess.TimeoutExpired:
                results[label] = None
                print(f"{label}: timed out", file=sys.stderr)

        if all(v is None for v in results.values()):
            _print_no_data_help()
        print(f"\n--- Triple compare (generate_and_save_predictions x {args.calls} calls) ---")
        for label, elapsed in results.items():
            if elapsed is not None:
                print(f"  {label}: {elapsed:.3f}s")
            else:
                print(f"  {label}: (failed or no data)")
        return 0

    if run_both:
        results = {}
        label_arima, label_cheap = _labels()
        lstm_default = not args.velocity_only
        for label, arima in [(label_arima, True), (label_cheap, False)]:
            try:
                out = _run_subprocess(lstm=lstm_default, arima=arima)
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
                results[label] = _parse_elapsed(out)
            except subprocess.TimeoutExpired:
                results[label] = None
                print(f"{label}: timed out", file=sys.stderr)

        if all(v is None for v in results.values()):
            _print_no_data_help()
        mode = "velocity-only (no LSTM)" if args.velocity_only else "full pipeline (LSTM + velocity)"
        print(f"\n--- {mode} (generate_and_save_predictions x {args.calls} calls) ---")
        for label, elapsed in results.items():
            if elapsed is not None:
                print(f"  {label}: {elapsed:.3f}s")
            else:
                print(f"  {label}: (failed or no data)")
        t_a = results.get(label_arima)
        t_c = results.get(label_cheap)
        if t_a is not None and t_c is not None and t_c > 0:
            if t_a > t_c:
                print(f"  → {label_arima} is ~{t_a / t_c:.1f}x slower than {label_cheap}")
            else:
                print(f"  → {label_arima} is ~{t_c / t_a:.1f}x faster than {label_cheap}")
        return 0

    arima = args.with_arima
    lstm_single = not args.velocity_only
    try:
        out = _run_subprocess(lstm=lstm_single, arima=arima)
    except subprocess.TimeoutExpired:
        print("Timed out", file=sys.stderr)
        return 1
    if out.returncode != 0:
        print("Subprocess failed", file=sys.stderr)
        if out.stderr:
            print(out.stderr, file=sys.stderr)
        return 1
    if "NO_DATA" in (out.stdout or ""):
        print("No data. Start Postgres + Redis, then start the test server and worker; wait 30–60s and re-run.", file=sys.stderr)
        return 1
    elapsed = _parse_elapsed(out)
    if elapsed is None:
        return 1
    label_arima, label_cheap = _labels()
    label = label_arima if arima else label_cheap
    mode = "velocity-only" if args.velocity_only else "full pipeline"
    print(f"{label}: {elapsed:.3f}s ({mode}, generate_and_save_predictions x {args.calls} calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
