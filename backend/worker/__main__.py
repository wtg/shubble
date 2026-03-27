"""Entry point for running the worker as a module."""
import asyncio
from backend.locations_worker import run_locations_worker
from backend.ml_worker import run_ml_worker

if __name__ == "__main__":
    asyncio.run(run_locations_worker())
    asyncio.run(run_ml_worker())