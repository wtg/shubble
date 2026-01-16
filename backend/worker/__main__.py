"""Entry point for running the worker as a module."""
import asyncio
from .worker import run_worker

if __name__ == "__main__":
    asyncio.run(run_worker())
