import asyncio
from .locations_worker import run_locations_worker

if __name__ == "__main__":
    asyncio.run(run_locations_worker())
