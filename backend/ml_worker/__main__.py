import asyncio
from .ml_worker import run_ml_worker

if __name__ == "__main__":
    asyncio.run(run_ml_worker())
