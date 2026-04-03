"""Async background worker for fetching vehicle data from Samsara API."""
import asyncio
import logging
import os

from sqlalchemy import select

from backend.config import settings
from backend.cache import init_cache, close_cache
from backend.database import create_async_db_engine, create_session_factory
from backend.models import VehicleLocation
from backend.utils import get_vehicles_in_geofence
from backend.ml_worker.data import generate_and_save_predictions
from backend.cache_dataframe import update_today_dataframe

# Logging config for Worker
worker_log_level = os.getenv("LOG_LEVEL") or settings.get_log_level("ml-worker")
numeric_level = logging._nameToLevel.get(worker_log_level.upper(), logging.INFO)
logging.basicConfig(
    level=numeric_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("locations-worker")
logger.info(f"Worker logging level: {worker_log_level}")

async def run_ml_worker():
   logger.info("ML worker started")

   db_engine = create_async_db_engine(settings.DATABASE_URL, echo=settings.DEBUG)
   session_factory = create_session_factory(db_engine)

   try:
      await init_cache(settings.REDIS_URL)
   except Exception as e:
      logger.error("Redis cache was not initialized properly:", e)

   # Prevents stale data when worker restarts
   logger.info("Starting up ML cache")
   try:
      await update_today_dataframe()
      vehicle_ids = await get_vehicles_in_geofence(session_factory)
      await generate_and_save_predictions(list(vehicle_ids))
      logger.info("ML cache start up complete")
   except Exception as e:
      logger.exception("Start up ML cache failed, worker will continue:", e)

   interval = 5

   async def ticker(interval_seconds):
      next_tick = asyncio.get_event_loop().time()
      while True:
         next_tick += interval_seconds
         yield
         now = asyncio.get_event_loop().time()
         sleep_time = next_tick - now
         if sleep_time > 0:
            await asyncio.sleep(sleep_time)
         else:
            next_tick = now
   
   try:
      async for _ in ticker(interval):
         try:
            logger.info("Refreshing ML dataframe")
            await update_today_dataframe()

            logger.info("Generating predictions")
            vehicle_ids = await get_vehicles_in_geofence(session_factory)

            await generate_and_save_predictions(vehicle_ids)
         except Exception as e:
            logger.exception("Error in ML worker loop")

   finally:
      logger.info("Shutting down worker...")
      await close_cache()
      await db_engine.dispose()
      logger.info("DB connections closed")

if __name__ == "__main__":
   asyncio.run(run_ml_worker())
