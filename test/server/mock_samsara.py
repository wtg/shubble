"""Mock Samsara API endpoints for the test server."""
import logging
from datetime import datetime

import numpy as np
from fastapi import APIRouter

from .shuttles import shuttle_lock, shuttles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fleet", tags=["samsara"])


@router.get("/vehicles/stats")
async def mock_stats(vehicleIds: str = "", after: str = None):
    """Mock Samsara vehicle stats endpoint."""
    vehicle_ids = vehicleIds.split(",") if vehicleIds else []

    logger.info(
        f"[MOCK API] Received stats snapshot request for vehicles {vehicle_ids} after={after}"
    )

    async with shuttle_lock:
        data = []
        for shuttle_id in vehicle_ids:
            if shuttle_id in shuttles:
                shuttle = shuttles[shuttle_id]
                lat, lon = shuttle.location
                # Add small noise for realism
                lat += np.random.normal(0, 0.000001)
                lon += np.random.normal(0, 0.000001)
                data.append(
                    {
                        "id": shuttle_id,
                        "name": shuttle_id[-3:],
                        "gps": {
                            "latitude": lat,
                            "longitude": lon,
                            "time": shuttle.last_updated.isoformat(timespec="seconds").replace("+00:00", "Z"),
                            "speedMilesPerHour": shuttle.speed,
                            "headingDegrees": 90,
                            "reverseGeo": {"formattedLocation": "Test Location"},
                        },
                    }
                )

        return {
            "data": data,
            "pagination": {"hasNextPage": False, "endCursor": "fake-token-next"},
        }


@router.get("/vehicles/stats/feed")
async def mock_feed(vehicleIds: str = "", after: str = None):
    """Mock Samsara vehicle stats feed endpoint."""
    vehicle_ids = vehicleIds.split(",") if vehicleIds else []

    logger.info(
        f"[MOCK API] Received stats feed request for vehicles {vehicle_ids} after={after}"
    )

    async with shuttle_lock:
        data = []
        for shuttle_id in vehicle_ids:
            if shuttle_id in shuttles:
                shuttle = shuttles[shuttle_id]
                lat, lon = shuttle.location
                # Add small noise for realism
                lat += np.random.normal(0, 0.00008)
                lon += np.random.normal(0, 0.00008)
                data.append(
                    {
                        "id": shuttle_id,
                        "name": shuttle_id[-3:],
                        "gps": [
                            {
                                "latitude": lat,
                                "longitude": lon,
                                "time": shuttle.last_updated.isoformat(timespec="seconds").replace("+00:00", "Z"),
                                "speedMilesPerHour": shuttle.speed,
                                "headingDegrees": 90,
                                "reverseGeo": {"formattedLocation": "Test Location"},
                            }
                        ],
                    }
                )

        return {
            "data": data,
            "pagination": {"hasNextPage": False, "endCursor": "fake-token-next"},
        }


@router.get("/driver-vehicle-assignments")
async def mock_driver_assignments(vehicleIds: str = ""):
    """Mock endpoint for driver-vehicle assignments."""
    vehicle_ids = vehicleIds.split(",") if vehicleIds else []

    logger.info(
        f"[MOCK API] Received driver-vehicle assignments request for vehicles {vehicle_ids}"
    )

    async with shuttle_lock:
        data = []
        for shuttle_id in vehicle_ids:
            if shuttle_id in shuttles:
                # Generate a mock driver for each active shuttle
                driver_id = f"driver-{shuttle_id[-3:]}"
                driver_name = f"Driver {shuttle_id[-3:]}"
                data.append(
                    {
                        "assignedAtTime": datetime.now()
                        .isoformat(timespec="seconds")
                        .replace("+00:00", "Z"),
                        "driver": {
                            "id": driver_id,
                            "name": driver_name,
                        },
                        "vehicle": {
                            "id": shuttle_id,
                        },
                    }
                )

        return {
            "data": data,
            "pagination": {"hasNextPage": False, "endCursor": "fake-token-next"},
        }
