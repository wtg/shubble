"""Load vehicle location data from database and save to CSV."""
import asyncio
from pathlib import Path
import pandas as pd
from sqlalchemy import select

from backend.config import settings
from backend.database import create_async_db_engine, create_session_factory
from backend.models import VehicleLocation


# CSV file path in the same directory as this script
DATA_DIR = Path(__file__).parent


async def _fetch_vehicle_locations_from_db() -> pd.DataFrame:
    """
    Fetch all vehicle location data from the database.

    Returns:
        DataFrame with columns: vehicle_id, latitude, longitude, timestamp
    """
    # Create database engine and session factory
    engine = create_async_db_engine(settings.DATABASE_URL, echo=False)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        # Query all vehicle locations
        stmt = select(
            VehicleLocation.vehicle_id,
            VehicleLocation.latitude,
            VehicleLocation.longitude,
            VehicleLocation.timestamp
        ).order_by(VehicleLocation.timestamp)

        result = await session.execute(stmt)
        rows = result.fetchall()

    # Close engine
    await engine.dispose()

    # Convert to DataFrame
    df = pd.DataFrame(
        rows,
        columns=['vehicle_id', 'latitude', 'longitude', 'timestamp']
    )

    return df


def load_vehicle_locations() -> pd.DataFrame:
    """
    Fetch all vehicle location data from the database.

    Returns:
        DataFrame with columns: vehicle_id, latitude, longitude, timestamp
    """
    print("Fetching vehicle locations from database...")
    df = asyncio.run(_fetch_vehicle_locations_from_db())

    return df


if __name__ == "__main__":
    # Example usage
    print("Loading vehicle locations (fetching from DB)...")
    df = load_vehicle_locations()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
