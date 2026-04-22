# Shubble Backend

FastAPI backend for the Shubble shuttle tracking application.

## Tech Stack

- **FastAPI** - Async web framework
- **Python 3.13** - Runtime
- **SQLAlchemy** - Async ORM
- **PostgreSQL 17** - Database
- **Redis 7** - Caching (fastapi-cache)
- **Alembic** - Database migrations

## Project Structure

```
backend/
├── __init__.py             # Package exports
├── config.py               # Pydantic settings (env vars)
├── database.py             # Async SQLAlchemy engine/session
├── models.py               # SQLAlchemy ORM models and table definitions
├── utils.py                # General utility functions
├── time_utils.py           # Timezone utilities
├── cache_dataframe.py      # ML pipeline caching
├── cache.py                # Custom Redis cache
├── function_timer.py       # Execution time decorators
├── alembic.ini             # Alembic configuration
├── alembic/                # Alembic migration scripts
├── fastapi/                # FastAPI application
│   ├── __init__.py         # App factory, CORS, Redis setup
│   ├── routes.py           # API endpoints (10 get, 1 post)
│   └── utils.py            # 11 Data access functions / helpers, and their respective models
└── worker/
    ├── __init__.py         # Worker package
    ├── __main__.py         # Worker entry point
    ├── data.py             # Data prediction utils for worker
    └── worker.py           # Async background worker for fetching vehicle data from Samsara API.
```

## API Endpoints

### Locations

```
GET /api/locations
```

Returns the latest location for each shuttle currently inside the geofence.

**Response Headers:**
- `X-Server-Time` - Current server time (UTC)
- `X-Oldest-Data-Time` - Oldest data point timestamp
- `X-Data-Age-Seconds` - Age of oldest data in seconds

**Response Body:**
```json
{
    "vehicle_id": {
        "name": "Shuttle 1",
        "latitude": 42.7284,
        "longitude": -73.6788,
        "timestamp": "2025-01-15T12:00:00Z",
        "heading_degrees": 180,
        "speed_mph": 25,
        "is_ecu_speed": true,
        "formatted_location": "1761 15th Street, City of Troy, NY, 12180",
        "address_id": "1234567",
        "address_name": "Address Name",
        "license_plate": "RPI123",
        "vin": "1FDFE4FS3KDC07453",
        "asset_type": "vehicle",
        "gateway_model": "VG54NAH",
        "gateway_serial": "GATE-123-WAY",
        "driver": {
            "id": "52558508",
            "name": "John, Doe"
        }
    }
}
```

### Routes & Schedule

```
GET /api/etas                   Returns ETA information for each vehicle currently inside the geofence.
GET /api/routes                 Returns route polylines, stops, and route colors.
GET /api/schedule               Returns shuttle schedules.
GET /api/aggregated-schedule    Returns compiled schedule data.
```

### Data

```
GET /api/locations              Returns the latest location for each shuttle currently inside the geofence.
GET /api/velocities             Returns the latest speed for each shuttle currently inside the geofence.
GET /api/today                  Returns all location and geofence event data for today.
GET /api/historical?start={timestamp}&end={timestamp}
                                Returns historical shuttle location data for the specified time range.
GET /api/matched-schedules      Returns ML-matched shuttle schedules.
GET /api/announcements          Returns active system announcements.
```

### Webhooks

```
POST /api/webhook               Receives Samsara geofence events.
```

## Database Schema

### Tables

1. **vehicles** - Shuttle vehicle metadata
   - `id` (PK) - Samsara vehicle ID
   - `name`, `asset_type`, `license_plate`, `vin`
   - `maintenance_id`, `gateway_model`, `gateway_serial`

2. **vehicle_locations** - GPS location history
   - `id` (PK) - Auto-increment
   - `vehicle_id` (FK), `name`, `timestamp`, `latitude`, `longitude`
   - `heading_degrees`, `speed_mph`, `is_ecu_speed`, `formatted_location`
   - `address_id`, `address_name`, `created_at`
   - Index: `(vehicle_id, timestamp DESC)`
   - Unique constraint: `(vehicle_id, timestamp)`

3. **geofence_events** - Entry/exit events
   - `id` (PK) - Samsara event ID
   - `vehicle_id` (FK), `event_type`, `event_time`
   - `address_name`, `address_formatted`, `latitude`, `longitude`
   - `created_at`
   - Index: `(vehicle_id, event_time)`

4. **drivers** - Driver information
   - `id` (PK) - Samsara driver ID
   - `name`, `created_at`

5. **driver_vehicle_assignments** - Driver-vehicle mapping
   - `id` (PK) - Auto-increment
   - `driver_id` (FK), `vehicle_id` (FK)
   - `assignment_start`, `assignment_end`
   - `created_at`

6. **etas** - Estimated time of arrival data
   - `id` (PK) - Auto-increment
   - `vehicle_id` (FK), `etas` (JSON), `timestamp`
   - `created_at`
   - Index: `(vehicle_id, timestamp)`

7. **predicted_locations** - ML-predicted shuttle locations
   - `id` (PK) - Auto-increment
   - `vehicle_id` (FK), `speed_kmh`, `timestamp`
   - `created_at`
   - Index: `(vehicle_id, timestamp)`

8. **announcements** - System announcements
   - `id` (PK) - Auto-increment
   - `message`, `type`, `active`, `expires_at`
   - `created_at`

### Migrations

```bash
# Create new migration
uv run alembic -c backend/alembic.ini revision --autogenerate -m "description"

# Apply migrations
uv run alembic -c backend/alembic.ini upgrade head

# Rollback
uv run alembic -c backend/alembic.ini downgrade -1
```

## Configuration

Environment variables (via `.env` or environment):

```bash
# Database
DATABASE_URL=postgresql://shubble:shubble@localhost:5432/shubble

# Redis
REDIS_URL=redis://localhost:6379/0

# CORS
# Every unique domain should be comma-separated
FRONTEND_URLS=http://localhost:3000
FRONTEND_URLS=http://localhost:3000,http://localhost:5173

# Samsara API
API_KEY=sms_live_...
SAMSARA_SECRET=base64_encoded_secret

# Logging
LOG_LEVEL=info
DEBUG=false

# Environment
DEPLOY_MODE=development  # development, staging, production
```

## Background Worker

The worker polls the Samsara API for GPS data and stores it in the database.

```bash
# Run worker
uv run python -m backend.worker

# Worker behavior:
# - Polls every 5 seconds
# - Only fetches vehicles currently in geofence
# - Handles pagination for large vehicle fleets
# - Skips duplicate locations (same vehicle + timestamp)
```

## Docker

### Development

```bash
docker build -f docker/backend/Dockerfile.backend.dev -t shubble-backend-dev .
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  shubble-backend-dev
```

### Production

```bash
docker build -f docker/backend/Dockerfile.backend.prod -t shubble-backend .
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  -e API_KEY=sms_live_... \
  shubble-backend
```

## Caching

The backend uses Redis caching via `fastapi-cache`:

```python
from fastapi_cache.decorator import cache

@router.get("/api/locations")
@cache(expire=60, namespace="locations")
async def get_locations():
    ...
```

**Cache TTLs:**
- `/api/locations` - 60 seconds
- Geofence queries - 900 seconds (15 minutes)
- Matched schedules - 3600 seconds (1 hour)

## Development

### Running Tests

```bash
uv run pytest backend/tests/
```

### Code Style

```bash
# Lint and format
uv run ruff check backend/
uv run ruff format backend/
```

### Local Development with Test Server

To develop without Samsara API access:

```bash
# Start test server (provides mock GPS data)
uv run uvicorn test.server.server:app --port 4000

# Configure backend to use test server
export API_KEY=test
export SAMSARA_API_URL=http://localhost:4000
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  FastAPI Backend                                        │
│  - Async request handling                               │
│  - Redis caching layer                                  │
│  - SQLAlchemy ORM                                       │
└─────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────┐
│  PostgreSQL     │          │  Redis          │
│  - Locations    │          │  - Cache        │
│  - Events       │          │  - Sessions     │
│  - Vehicles     │          │                 │
└─────────────────┘          └─────────────────┘
         ▲
         │
┌─────────────────────────────────────────────────────────┐
│  Background Worker                                      │
│  - Polls Samsara API every 5 seconds                    │
│  - Inserts new locations to database                    │
│  - Handles geofence filtering                           │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Samsara API (or Test Server)                           │
│  - GET /fleet/vehicles/stats                            │
│  - Vehicle GPS data                                     │
└─────────────────────────────────────────────────────────┘
```
