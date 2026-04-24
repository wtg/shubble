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
├── __init__.py           # Package exports
├── config.py             # Pydantic settings (env vars)
├── database.py           # Async SQLAlchemy engine/session
├── models.py             # ORM models (5 tables)
├── utils.py              # Database query helpers
├── time_utils.py         # Timezone utilities
├── alembic.ini           # Alembic configuration
│
├── fastapi/              # FastAPI application
│   ├── __init__.py       # App factory, CORS, Redis setup
│   └── routes.py         # API endpoints
│
└── worker/               # Background worker
    ├── __init__.py       # Package exports
    ├── __main__.py       # Module entry point
    └── worker.py         # GPS polling worker
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
    "route_name": "East Route",
    "polyline_index": 0,
    "driver": {"id": "123", "name": "John Doe"},
    "stop_times": {"stop_name": "2025-01-15T12:05:00Z"},
    "is_at_stop": false,
    "current_stop": null
  }
}
```

### Routes & Schedule

```
GET /api/routes      # Route polylines, stops, colors
GET /api/schedule    # Shuttle schedules
GET /api/aggregated-schedule  # Compiled schedule data
```

### Data

```
GET /api/today       # All location data for today
GET /api/matched-schedules  # ML-matched shuttle schedules
```

### Webhooks

```
POST /api/webhook    # Samsara geofence events
```

## Database Schema

### Tables

1. **vehicles** - Shuttle vehicle metadata
   - `id` (PK) - Samsara vehicle ID
   - `name`, `asset_type`, `license_plate`, `vin`
   - `gateway_model`, `gateway_serial`

2. **vehicle_locations** - GPS location history
   - `id` (PK) - Auto-increment
   - `vehicle_id` (FK), `timestamp`, `latitude`, `longitude`
   - `heading_degrees`, `speed_mph`, `formatted_location`
   - Unique constraint: `(vehicle_id, timestamp)`

3. **geofence_events** - Entry/exit events
   - `id` (PK) - Samsara event ID
   - `vehicle_id` (FK), `event_type`, `event_time`
   - `address_name`, `latitude`, `longitude`

4. **drivers** - Driver information
   - `id` (PK), `name`

5. **driver_vehicle_assignments** - Driver-vehicle mapping
   - `driver_id` (FK), `vehicle_id` (FK)
   - `assignment_start`, `assignment_end`

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

The backend uses two workers: the locations worker and ML worker. The locations worker is designed to run independently from the ML worker. Thus, the locations worker can continue to run even if the ML worker crashes.

### Locations Worker

The locations worker polls the Samsara API for GPS data and stores it in the database.

```bash
# Run locations worker on host
uv run python -m backend.locations_worker.locations_worker

# Run locations worker on Docker
docker compose --profile backend up locations_worker

# Worker behavior:
# - Polls every 5 seconds
# - Only fetches vehicles currently in geofence
# - Handles pagination for large vehicle fleets
# - Skips duplicate locations (same vehicle + timestamp)
```

### ML Worker

The ML worker reads location data from database, runs route matching and ETA predictions, and writes results to the database.

```bash
# Run ML worker on host
uv run python -m backend.ml_worker.ml_worker

# Run ML worker on Docker 
docker compose --profile backend up ml_worker

# Worker behavior:
# - Runs every 5 seconds
# - Predicts ETA for each stop with an LSTM model
# - Predicts vehicle speed with ARIMA
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
│    FastAPI Backend                                      │
│  - Async request handling                               │
│  - Redis caching layer                                  │
│  - SQLAlchemy ORM                                       │
└─────────────────────────────────────────────────────────┘
         │                                             │
         ▼                                             ▼
┌─────────────────┐                           ┌─────────────────┐
│  PostgreSQL     │                           │  Redis          │
│  - Locations    │                           │  - Cache        │
│  - Events       │                           │  - Sessions     │
│  - Vehicles     │                           │                 │
└─────────────────┘                           └─────────────────┘
         ▲                                          ▲          │
         │                                          │          ▼
┌───────────────────────────────────────┐        ┌──────────────────────────────────────┐
│  Locations Worker                     │        │  ML Worker                           │
│  - Polls Samsara API every 5 seconds  │        │  - Update dataframe for current day  │
│  - Inserts new locations to database  │        │  - Generate predictions for ETA and  │
│  - Handles geofence filtering         │        │    velocities                        │
└───────────────────────────────────────┘        └──────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Samsara API (or Test Server)                           │
│  - GET /fleet/vehicles/stats                            │
│  - Vehicle GPS data                                     │
└─────────────────────────────────────────────────────────┘
```
