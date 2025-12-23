# Architecture Guide

Technical architecture and project structure for Shubble.

## Project Overview

Shubble is a real-time shuttle tracking application for RPI shuttles. It integrates with the Samsara GPS API to track vehicle locations and matches them to scheduled routes using geofencing and algorithmic schedule matching.

**Tech Stack:**
- Frontend: React 19 + TypeScript + Vite + Apple MapKit JS
- Backend: Flask (Python) + SQLAlchemy + PostgreSQL
- Cache: Redis for performance optimization
- Worker: Background polling service for GPS updates
- Deployment: Docker containers (Dokploy/Dokku)

## Directory Structure

```
shubble/
├── backend/              # Flask backend application
│   ├── __init__.py      # Flask app factory, db/cache/migrations initialization
│   ├── routes.py        # API endpoints
│   ├── models.py        # SQLAlchemy database models
│   ├── worker.py        # Background GPS polling service
│   ├── config.py        # Configuration from environment variables
│   └── time_utils.py    # Campus timezone utilities
│
├── frontend/            # React frontend application
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   ├── ts/          # TypeScript utilities
│   │   └── data/        # Static data (copied from /data during build)
│   ├── public/          # Static assets
│   └── dist/            # Build output (generated)
│
├── data/                # Static data and algorithms
│   ├── schedules.py     # Schedule matching algorithm (Hungarian)
│   ├── stops.py         # Route and stop definitions
│   ├── parseSchedule.js # Build-time schedule parser (Node.js)
│   ├── schedule.json    # Master schedule (manually edited)
│   ├── aggregated_schedule.json  # Generated schedule (do not edit)
│   └── routes.json      # Route polylines and stop coordinates
│
├── testing/             # Testing infrastructure
│   ├── tests/           # Pytest test suite (automated tests)
│   │   ├── conftest.py
│   │   ├── test_api_endpoints.py
│   │   ├── test_models.py
│   │   ├── test_worker.py
│   │   └── integration/
│   │       └── test_shuttle_workflow.py
│   ├── test-server/     # Mock Samsara API backend (development tool)
│   │   ├── server.py
│   │   └── shuttle.py
│   └── test-client/     # UI for controlling test shuttles
│
├── docker/              # Docker configuration
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── Dockerfile.worker
│   ├── Dockerfile.test-server
│   └── Dockerfile.test-client
│
├── docs/                # Documentation
│   ├── ARCHITECTURE.md  # This file - project structure
│   ├── INSTALLATION.md  # Setup instructions
│   ├── DEVELOPMENT.md   # Development commands and workflows
│   ├── DEPLOYMENT.md    # Production deployment guide
│   ├── TESTING.md       # Automated testing documentation
│   └── OVERVIEW.md      # Project overview
│
├── migrations/          # Database migrations (Alembic/Flask-Migrate)
├── shubble.py          # Main entry point for Flask app
├── requirements.txt    # Python dependencies
├── package.json        # Node.js dependencies and scripts
├── docker-compose.yml  # Docker services configuration
├── .env.example        # Example environment configuration
└── PORTS.md            # Port reference documentation
```

## System Architecture

### Data Flow

```
┌─────────────┐
│   Samsara   │  GPS Provider
│   GPS API   │
└──────┬──────┘
       │ Webhooks (geofence events)
       │ Polling (location data every 5s)
       ↓
┌──────────────────────────────────────┐
│           Backend Services           │
├──────────────┬───────────────────────┤
│  Flask API   │  Background Worker    │
│  (Routes)    │  (GPS Polling)        │
└──────┬───────┴──────┬────────────────┘
       │              │
       ↓              ↓
┌─────────────────────────────────┐
│     PostgreSQL + Redis          │
│  (Storage + Cache)              │
└─────────────┬───────────────────┘
              │
              ↓ API polling (every 5s)
       ┌──────────────┐
       │   React UI   │
       │  (MapKit JS) │
       └──────────────┘
```

### Component Responsibilities

#### 1. Geofence Events (Webhooks)
- Samsara sends POST to `/api/webhook` when vehicles enter/exit campus geofence
- Backend stores events in `geofence_events` table
- Cache `vehicles_in_geofence` is invalidated

#### 2. Location Polling (Worker Process)
- Background worker (`backend/worker.py`) runs infinite loop with 5-second sleep
- Queries vehicles currently in geofence (latest event = `geofenceEntry`)
- Fetches GPS data from Samsara API for active vehicles only
- Stores location data in `vehicle_locations` table
- Updates driver-vehicle assignments from Samsara API
- Invalidates `vehicle_locations` and `schedule_entries` cache

#### 3. Schedule Matching (Algorithm)
- `data/schedules.py` contains Hungarian algorithm implementation
- Matches real-time shuttle positions to scheduled routes based on:
  - Stop detection (vehicle within threshold of known stop)
  - Time-based scheduling from `data/schedule.json`
- Results cached in Redis with key `schedule_entries` (1 hour TTL)
- Recomputed when cache expires or when invalidated by worker

#### 4. Client Updates
- Frontend polls `/api/locations` every 5 seconds
- Receives vehicle positions with route names
- Animates shuttle movement between polling intervals using heading/speed
- MapKit displays shuttles on route polylines with direction indicators

## Database Schema

Located in `backend/models.py`:

### Tables

**Vehicle**
- Represents GPS-tracked shuttles
- Fields: id, name, license_plate, vin, gateway info
- Primary key: id (string)

**GeofenceEvent**
- Entry/exit events for campus boundary
- Fields: id, vehicle_id, event_type, event_time, latitude, longitude
- Indexed on: vehicle_id, event_time

**VehicleLocation**
- GPS coordinates with timestamp
- Fields: id, vehicle_id, latitude, longitude, heading, speed, timestamp
- Indexed on: vehicle_id, timestamp

**Driver**
- Driver records from Samsara API
- Fields: id, name
- Primary key: id (string)

**DriverVehicleAssignment**
- Tracks driver-vehicle assignments over time
- Fields: id, driver_id, vehicle_id, start_time, end_time
- Null end_time indicates active assignment

## Backend Components

### API Endpoints (`backend/routes.py`)

- `GET /api/locations` - Current vehicle positions with route matching
- `POST /api/webhook` - Samsara webhook handler for geofence events
- `GET /api/matched-schedules` - Schedule algorithm results (cached)
- `GET /api/today` - All location data for current day
- `GET /api/routes` - Route polylines from JSON
- `GET /api/schedule` - Schedule data from JSON

### Worker Service (`backend/worker.py`)

Background service that:
- Runs in infinite loop with 5-second intervals
- `update_locations(app)` - Fetches GPS data for vehicles in geofence
- `update_driver_assignments(app, vehicle_ids)` - Syncs driver assignments
- `get_vehicles_in_geofence()` - Cached query returning set of vehicle IDs

### Configuration (`backend/config.py`)

Loads settings from environment variables:
- Database and Redis connection strings
- Service URLs for CORS and port binding
- API credentials for Samsara
- Flask environment and debug settings
- Campus timezone (hardcoded to America/New_York)

### Time Utilities (`backend/time_utils.py`)

- `get_campus_start_of_day()` - Returns 4am ET as "start of day"
- Shuttle service day boundary (overnight shuttles count toward same day)

## Data Processing Components

### Schedule Algorithm (`data/schedules.py`)

Hungarian algorithm for matching shuttles to scheduled routes:

- `match_shuttles_to_schedules()` - Main algorithm entry point
  1. Loads all vehicle locations from today since 4am ET
  2. Labels each location with (route_name, stop_name) using proximity
  3. Matches vehicle stop visits to scheduled stop times
  4. Computes cost matrix based on time differences
  5. Uses scipy's `linear_sum_assignment` for optimal assignment
  6. Returns matched schedules with confidence scores

- `load_and_label_stops()` - Labels location data with stop information
- Uses extensive Redis caching for performance

### Route and Stop Definitions (`data/stops.py`)

Hardcoded coordinate data for all routes and stops:

- `Stops.get_closest_point(coords)` - Find closest point on route polylines
- `Stops.is_at_stop(coords)` - Returns (route_name, stop_name) or (None, None)
- `Stops.active_routes` - Set of currently active route names
- Contains polyline coordinates and stop positions for all shuttle routes

### Schedule Parser (`data/parseSchedule.js`)

Build-time Node.js script:
- Parses `schedule.json` into `aggregated_schedule.json`
- Groups schedules by day of week and route
- Runs automatically before `npm run dev` and `npm run build`
- Output format: Array indexed by day (0=Sunday), containing route → times mapping

## Frontend Components

### Application Structure

**App.tsx**
- React Router setup
- Route definitions for pages

**Pages:**
- `pages/LiveLocation.tsx` - Main page combining map + schedule
- `pages/Data.tsx` - Data analysis and historical views

**Components:**
- `components/MapKitMap.tsx` - Apple MapKit integration
  - Displays routes, stops, and animated shuttle positions
  - Handles real-time updates and smooth transitions
  - Uses heading/speed for prediction between GPS updates
  - Implements polyline snapping for accurate route following

- `components/Schedule.tsx` - Schedule display component
- `components/DataAgeIndicator.tsx` - Shows age of location data

**Utilities:**
- `ts/api.ts` - API wrapper and endpoint configuration
- `ts/config.ts` - Frontend configuration
- `ts/mapUtils.ts` - Map calculation utilities (bearing, distance, etc.)
- `ts/types/` - TypeScript type definitions

## Static Data Files

### schedule.json (Master Schedule)

Manually edited file defining shuttle schedules:

```json
{
  "MONDAY": "BUS_SCHEDULE_A",
  "BUS_SCHEDULE_A": {
    "BUS_1": [
      ["10:00", "ROUTE_NAME"],
      ["10:30", "ANOTHER_ROUTE"]
    ]
  }
}
```

### aggregated_schedule.json (Generated)

**Do not edit directly** - generated by parseSchedule.js:
- Array indexed by day of week (0=Sunday)
- Each day contains routes mapped to arrays of times
- Optimized format for runtime lookups

### routes.json

Route polylines and stop coordinates:
- Lat/lon arrays for drawing routes on map
- Stop locations for proximity detection
- Polyline stop ordering for route following

## Caching Strategy

### Redis Cache Keys

**vehicle_locations** (5 min TTL)
- Current vehicle positions from latest worker poll

**vehicles_in_geofence** (5 min TTL)
- Set of active vehicle IDs currently on campus

**schedule_entries** (1 hour TTL)
- Matched schedule algorithm results

**coords:{lat}:{lon}** (24 hour TTL)
- Stop detection results for coordinate pairs

**labeled_stops:{date}** (10 min TTL)
- Daily stop visit data with labels

### Cache Invalidation

Triggers for cache invalidation:
- New geofence events → `vehicles_in_geofence`, `vehicle_locations`
- New location data → `vehicle_locations`, `schedule_entries`
- Manual clear via Redis CLI when needed

## Testing Infrastructure

### Automated Tests

**Frontend Tests** (`frontend/src/test/`)
- Framework: Vitest
- Component tests with React Testing Library
- Integration tests for API interaction
- Pattern: `*.test.tsx` or `*.test.ts`

**Backend Tests** (`testing/tests/`)
- Framework: Pytest
- Unit tests for models, utilities, algorithms
- Integration tests for API endpoints
- End-to-end workflow tests
- Pattern: `test_*.py`

### Mock Samsara API (Development Tool)

**Test Server** (`testing/test-server/`)
- Flask API simulating Samsara GPS endpoints
- Generates mock geofence events and GPS data
- Controls simulated shuttle movement
- Useful for development without API credentials

**Test Client** (`testing/test-client/`)
- React UI for controlling mock shuttles
- Create/manage simulated vehicles
- Trigger state changes and movement
- Monitor events in real-time

## Configuration & Environment

### Environment Variables

**Required:**
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string

**Service URLs:**
- `FRONTEND_URL` - Frontend URL (for CORS)
- `VITE_BACKEND_URL` - Backend URL (for CORS and port binding)
- `TEST_FRONTEND_URL` - Test client URL (for CORS)
- `VITE_TEST_BACKEND_URL` - Test server URL

**Optional (Production):**
- `API_KEY` - Samsara API key
- `SAMSARA_SECRET` - Base64-encoded webhook secret

**Flask Settings:**
- `FLASK_ENV` - development or production
- `FLASK_DEBUG` - true or false
- `LOG_LEVEL` - INFO, DEBUG, WARNING, ERROR

## Deployment Architecture

### Docker Services

**backend** - Flask API server
- Runs gunicorn with multiple workers
- Automatically runs database migrations on startup
- Health check endpoint

**worker** - Background GPS poller
- Polls Samsara API every 5 seconds
- Updates database with latest positions
- No HTTP interface

**frontend** - React application
- nginx serving static files

- Gzip compression and caching headers

**postgres** - PostgreSQL database
- Persistent volume for data
- Shared by backend, worker, and test services

**redis** - Redis cache
- Persistent volume with AOF
- Shared by all backend services

**test-server** (optional, --profile dev)
- Mock Samsara API for development
- Shares database with main app

**test-client** (optional, --profile dev)
- UI for controlling mock server


## Important Implementation Details

### Schedule Matching Algorithm

The Hungarian algorithm matches vehicles to routes by:
1. Creating cost matrix of (vehicle, route) pairs
2. Cost = time difference between observed and scheduled stops
3. Optimal assignment minimizes total cost
4. Handles missing data and partial routes gracefully

### Stop Detection

Vehicle considered "at stop" when:
- Within 0.050 degrees (~5.5km) of stop coordinates
- Closest to that route's polyline
- Results cached per coordinate to avoid recalculation

### Geofence Logic

Only vehicles inside campus geofence are tracked:
- Latest geofence event determines active status
- `geofenceEntry` = active, poll GPS
- `geofenceExit` = inactive, stop polling
- Minimizes API calls and processing

### Time Handling

All database times in UTC:
- Campus timezone: America/New_York
- "Start of day" = 4am ET (service day boundary)
- Overnight shuttles belong to same service day
- Display times converted to ET for users

## Security Considerations

- **Webhook Validation**: HMAC SHA256 signature verification
- **Docker Security**: Non-root user (uid 1000) in containers
- **Secrets Management**: Environment variables only, never committed
- **CORS**: Configured to allow only known frontend origins
- **Headers**: Security headers in nginx (X-Frame-Options, etc.)
- **WSGI**: Gunicorn for production (not Flask dev server)

## Performance Considerations

- Worker polling interval: 5 seconds (adjustable)
- Frontend polling interval: 5 seconds (adjustable)
- Schedule algorithm: Expensive, cached for 1 hour
- Stop detection: Expensive, cached for 24 hours
- Database queries: Indexed on commonly queried fields
- Redis caching: Reduces database load significantly

## Related Documentation

- [INSTALLATION.md](INSTALLATION.md) - Setup instructions
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development commands and workflows
- [testing/README.md](../testing/README.md) - Testing guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment
- [../PORTS.md](../PORTS.md) - Port reference
- [../README.md](../README.md) - Project overview
