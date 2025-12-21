# Architecture Guide

Technical architecture and development guide for Shubble.

## Project Overview

Shubble is a real-time shuttle tracking application for RPI shuttles. It integrates with the Samsara GPS API to track vehicle locations and matches them to scheduled routes using geofencing and algorithmic schedule matching.

**Tech Stack:**
- Frontend: React 19 + TypeScript + Vite + Apple MapKit JS
- Backend: Flask (Python) + SQLAlchemy + PostgreSQL
- Cache: Redis
- Worker: Background polling service for GPS updates
- Deployment: Docker containers (Dokploy/Dokku)

## Quick Reference

### Start Development Environment
```bash
# Using Docker (recommended)
docker-compose up -d
docker-compose logs -f

# Native setup
# Terminal 1: Backend
flask --app server:create_app run

# Terminal 2: Frontend
npm run dev

# Terminal 3: Worker
python -m server.worker
```

### Common Commands
```bash
# Database migrations
docker-compose exec backend flask --app server:create_app db upgrade
docker-compose exec backend flask --app server:create_app db migrate -m "description"

# Run tests
npm test                    # Frontend tests
pytest                      # Backend tests
npm run test:coverage       # Frontend with coverage
pytest --cov=server         # Backend with coverage

# Build frontend for production
npm run build

# Lint frontend
npm run lint

# Access database
docker-compose exec postgres psql -U shubble -d shubble

# Access Redis
docker-compose exec redis redis-cli
```

## System Architecture

### Data Flow

```
Samsara API
    ↓ (webhooks + polling)
Backend Worker (every 5s)
    ↓ (writes to DB)
PostgreSQL + Redis
    ↓ (API reads)
Backend API
    ↓ (frontend polls every 5s)
React Frontend
```

#### 1. Geofence Events (Webhooks)
- Samsara sends POST to `/api/webhook` when vehicles enter/exit campus geofence
- Server stores events in `geofence_events` table
- Cache `vehicles_in_geofence` is invalidated

#### 2. Location Polling (Worker Process)
- Background worker (`server/worker.py`) runs infinite loop with 5-second sleep
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

### Database Models

Located in `server/models.py`:

- **Vehicle**: GPS-tracked shuttles (id, name, license_plate, vin, gateway info)
- **GeofenceEvent**: Entry/exit events for campus boundary (event_type, event_time)
- **VehicleLocation**: GPS coordinates with timestamp (latitude, longitude, heading, speed)
- **Driver**: Driver records from Samsara (id, name)
- **DriverVehicleAssignment**: Tracks driver-vehicle assignments (start/end times, null end = active)

### Critical Files and Their Roles

#### Backend
- `server/__init__.py` - Flask app factory, initializes db/cache/migrations
- `server/routes.py` - API endpoints
  - `/api/locations` - Current vehicle positions with route matching
  - `/api/webhook` - Samsara webhook handler for geofence events
  - `/api/matched-schedules` - Schedule algorithm results (cached)
  - `/api/today` - All location data for current day
  - `/api/routes` - Route polylines from JSON
  - `/api/schedule` - Schedule data from JSON

- `server/worker.py` - Background GPS polling service
  - `update_locations(app)` - Fetches GPS data for vehicles in geofence
  - `update_driver_assignments(app, vehicle_ids)` - Syncs driver assignments
  - `get_vehicles_in_geofence()` - Cached query returning set of vehicle IDs
  - Runs in infinite loop with 5-second sleep

- `server/models.py` - SQLAlchemy database models
- `server/config.py` - Configuration from environment variables
- `server/time_utils.py` - Campus timezone utilities
  - `get_campus_start_of_day()` - Returns 4am ET as "start of day" (shuttle service day boundary)

#### Data Processing
- `data/schedules.py` - Schedule matching algorithm
  - `match_shuttles_to_schedules()` - Hungarian algorithm for schedule assignment
  - `load_and_label_stops()` - Labels location data with stop information
  - Uses extensive Redis caching for performance

- `data/stops.py` - Route and stop definitions
  - `Stops.get_closest_point(coords)` - Find closest point on route polylines
  - `Stops.is_at_stop(coords)` - Returns (route_name, stop_name) or (None, None)
  - Contains hardcoded coordinate data for all routes and stops

- `data/parseSchedule.js` - Build-time script (Node.js)
  - Parses `schedule.json` into `aggregated_schedule.json`
  - Groups schedules by day and route for efficient lookup
  - Runs before `npm run dev` and `npm run build` via package.json scripts

#### Frontend
- `client/src/App.tsx` - React Router setup, route definitions
- `client/src/pages/LiveLocation.tsx` - Main page combining map + schedule
- `client/src/components/MapKitMap.tsx` - Apple MapKit integration
  - Displays routes, stops, and animated shuttle positions
  - Handles real-time updates and smooth transitions
  - Uses heading/speed for animation between GPS updates

- `client/src/components/Schedule.tsx` - Schedule display component
- `client/src/ts/config.ts` - Frontend configuration
- `client/src/ts/mapUtils.ts` - Map utility functions

#### Static Data
- `data/schedule.json` - Master schedule (manually edited)
  - Format: `{ "MONDAY": "BUS_SCHEDULE_A", "BUS_SCHEDULE_A": { "BUS_1": [["10:00", "ROUTE_NAME"], ...] } }`

- `data/aggregated_schedule.json` - Generated from schedule.json (do not edit)
  - Array indexed by day of week (0=Sunday)
  - Each day contains routes mapped to arrays of times

- `data/routes.json` - Route polylines and stop coordinates
  - Contains lat/lon arrays for drawing routes on map
  - Stop locations for proximity detection

### Configuration & Environment

Environment variables (`.env`):
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `API_KEY` - Samsara API key for GPS data
- `SAMSARA_SECRET` - Base64-encoded webhook signature secret (optional)
- `FLASK_ENV` - `development` or `production`
- `FLASK_DEBUG` - `true` or `false`
- `LOG_LEVEL` - Logging level (INFO, DEBUG, WARNING, ERROR)

Campus timezone is hardcoded to `America/New_York` in `server/config.py`.

### Caching Strategy (Redis)

Cache keys and TTLs:
- `vehicle_locations` - Current vehicle positions (5 min TTL)
- `vehicles_in_geofence` - Set of active vehicle IDs (5 min TTL)
- `schedule_entries` - Matched schedule results (1 hour TTL)
- `coords:{lat}:{lon}` - Stop detection results (24 hour TTL)
- `labeled_stops:{date}` - Daily stop visit data (10 min TTL)

Cache invalidation triggers:
- New geofence events → invalidate `vehicles_in_geofence`
- New location data → invalidate `vehicle_locations` and `schedule_entries`

### Important Implementation Details

#### Schedule Matching Algorithm
The Hungarian algorithm in `data/schedules.py` works by:
1. Loading all vehicle locations from today since 4am ET
2. Labeling each location with (route_name, stop_name) using proximity detection
3. Matching vehicle stop visits to scheduled stop times
4. Computing assignment cost matrix based on time differences
5. Using scipy's `linear_sum_assignment` to find optimal matching
6. Caching results for 1 hour

#### Stop Detection
A vehicle is considered "at a stop" if:
- Within threshold distance of a stop coordinate (default 0.050 degrees ~5.5km)
- Closest to a specific route polyline

Stop detection is cached per coordinate to avoid repeated calculations.

#### Geofence Logic
- Vehicles are tracked only when inside geofence (on campus)
- Latest geofence event determines if vehicle is active
- If latest event = `geofenceEntry` → vehicle is active
- If latest event = `geofenceExit` → vehicle is inactive
- Worker only polls active vehicles to minimize API calls

#### Time Handling
- All times stored in UTC in database
- Campus timezone (America/New_York) used for display and "day" calculations
- "Start of day" = 4am ET (shuttle service starts late night/early morning)
- This ensures overnight shuttles count towards the same service day

### Testing

#### Unit and Integration Tests

The project uses **Vitest** for frontend testing and **pytest** for backend testing.

**Frontend Tests** (`client/src/test/`)
```bash
# Run tests
npm test

# Watch mode
npm test -- --watch

# With UI
npm run test:ui

# With coverage
npm run test:coverage
```

Test files use the pattern `*.test.tsx` or `*.test.ts`.

**Backend Tests** (`tests/`)
```bash
# Run all tests
pytest

# With coverage
pytest --cov=server --cov-report=html

# Run specific markers
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Exclude slow tests
```

Test files use the pattern `test_*.py`.

#### Mock Samsara API Server (Development Tool)
`test-server/` is a mock Samsara API server for local development (NOT automated tests):
- Simulates vehicle movement along routes
- Returns mock GPS data and geofence events
- Includes Flask API endpoints for controlling simulated shuttles
- Useful for local development without real API credentials

`test-client/` is a React UI for managing the mock server:
- Control shuttle states (entering, looping, exiting)
- Run automated test scenarios from JSON files
- View location and geofence event counts
- Manage multiple simulated shuttles

To use the mock server:
```bash
# Terminal 1: Start mock server
cd test-server
python server.py

# Terminal 2: Start test client UI
cd test-client
npm install
npm run dev
```

Set `FLASK_ENV=development` and leave `API_KEY` empty to use mock server instead of real Samsara API.

#### Manual Testing Endpoints
```bash
# Get current locations
curl http://localhost:8000/api/locations

# Force recompute schedule matching
curl "http://localhost:8000/api/matched-schedules?force_recompute=true"

# Check today's location data
curl http://localhost:8000/api/today
```

### Development Workflow

#### Making Changes to Schedule Data
1. Edit `data/schedule.json` manually
2. Run `node data/parseSchedule.js` to regenerate aggregated_schedule.json
3. Or run `npm run dev` or `npm run build` (runs parseSchedule automatically)
4. Restart backend/worker to pick up changes

#### Making Database Schema Changes
1. Edit `server/models.py`
2. Create migration: `flask --app server:create_app db migrate -m "description"`
3. Review migration file in `migrations/versions/`
4. Apply migration: `flask --app server:create_app db upgrade`

#### Debugging Cache Issues
```bash
# Access Redis
docker-compose exec redis redis-cli

# Check what's cached
KEYS *

# Manually delete cache
DEL schedule_entries
DEL vehicle_locations

# Monitor cache activity
MONITOR
```

#### Debugging Worker Issues
```bash
# Check worker logs
docker-compose logs -f worker

# Common issues:
# - API_KEY not set or invalid
# - No vehicles in geofence (check geofence_events table)
# - Redis connection errors
# - Database connection errors
```

### Performance Considerations

- Worker polls every 5 seconds - adjust in `server/worker.py` if rate limited
- Frontend polls every 5 seconds - adjust in MapKitMap.tsx if needed
- Schedule matching is expensive - cached for 1 hour
- Stop detection is expensive - cached for 24 hours
- Consider increasing cache TTLs if API costs are high
- Consider decreasing polling intervals if real-time accuracy is critical

### Security Notes

- Webhook signature validation uses HMAC SHA256 (Samsara standard)
- Non-root user in Docker containers (uid 1000)
- Secrets should be in environment variables, never committed
- Frontend runs on nginx with security headers
- Backend runs with Gunicorn (production WSGI server)

## Common Development Tasks

### Add a new API endpoint
1. Add route function in `server/routes.py` with `@bp.route()` decorator
2. Add database queries if needed
3. Return JSON with `jsonify()`
4. Test with curl or frontend

### Add a new database table
1. Add model class in `server/models.py` inheriting from `db.Model`
2. Define columns with `db.Column()`
3. Add relationships with `db.relationship()` if needed
4. Create migration and apply

### Add a new route to the map
1. Add polyline coordinates to `data/routes.json`
2. Add stop coordinates with route association
3. Update schedule in `data/schedule.json` if applicable
4. Restart frontend to pick up changes

### Change polling intervals
- Worker: Edit sleep time in `server/worker.py` (default: 5 seconds)
- Frontend: Edit polling interval in `MapKitMap.tsx` (default: 5000ms)
- Cache: Edit TTL values in respective cache.set() calls

## Related Documentation

- [README.md](README.md) - Project overview and quick start
- [INSTALLATION.md](INSTALLATION.md) - Development setup instructions
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment guide
