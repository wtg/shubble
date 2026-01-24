# Shubble - RPI Shuttle Tracker

## Overview

Shubble is a real-time shuttle tracking application for Rensselaer Polytechnic Institute (RPI). The system provides live GPS tracking, route information, and schedules through a modern web interface.

**Tech Stack:**
- Backend: FastAPI (Python 3.13)
- Frontend: React 19 + TypeScript + Vite
- Database: PostgreSQL 17
- Cache: Redis 7
- Maps: Apple MapKit JS
- GPS Data: Samsara API (with mock test server)

---

## Project Structure

```
shuttletracker-new/
├── backend/
│   ├── __init__.py        # Package exports (app, models, utils)
│   ├── config.py          # Pydantic settings (DB, Redis, Samsara API) - shared
│   ├── database.py        # Async SQLAlchemy engine/session - shared
│   ├── models.py          # ORM models (5 tables) - shared
│   ├── utils.py           # Database query helpers - shared
│   ├── time_utils.py      # Timezone utilities - shared
│   │
│   ├── flask/             # FastAPI backend application
│   │   ├── __init__.py   # App factory, CORS, Redis setup
│   │   └── routes.py     # API endpoints
│   │
│   └── worker/            # Background worker package
│       ├── __init__.py   # Package exports
│       ├── __main__.py   # Module entry point
│       └── worker.py     # Background GPS polling worker
│
├── frontend/              # React frontend application
│   ├── src/              # Source code
│   │   ├── main.tsx      # Entry point
│   │   ├── App.tsx       # Router setup, main layout
│   │   ├── types/        # TypeScript interfaces
│   │   ├── components/   # Shared components (Navigation, ErrorBoundary)
│   │   ├── locations/    # Live map page + MapKit components
│   │   ├── schedule/     # Schedule page
│   │   ├── dashboard/    # Data analytics page
│   │   ├── about/        # About page
│   │   └── utils/        # Config, logger, map utilities
│   ├── package.json      # Frontend dependencies and scripts
│   ├── vite.config.ts    # Vite build configuration
│   ├── tsconfig.json     # TypeScript configuration
│   └── eslint.config.js  # ESLint configuration
│
├── shared/                # Shared resources (routes, schedules, utilities)
│   ├── routes.json        # Route polylines, stops, colors (39.5 KB)
│   ├── schedule.json      # Schedule by day/route/time (26.5 KB)
│   ├── aggregated_schedule.json  # Compiled schedule (16.7 KB)
│   ├── announcements.json # System announcements
│   ├── stops.py           # Route matching logic (haversine distance)
│   └── schedules.py       # Schedule analysis with scipy
│
├── alembic/               # Database migrations
│   ├── env.py            # Async migration config
│   └── versions/         # 3 migrations (initial, indices, constraints)
│
├── docker/                # Container configurations
│   ├── Dockerfile.backend
│   ├── Dockerfile.worker
│   ├── Dockerfile.frontend
│   ├── Dockerfile.test/server
│   └── test/client/      # Test client Docker config
│
├── test/                  # Test environment
│   ├── server/           # Mock Samsara API for dev
│   │   ├── server.py     # FastAPI mock server
│   │   └── shuttle.py    # Shuttle simulation
│   ├── client/           # Test frontend
│   └── files/            # Example test files
│
├── .github/workflows/     # CI/CD pipelines
├── shubble.py            # FastAPI entry point
└── docker-compose.yml    # Multi-service orchestration
```

---

## Database Schema

**5 Tables (PostgreSQL with async SQLAlchemy):**

1. **`vehicles`**
   - PK: `id` (Samsara vehicle ID)
   - Fields: name, asset_type, license_plate, VIN, gateway info
   - Relationships: locations, geofence_events, driver_assignments

2. **`vehicle_locations`**
   - PK: `id` (auto-increment)
   - Fields: vehicle_id (FK), timestamp, lat, lon, heading, speed, address
   - Unique constraint: (vehicle_id, timestamp)
   - Index: (vehicle_id, timestamp)

3. **`geofence_events`**
   - PK: `id` (Samsara event ID)
   - Fields: vehicle_id (FK), event_type, event_time, location
   - Index: (vehicle_id, event_time)
   - Tracks when vehicles enter/exit service area

4. **`drivers`**
   - PK: `id` (Samsara driver ID)
   - Fields: name

5. **`driver_vehicle_assignments`**
   - Fields: driver_id (FK), vehicle_id (FK), assignment_start, assignment_end
   - Links drivers to vehicles over time

---

## API Endpoints

**Backend (`routes.py`):**

- `GET /api/locations` - Latest location for each shuttle in geofence
  - Cache: 60 seconds
  - Returns: VehicleLocationData with route name, vehicle/driver metadata
  - Includes data latency calculation

- `GET /api/schedule` - Route schedules
- `GET /api/routes` - Route definitions
- `GET /api/stops` - Stop locations
- `POST /api/webhook/geofence` - Samsara geofence events
- `POST /api/webhook/stats` - Samsara location updates

**Frontend Routes (`App.tsx`):**

- `/` - Live location map (default)
- `/schedule` - Schedule view
- `/about` - About page
- `/data` - Data dashboard
- `/map` - Fullscreen map
- `/generate-static-routes` - Route generation utility

---

## Data Flow

```
┌─────────────────┐
│  React Frontend │ HTTP GET /api/locations
│   (port 3000)   │────────────────────────┐
└─────────────────┘                        │
                                           ▼
┌─────────────────────────────────────────────────┐
│  FastAPI Backend (port 8000)                    │
│  - routes.py serves API                         │
│  - Redis cache (60s TTL for locations)          │
│  - Queries PostgreSQL (async)                   │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│  PostgreSQL (port 5432)                          │
│  - vehicle_locations (GPS data)                  │
│  - geofence_events (in/out service area)         │
└──────────────────────────────────────────────────┘
                      ▲
                      │
┌─────────────────────┴────────────────────────────┐
│  Background Worker (separate container)           │
│  - backend/worker polls Samsara API every N secs │
│  - Fetches GPS for vehicles in geofence          │
│  - Inserts to vehicle_locations table            │
└─────────────────────┬────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│  Samsara API (Production)                        │
│  OR Mock Test Server (Development - port 4000)   │
│  - GET /fleet/vehicles/stats                     │
└──────────────────────────────────────────────────┘
```

---

## Key Components

### Backend

**`backend/config.py`** - Environment-based settings (shared by flask and worker)
- Database URL (postgres:// → postgresql+asyncpg://)
- Redis URL
- Samsara API credentials (base64 decoded)
- Timezone: America/New_York
- Modes: development, staging, production

**`backend/database.py`** - Database infrastructure (shared)
- `Base`: SQLAlchemy declarative base for all models
- `create_async_db_engine()`: Creates async PostgreSQL+asyncpg engine
- `create_session_factory()`: Creates async session maker
- `get_db()`: FastAPI dependency injection for database sessions
- Uses connection pooling and pre-ping for reliability

**`backend/models.py`** - SQLAlchemy ORM models (shared)
- 5 database models: Vehicle, GeofenceEvent, VehicleLocation, Driver, DriverVehicleAssignment
- Async-compatible relationships
- Indexes for performance (vehicle_id, timestamp)

**`backend/utils.py`** - Database query helpers (shared)
- `get_vehicles_in_geofence_query()`: Subquery for active vehicles
- `get_vehicles_in_geofence()`: Cached version (900s TTL)

**`backend/time_utils.py`** - Timezone utilities (shared)
- `get_campus_start_of_day()`: Campus timezone midnight to UTC conversion
- Uses America/New_York timezone

**`backend/fastapi/routes.py`** - FastAPI endpoints
- CORS configured via middleware
- Cache decorator for frequently accessed data
- Webhook signature verification

**`backend/worker/worker.py`** - Background task
- Async location polling from Samsara
- Pagination handling
- Duplicate location filtering
- Environment-aware (test/server vs production)

### Frontend (`frontend/src/`)

**`App.tsx`** - Main component
- React Router setup
- Selected route state management
- Git revision tracking
- Analytics integration

**`locations/LiveLocation.tsx`** - Live tracking
- Fetches from `/api/locations` endpoint
- Real-time vehicle position display
- MapKit integration

**`locations/MapKitMap.tsx`** - Map component
- Apple MapKit JS wrapper
- Shuttle markers with custom icons
- Route polyline rendering
- Data age indicator
- Fullscreen and embedded modes

**`components/Navigation.tsx`** - App navigation
- Header/footer with route selection
- Responsive design

**`utils/config.ts`** - Runtime config
- Backend URL from runtime config.json (or `VITE_MAPKIT_KEY` fallback for local dev)
- Staging vs production detection
- Analytics configuration

### Shared Resources (`shared/`)

**`stops.py`** - Route matching
- Loads `routes.json` polylines
- `get_closest_point(origin_point)` - Haversine distance to find route
- Returns: (distance, coords, route_name, polyline_index)
- Handles ambiguous proximity (returns None if routes too similar)

**`schedules.py`** - Schedule analysis
- Redis caching for coordinate lookups (24h TTL)
- scipy's linear_sum_assignment for optimization
- Labels vehicle locations with stop info

**`routes.json`** - Route definitions
- Per-route: COLOR, STOPS, POLYLINE_STOPS, ROUTES
- Used by backend (route matching) and frontend (map rendering)

---

## Docker Services

**docker-compose.yml profiles:**

**Backend Profile:**
- `postgres`: PostgreSQL 17 with persistent volume
- `redis`: Redis 7 with AOF persistence
- `backend`: FastAPI server (2 uvicorn workers)
- `worker`: Background GPS poller

**Frontend Profile:**
- `frontend`: Nginx serving React build

**Test Profile:**
- `test/server`: Mock Samsara API (port 4000)
- `test/client`: Test frontend (port 5174)

**Health Checks:**
- Backend: HTTP GET /api/locations
- Worker: Process running check
- Postgres/Redis: Native health checks

---

## Environment Variables

**Key variables (from `.env.example`):**

```bash
# Service URLs
FRONTEND_URLS=http://localhost:3000
BACKEND_URL=http://localhost:8000

# Database
DATABASE_URL=postgresql://shubble:shubble@postgres:5432/shubble

# Cache
REDIS_URL=redis://redis:6379

# Samsara API
API_KEY=sms_live_...
SAMSARA_SECRET=...

# Environment
DEPLOY_MODE=development
```

---

## Development Workflow

**Start all services:**
```bash
docker-compose --profile backend --profile frontend up
```

**Start with test server:**
```bash
docker-compose --profile test --profile backend up
```

**Frontend development:**
```bash
cd frontend
npm install
npm run dev  # Parses schedule, copies data, runs Vite
```

**Backend development:**
```bash
uv sync
uv run alembic -c backend/alembic.ini upgrade head
uv run uvicorn shubble:app --reload
```

**Database migrations:**
```bash
uv run alembic -c backend/alembic.ini revision --autogenerate -m "description"
uv run alembic -c backend/alembic.ini upgrade head
```

---

## Important Files

| File | Purpose |
|------|---------|
| `shubble.py` | FastAPI app entry point |
| `backend/config.py` | Shared configuration (settings) |
| `backend/database.py` | Shared database infrastructure |
| `backend/models.py` | Shared ORM models (database schema) |
| `backend/utils.py` | Shared database query utilities |
| `backend/time_utils.py` | Shared timezone utilities |
| `backend/fastapi/__init__.py` | App factory, middleware, Redis |
| `backend/fastapi/routes.py` | API endpoints |
| `backend/worker/worker.py` | GPS polling worker |
| `frontend/src/App.tsx` | Frontend router/layout |
| `frontend/src/locations/LiveLocation.tsx` | Live tracking page |
| `frontend/package.json` | Frontend dependencies and scripts |
| `frontend/vite.config.ts` | Frontend build config |
| `shared/routes.json` | Route polylines/stops/colors |
| `shared/stops.py` | Route matching algorithm |
| `docker-compose.yml` | Service orchestration |
| `alembic.ini` | Migration config |

---

## Testing

**Mock API Server (`test/server/`):**
- Simulates Samsara API for development
- Provides realistic vehicle movement
- No external API keys needed
- Reads real route polylines from `shared/`

**Test Client (`test/client/`):**
- Separate Vite frontend for testing
- Uses test/server backend (port 4000)

**CI/CD (`.github/workflows/`):**
- Build validation
- Schedule data linting
- YAML validation
- Deployment pipelines (staging, production, Dokku)

---

## Security

**Authentication:**
- No user authentication currently implemented
- Webhook signature verification using `SAMSARA_SECRET`

**CORS:**
- Configured to whitelist `FRONTEND_URL` only
- Set in `backend/fastapi/__init__.py`

**Database:**
- Async SQLAlchemy prevents SQL injection
- Pydantic models validate API responses
- Connection pooling with pre-ping health checks

**Secrets:**
- Stored in `.env` files (not committed)
- Base64 encoding for webhook secrets
- Separate test/production API keys

---

## Key Algorithms

**Route Matching (`shared/stops.py`):**
1. Load all route polylines from `routes.json`
2. For each polyline coordinate, calculate haversine distance to GPS point
3. Find minimum distance across all routes
4. Return route name if distance is unambiguous
5. Return None if multiple routes are too close (ambiguous)

**Schedule Assignment (`shared/schedules.py`):**
1. Load vehicle locations from database
2. Load scheduled stops from `schedule.json`
3. Use scipy's `linear_sum_assignment` to optimize vehicle-to-stop matching
4. Cache results in Redis (24-hour TTL)

**Location Caching (`backend/fastapi/routes.py`):**
1. Check Redis for cached location data (60s TTL)
2. If miss, query PostgreSQL for latest locations
3. Join with geofence events to filter active vehicles
4. Join with driver assignments
5. Calculate route name using `stops.py`
6. Cache result and return

---

## Deployment Architecture

**Production:**
- Multiple Docker containers on single host
- Nginx reverse proxy (frontend container)
- FastAPI backend (2 uvicorn workers)
- Separate worker container for GPS polling
- PostgreSQL + Redis with persistent volumes

**Startup Sequence:**
1. PostgreSQL starts, waits for health check
2. Redis starts, waits for health check
3. Backend runs `alembic upgrade head`, then starts uvicorn
4. Worker starts after backend is healthy
5. Frontend nginx serves static files

**Environment-based Configuration:**
- Development: Uses test/server, detailed logs
- Staging: Real Samsara API, verbose logs
- Production: Real Samsara API, minimal logs

---

## Notes

**Branch Status:**
- Current branch: `split-worker`
- Many client/ files deleted (legacy frontend being removed)
- New structure: `backend/` and `frontend/` directories
- Git status shows migration in progress

**Timezone Handling:**
- Campus timezone: America/New_York
- Backend stores all timestamps in UTC
- `time_utils.py` provides conversion helpers
- Schedule queries use campus midnight in UTC

**Performance Optimizations:**
- Redis caching (60s for locations, 900s for geofence queries)
- Database indexes on (vehicle_id, timestamp) and (vehicle_id, event_time)
- Unique constraint prevents duplicate location records
- Async database queries throughout
- Connection pooling with pre-ping

**Future Considerations:**
- User authentication system
- Admin interface for announcements/schedules
- Real-time WebSocket updates
- Push notifications for delays
- Historical analytics dashboard
