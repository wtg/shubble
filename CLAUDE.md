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

<!-- GSD:project-start source:PROJECT.md -->
## Project

**Shubble — ETA Accuracy & UX Milestone**

Shubble is a real-time shuttle tracking app for RPI students. It shows live GPS positions on a map and estimated arrival times on a schedule page. This milestone focuses on making the ETA system trustworthy and polished — students should see whether data is live or scheduled, know if a shuttle is early/late, and never be confused by missing data.

**Core Value:** Students trust the ETA numbers. They know if it's a live GPS estimate or a schedule guess, and they can see early/late status at a glance.

### Constraints

- **Tech stack**: React 19, FastAPI, existing Apple MapKit integration — no framework changes
- **Data freshness**: ETAs refresh every 30s; countdown display must stay in sync
- **Mobile-first**: Most students check on phones — UI must work well on small screens
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.13 - Backend, worker, test server, database migrations
- TypeScript 5.9.2 - Frontend type system for React
- JavaScript (Node.js) - Frontend build and runtime
- SQL (PostgreSQL dialect) - Database queries via SQLAlchemy ORM
## Runtime
- Python 3.13 (via `docker/backend/Dockerfile.backend.dev`)
- Node.js 24-alpine (via `docker/frontend/Dockerfile.frontend.dev`)
- **Backend:** uv (Astral, replaces pip/poetry) - Version 0.9+
- **Frontend:** npm (Node Package Manager)
## Frameworks
- **FastAPI** 0.115.0+ - Python backend web framework, async-first
- **React** 19.2.4 - Frontend UI library
- **React Router** 7.13.1 - Client-side routing
- **SQLAlchemy** 2.0.41+ - Python async ORM
- **Alembic** 1.14.0+ - Database migration tool
- **pytest** - Python unit/integration testing framework
- **pytest-asyncio** - Async test support
- **Vite** 7.3.1 - Frontend build tool (React)
- **TypeScript** 5.9.2 - Static type checking for frontend
- **ESLint** 9.39.4 - JavaScript/TypeScript linting
## Key Dependencies
- **httpx** 0.28.1+ - Async HTTP client for Samsara API calls
- **asyncpg** 0.30.0+ - PostgreSQL async driver
- **redis** - Async Redis client for caching
- **pydantic** 2.10.0+ - Python data validation
- **pydantic-settings** 2.7.0+ - Environment configuration management
- **python-dotenv** 1.1.1+ - Load `.env` files into environment
- **uvicorn[standard]** 0.34.0+ - ASGI server
- **brotli-asgi** - Response compression middleware
- **numpy** - Numerical computing
- **pandas** 2.0.0-4.0.0 - Data manipulation (used for schedule caching)
- **scipy** - Scientific computing (schedule optimization via `linear_sum_assignment`)
- **torch** - PyTorch deep learning
- **scikit-learn** - Machine learning utilities
- **matplotlib** - Plotting/visualization
- **seaborn** - Statistical visualization
- **tqdm** 4.67.1+ - Progress bars
- **statsmodels** - Statistical modeling
- **react-icons** 5.6.0 - Icon library
- **@types/apple-mapkit-js-browser** 5.78.1 - TypeScript types for MapKit
- **@vitejs/plugin-react** 5.1.4 - React Fast Refresh for Vite
- **vite-plugin-pwa** 1.2.0 - Progressive Web App support
- **ruff** - Fast Python linter/formatter
- **requests** - HTTP library for testing
## Configuration
- `.env` (root, development) - Primary config via `pydantic-settings`
- `.env.example` - Template (available at: `.env.example`)
- Environment-specific: Set via `docker-compose.yml` environment sections
- `DATABASE_URL` - PostgreSQL connection string (auto-converted to asyncpg driver)
- `REDIS_URL` - Redis cache connection
- `API_KEY` - Samsara API key for production GPS data
- `SAMSARA_SECRET` - Base64-encoded webhook signature verification secret
- `DEPLOY_MODE` - One of: `development`, `staging`, `production`
- `DEBUG` - Enable debug mode and SQL echo
- `LOG_LEVEL` - Global log level (fastapi, worker, ml can override)
- `FRONTEND_URLS` - CORS whitelist (comma-separated)
- `MAPKIT_KEY` - Apple MapKit JS authentication token
- `frontend/vite.config.ts` - Vite build settings, dev server proxy, PWA manifest
- `frontend/tsconfig.json` - TypeScript compiler options
- `pyproject.toml` - Python project metadata and dependency groups
- `docker-compose.yml` - Multi-service orchestration with profiles
## Platform Requirements
- Docker (containers for all services)
- Docker Compose (service orchestration)
- Python 3.13+ (or Docker image)
- Node.js 24+ (or Docker image)
- uv 0.9+ (Python package manager)
- npm 10+ (Node package manager)
- Docker runtime (builds via provided Dockerfiles)
- PostgreSQL 17 (persistent database)
- Redis 7 (persistent cache)
- External: Samsara API (GPS data) or mock test server
- External: Apple MapKit JS (map rendering)
- Development: Linux, macOS, Windows (via WSL2/Docker Desktop)
- Production: Linux (typical cloud/VPS deployment)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Python: `snake_case.py` (e.g., `backend/models.py`, `backend/worker/worker.py`)
- TypeScript/React: `PascalCase.tsx` for components, `camelCase.ts` for utilities (e.g., `frontend/src/components/ErrorBoundary.tsx`, `frontend/src/utils/config.ts`)
- Tests: `test_*.py` prefix for Python, no specific pattern for frontend (tests are minimal)
- Python: `snake_case` (e.g., `update_locations()`, `compute_per_stop_etas()`, `get_latest_etas()`)
- TypeScript: `camelCase` (e.g., `renderToStaticMarkup()`, `pollLocation()`)
- React Hooks: `use` prefix (e.g., `useStopETAs()`, `useEffect()`)
- Python: `snake_case` (e.g., `vehicle_ids`, `session_factory`, `mock_etas`)
- TypeScript: `camelCase` (e.g., `selectedRoute`, `setSelectedRoute`, `vehicleAnnotations`)
- React State: Use getter/setter pairs with `const [state, setState]` pattern
- Python: PascalCase for classes (e.g., `Vehicle`, `VehicleLocation`, `Settings`)
- TypeScript: PascalCase for interfaces/types (e.g., `VehicleLocationData`, `StopETA`, `LiveLocationMapKitProps`)
- Discriminated unions in types (e.g., `StopETA` has required `eta`, `vehicle_id`, `route` fields)
- Python: `UPPER_SNAKE_CASE` for module-level configuration (e.g., `LOG_LEVEL`, `DATABASE_URL`)
- TypeScript: `UPPER_CASE` for true constants (rare; most configuration is imported from `utils/config`)
## Code Style
- Python: Inferred to follow PEP 8 conventions based on project structure (no Prettier/Black config found)
- TypeScript/JavaScript: Likely Prettier-formatted via Vite build pipeline (no `.prettierrc` found, but standard React project setup)
- Python: Ruff available in dev dependencies (`pyproject.toml`), but usage pattern unclear
- TypeScript: ESLint with custom config in `frontend/eslint.config.js` (Flat ESLint format)
- `@typescript-eslint/no-unsafe-member-access`: **error** (strict type safety)
- `@typescript-eslint/no-unsafe-assignment`: **warn** (allows unknown types with warning)
- `@typescript-eslint/no-unsafe-call`: **warn** (allows calling unknown types with warning)
- `@typescript-eslint/no-unused-vars`: **warn** with `argsIgnorePattern: "^_"` (allows intentionally unused params prefixed with `_`)
- `react-hooks/rules-of-hooks`: **error** (enforce Hook Rules)
- `react-hooks/exhaustive-deps`: **warn** (warn on missing dependencies)
- `react/react-in-jsx-scope`: **off** (React 17+ doesn't require explicit import)
## Import Organization
- TypeScript: No path aliases configured in `tsconfig.json` (uses relative imports)
- Shared build process copies `/shared/` into `src/shared/` at build time (see `frontend/package.json` scripts)
## Error Handling
- Async functions use try/except blocks (e.g., in `backend/worker/worker.py` lines 62-90)
- API errors logged with `logger.error()` before returning error response
- HTTP non-200 responses handled explicitly: `if response.status_code != 200: logger.error(...); return []`
- Optional returns: Functions return empty dict `{}` or empty list `[]` on error, not `None` (e.g., `compute_per_stop_etas` returns `{}`)
- Async context managers use `async with` for resource cleanup (e.g., `async with httpx.AsyncClient() as client:`)
- Error Boundary component (`frontend/src/components/ErrorBoundary.tsx`) catches React component errors
- ErrorBoundary uses `getDerivedStateFromError()` and `componentDidCatch()` lifecycle methods
- Error messages logged to console: `console.error('ErrorBoundary caught an error:', error, errorInfo)`
- Graceful UI fallback: Shows error banner with reload button instead of crashing
- Fetch errors in React effects: Use AbortController for cancellation (`new AbortController()`)
- Pydantic models validate all input in Python backend (e.g., `Settings` class in `backend/config.py`)
- Type annotations throughout TypeScript prevent runtime type errors
- No defensive null-checks; rely on TypeScript strict mode
## Logging
- Logger created per module: `logger = logging.getLogger(__name__)`
- Log level configured from environment: `settings.get_log_level(component)` (supports per-component levels)
- Logging setup in module initialization (see `backend/worker/worker.py` lines 20-28, `backend/fastapi/__init__.py` lines 13-22)
- Log messages include context: `logger.error(f"API error: {response.status_code} {response.text}")`
- Lifecycle events logged at startup/shutdown (e.g., "Starting up FastAPI application...", "Database engine initialized")
- `console.error()` for exceptions and boundaries
- `console.log()` for development (no structured logging observed)
## Comments
- **Docstrings required**: All module, function, and class definitions include docstrings
- Module docstrings: Single-line summary (e.g., `"""Async background worker for fetching vehicle data from Samsara API."""`)
- Function docstrings: Args, Returns, purpose (e.g., in `backend/config.py` lines 59-68)
- Inline comments: Used for non-obvious logic (e.g., in `backend/fastapi/routes.py` line 48: `# lazy="raise" prevents accidental N+1 queries`)
- TODO comments: Rare but present (e.g., `frontend/src/locations/LiveLocation.tsx` line 15: `// TODO: figure out how to make this type correct...`)
- React components use TypeScript interfaces for prop documentation (e.g., `interface ErrorBoundaryProps`, `type LiveLocationMapKitProps`)
- No explicit JSDoc comments observed; types serve as documentation
## Function Design
- Python async workers: 15-60 lines for core logic (e.g., `update_locations()` has ~100 lines with pagination loop)
- TypeScript components: 40-120 lines (e.g., `ErrorBoundary` is 47 lines, `LiveLocation` is 49 lines)
- React hooks: Extract complex logic into custom hooks (e.g., `useStopETAs()` for shared ETA fetching)
- Python: Use explicit positional args for required params, `*args`/`**kwargs` avoided
- Python async: Inject dependencies (e.g., `session_factory`, `cache` decorator) rather than global state
- TypeScript: Destructure props in function signature (e.g., `{ routeData, selectedRoute, ...}: LiveLocationMapKitProps`)
- Optional params documented in type interfaces with `?` (e.g., `displayVehicles?: boolean`)
- Python: Functions document return type in type hints (e.g., `async def compute_per_stop_etas(...) -> dict`)
- TypeScript: Return types explicit (e.g., `() => JSX.Element`, `async () => Promise<VehicleLocationMap>`)
- Empty collections preferred over `None`: Return `{}` or `[]` on empty/error (not `null`)
## Module Design
- Python: Explicit imports (no `from backend import *`)
- Backend `__init__.py` files provide package-level exports (e.g., `backend/__init__.py` exports `app`, `models`, `utils`)
- TypeScript: Default exports for components, named exports for utilities/types
- Python: Each package has `__init__.py` with selective re-exports (e.g., `backend/worker/__init__.py` exports `run_worker`)
- TypeScript: No barrel files observed; direct imports used (no `index.ts` re-export pattern)
- Shared data: JSON files imported directly (`import routeData from './shared/routes.json'`)
- FastAPI routes receive `request: Request` and access `request.app.state.session_factory`
- Cache decorator handles Redis connection via `@cache(...)` (see `backend/fastapi/routes.py` line 44)
- Database session passed explicitly: `async def get_locations(...) -> AsyncGenerator[AsyncSession, None]` pattern in `backend/database.py`
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Client-server architecture with React SPA frontend and FastAPI backend
- Background worker process polling external APIs independently
- Redis caching layer for real-time data freshness
- PostgreSQL as source of truth for all transactional data
- Webhook integration for external event processing (Samsara geofence events)
- Async-first design throughout (asyncio, SQLAlchemy async, FastAPI)
## Layers
- Purpose: Real-time shuttle tracking UI with interactive map and schedule
- Location: `frontend/src/`
- Contains: React components, hooks, type definitions, utility functions
- Depends on: Backend API endpoints (`/api/*`), shared JSON data (routes, schedules)
- Used by: Web browsers, accessed via Vite dev server or Nginx in production
- Purpose: HTTP request handling, data transformation, caching, webhook processing
- Location: `backend/fastapi/`
- Contains: Route handlers, request/response serialization, cache decorators
- Depends on: Database (async SQLAlchemy), Redis, shared utilities
- Used by: Frontend, external webhooks (Samsara API)
- Purpose: Async database operations, ORM model definitions, connection pooling
- Location: `backend/database.py`, `backend/models.py`
- Contains: SQLAlchemy Base, async engine/session factory, all ORM models
- Depends on: PostgreSQL 17
- Used by: FastAPI routes, background worker
- Purpose: Periodic polling of external APIs, asynchronous data ingestion
- Location: `backend/worker/`
- Contains: Long-running async tasks, Samsara API client, location update logic
- Depends on: Database, Redis, external API (Samsara or mock test server)
- Used by: Container orchestration (docker-compose) as separate service
- Purpose: Cross-layer utilities, route matching, schedule processing
- Location: `backend/config.py`, `backend/utils.py`, `backend/time_utils.py`, `shared/`
- Contains: Configuration, timezone handling, geofence queries, shared JSON data
- Depends on: Pydantic settings, database models
- Used by: All other layers
## Data Flow
- **Frontend**: React hooks for local UI state, localStorage for selected route
- **Backend**: Redis for cache (locations, geofence queries), PostgreSQL for persistence
- **Worker**: In-memory state during runtime, references database/Redis for querying
- **Timezone handling**: All timestamps stored in UTC, conversions at layer boundaries via `time_utils.py`
## Key Abstractions
- Purpose: Represent current position of a shuttle in service
- Examples: `backend/models.py:VehicleLocation`, `frontend/src/types/vehicleLocation.ts`
- Pattern: ORM model on backend, TypeScript interface on frontend, shared through JSON API
- Purpose: Track when shuttles enter/exit service area boundary
- Examples: `backend/models.py:GeofenceEvent`
- Pattern: Webhook-triggered database writes, used for filtering active vehicles
- Purpose: Predicted arrival time at specific stops
- Examples: `backend/models.py:ETA`, `frontend/src/hooks/useStopETAs`
- Pattern: ML-generated predictions stored in database, served via REST endpoint, cached
- Purpose: Static route definitions and service schedule
- Examples: `shared/routes.json`, `shared/schedule.json`, `shared/stops.py`
- Pattern: Static JSON files loaded at startup, shared between frontend and backend
- Purpose: Link drivers to vehicles over time periods
- Examples: `backend/models.py:DriverVehicleAssignment`
- Pattern: Temporal relationship tracking start/end times, queried for current assignments
## Entry Points
- Location: `/c/Users/Jzgam/OneDrive/Documents/GitHub/shubble/shubble.py`
- Triggers: `uvicorn shubble:app --reload` (local) or container startup
- Responsibilities: Exports FastAPI app instance for ASGI servers
- Initialization: Triggers `backend/fastapi/__init__.py:create_app()` which sets up lifespan, CORS, middleware, routes
- Location: `backend/worker/__main__.py`
- Triggers: `python -m backend.worker` or Docker worker container
- Responsibilities: Runs async event loop with periodic polling tasks
- Initialization: Connects to database/Redis, starts infinite loop calling `update_locations()`
- Location: `frontend/src/main.tsx`
- Triggers: `vite dev` (local) or Nginx serving compiled assets (production)
- Responsibilities: Loads config, renders React app
- Initialization: Calls `loadConfig()` to fetch runtime config, then renders `App.tsx` with Router
## Error Handling
- **Backend API errors**: Try-catch blocks log exceptions, return HTTP error status
- **Frontend component errors**: ErrorBoundary catches unhandled errors, displays banner
- **Database/network errors**: Async operations timeout and fail gracefully
- **Missing data**: Endpoints return empty/null gracefully
## Cross-Cutting Concerns
- Each module imports logger: `logger = logging.getLogger(__name__)`
- Log levels configurable per component via `settings.get_log_level(component)`
- Backend, worker, and ML pipeline can have different log levels
- Pydantic settings validation in `backend/config.py`
- Database constraints (unique, foreign key, check) in models
- Frontend TypeScript interfaces for compile-time checking
- API response validation with TypeScript types in frontend
- Samsara API calls use Bearer token in Authorization header
- Incoming webhooks validated with HMAC-SHA256 using `SAMSARA_SECRET`
- No user authentication system (public-facing tracker)
- Redis cache via decorator in `backend/cache.py`
- Soft TTL (15s): Cache hit returns stale data while refresh happens in background
- Hard TTL (300s): Forces database query if past hard TTL
- Namespaces separate cache keys: "locations", "geofence_vehicles", "smart_closest_point"
- Frontend hooks implement 30s polling with exponential backoff on errors
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
