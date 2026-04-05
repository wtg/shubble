# Codebase Structure

**Analysis Date:** 2026-04-05

## Directory Layout

```
shubble/
├── backend/                   # Python FastAPI backend + worker
│   ├── __init__.py           # Package initialization
│   ├── alembic/              # Database migration system
│   │   ├── versions/         # Migration files (incrementally versioned)
│   │   ├── env.py            # Async migration configuration
│   │   └── alembic.ini       # Migration settings (from project root)
│   ├── cache.py              # Redis cache decorator and management
│   ├── cache_dataframe.py    # Cached dataframe for ML pipeline output
│   ├── config.py             # Pydantic BaseSettings (shared by backend + worker)
│   ├── database.py           # Async SQLAlchemy engine/session factory
│   ├── models.py             # 10 ORM table definitions
│   ├── function_timer.py     # Timing decorator for profiling
│   ├── time_utils.py         # Timezone conversion utilities
│   ├── utils.py              # Database query helpers (geofence queries)
│   ├── fastapi/              # FastAPI application
│   │   ├── __init__.py       # App factory, lifespan, middleware setup
│   │   ├── routes.py         # 10+ API endpoint definitions
│   │   └── utils.py          # Route-specific utilities (data serialization)
│   └── worker/               # Background GPS polling worker
│       ├── __init__.py       # Package exports
│       ├── __main__.py       # Module entry point for python -m execution
│       ├── worker.py         # Main polling loop, Samsara API client
│       ├── data.py           # ML pipeline, ETA generation
│
├── frontend/                  # React 19 + TypeScript + Vite
│   ├── src/
│   │   ├── main.tsx          # React app entry point
│   │   ├── App.tsx           # Router setup, layout, page routes
│   │   ├── App.css           # Global app styles
│   │   ├── index.css         # Global CSS resets
│   │   ├── globals.d.ts      # Global TypeScript definitions
│   │   ├── vite-env.d.ts     # Vite environment types
│   │   ├── about/            # About page (info/documentation)
│   │   │   ├── About.tsx
│   │   │   ├── TextAnimation.tsx
│   │   │   └── styles/
│   │   ├── components/       # Shared UI components
│   │   │   ├── Navigation.tsx    # Header/footer with route selector
│   │   │   ├── ErrorBoundary.tsx # Error handling wrapper
│   │   │   ├── AnnouncementBanner.tsx
│   │   │   ├── Feedback.tsx
│   │   │   ├── NotFound.tsx
│   │   │   └── styles/
│   │   ├── dashboard/        # Data analytics page
│   │   │   ├── Dashboard.tsx
│   │   │   ├── components/   # DataBoard, ShuttleRow, charts
│   │   │   └── styles/
│   │   ├── hooks/            # React hooks
│   │   │   ├── useStopETAs.ts   # Fetches ETA data, polls every 30s
│   │   │   └── [other hooks]
│   │   ├── locations/        # Live tracking page
│   │   │   ├── LiveLocation.tsx # Main live tracking component
│   │   │   ├── components/   # LiveLocationMapKit, ShuttleIcon, DataAgeIndicator
│   │   │   └── styles/
│   │   ├── mapkit/           # Apple MapKit JS utilities
│   │   ├── privacy/          # Privacy policy page
│   │   ├── schedule/         # Schedule view page
│   │   │   ├── Schedule.tsx  # Schedule display with ETAs
│   │   │   └── styles/
│   │   ├── shared/           # Static JSON data (symlink/copy from root shared/)
│   │   │   ├── routes.json   # Route definitions, polylines, colors
│   │   │   ├── schedule.json # Day/route/time schedule
│   │   │   └── aggregated_schedule.json # Compiled schedule
│   │   ├── support/          # App support/help page
│   │   ├── types/            # TypeScript interface definitions
│   │   │   ├── announcement.ts
│   │   │   ├── ClosestStop.ts
│   │   │   ├── route.ts
│   │   │   ├── schedule.ts
│   │   │   ├── vehicleLocation.ts
│   │   └── utils/            # Frontend utilities
│   │       ├── config.ts     # Runtime config loader (fetches config.json)
│   │       ├── logger.ts     # Logging utility
│   │       └── devTime.ts    # Dev time mock for testing
│   ├── public/               # Static assets (favicon, etc)
│   ├── package.json          # Frontend dependencies and build scripts
│   ├── vite.config.ts        # Vite build configuration with API proxy
│   ├── tsconfig.json         # TypeScript configuration
│   └── eslint.config.js      # ESLint rules
│
├── shared/                    # Shared static data and utilities
│   ├── __init__.py           # Python package
│   ├── routes.json           # Route definitions (39.5 KB)
│   │   - Per-route: COLOR, STOPS, POLYLINES, ROUTES, POLYLINE_STOPS
│   ├── schedule.json         # Day/route/time schedule (27 KB)
│   ├── aggregated_schedule.json # Compiled schedule (16.7 KB)
│   ├── announcements.json    # System announcements (JSON)
│   ├── stops.py              # Route matching: haversine distance to polylines
│   ├── schedules.py          # Schedule analysis, scipy linear_sum_assignment
│   ├── parseSchedule.js      # Node script to parse schedule
│   └── timeUtils.js          # JavaScript timezone utilities
│
├── test/                      # Test environment (development only)
│   ├── server/               # Mock Samsara API (port 4000)
│   │   ├── server.py         # FastAPI mock endpoints
│   │   ├── shuttle.py        # Shuttle simulator with realistic movement
│   │   └── replay.py         # Replay logged data
│   ├── client/               # Test frontend (port 5174, legacy)
│   ├── files/                # Test data files
│   │   └── shubble-sample.csv
│
├── docker/                    # Dockerfile definitions
│   ├── backend/              # FastAPI server
│   │   ├── Dockerfile.backend.dev
│   │   ├── Dockerfile.backend.prod
│   ├── worker/               # Background worker
│   │   └── Dockerfile.worker
│   ├── frontend/             # React + Nginx
│   │   ├── Dockerfile.frontend.dev
│   │   ├── Dockerfile.frontend.prod
│   │   ├── nginx.conf        # Nginx configuration
│   └── test-server/          # Mock API
│       └── Dockerfile
│
├── alembic/                   # Database migrations (legacy location, use backend/alembic/)
│   └── versions/
│
├── ml/                        # Machine learning models
│   ├── cache/                # Cached predictions (LSTM, ARIMA)
│   └── [notebooks, scripts]
│
├── .github/workflows/         # CI/CD pipelines
│   ├── validate-schedule.yml
│   ├── docker-build.yml
│   ├── deploy-*.yml
│
├── docs/                      # Project documentation
│   ├── architecture.md
│   └── more/
│
├── docker-compose.yml         # Multi-service orchestration (backend, frontend, test profiles)
├── shubble.py                # FastAPI app entry point
├── CLAUDE.md                 # Project instructions (this file)
├── package.json              # Root-level package.json (if exists)
└── pyproject.toml            # Python project configuration (uv/pip)
```

## Directory Purposes

**backend/:**
- Purpose: Python FastAPI backend and background worker
- Contains: API routes, ORM models, database config, async workers
- Key files: `fastapi/routes.py` (endpoints), `models.py` (database schema), `worker/worker.py` (polling)

**backend/fastapi/:**
- Purpose: FastAPI ASGI application
- Contains: Route handlers, request/response logic, middleware
- Key files: `routes.py` (all endpoints), `__init__.py` (app factory)

**backend/worker/:**
- Purpose: Async background tasks
- Contains: Samsara API polling, location data ingestion, ETA generation
- Key files: `worker.py` (polling loop), `data.py` (ML predictions)

**frontend/src/:**
- Purpose: React TypeScript frontend source
- Contains: Pages, components, hooks, utilities, type definitions
- Key files: `App.tsx` (router), `main.tsx` (entry), `locations/LiveLocation.tsx` (main page)

**frontend/src/types/:**
- Purpose: Shared TypeScript interface definitions
- Contains: Type definitions for vehicles, routes, schedules, ETAs
- Used by: All frontend components for compile-time type checking

**frontend/src/hooks/:**
- Purpose: Reusable React hooks for data fetching
- Contains: `useStopETAs` (polls ETA endpoint)
- Pattern: Each hook encapsulates fetch logic, polling, error handling

**shared/:**
- Purpose: Shared static data and utility functions
- Contains: Route polylines, schedule data, route matching algorithm
- Used by: Backend (route matching, schedule analysis), Frontend (map rendering, schedule display)

**test/server/:**
- Purpose: Mock Samsara API for development
- Contains: FastAPI server with fake vehicle movement simulator
- Used when: `DEPLOY_MODE=development` (worker points to localhost:4000)

**docker/:**
- Purpose: Container image definitions
- Contains: Separate Dockerfiles for each service (backend, worker, frontend, test server)
- Pattern: Development and production variants

## Key File Locations

**Entry Points:**
- `shubble.py` - FastAPI app, imported by uvicorn
- `backend/worker/__main__.py` - Worker, run with `python -m backend.worker`
- `frontend/src/main.tsx` - React app initialization

**Configuration:**
- `backend/config.py` - Pydantic Settings (shared by all Python services)
- `frontend/src/utils/config.ts` - Runtime config loader
- `docker-compose.yml` - Service orchestration

**Core Logic:**
- `backend/fastapi/routes.py` - 10+ API endpoints (566 lines)
- `backend/models.py` - 10 ORM table definitions (300 lines)
- `backend/worker/worker.py` - Polling loop, API client (368 lines)
- `frontend/src/App.tsx` - Router, page layout
- `shared/stops.py` - Route matching algorithm (Haversine distance)

**Testing:**
- `test/server/server.py` - Mock Samsara API
- `test/server/shuttle.py` - Vehicle simulator
- No pytest suite currently; integration tests via docker-compose

**Database:**
- `backend/alembic/` - Migration system
- `backend/alembic.ini` - Migration config
- `backend/database.py` - Async engine/session factory

## Naming Conventions

**Files:**
- Python: `snake_case.py` (e.g., `config.py`, `time_utils.py`)
- React: `PascalCase.tsx` for components (e.g., `LiveLocation.tsx`), `camelCase.ts` for utilities
- JSON: `kebab-case.json` or `snake_case.json` (e.g., `routes.json`, `aggregated_schedule.json`)

**Directories:**
- Backend: `snake_case/` (e.g., `backend/`, `fastapi/`, `worker/`)
- Frontend: `camelCase/` or `lowercase/` (e.g., `src/`, `components/`, `locations/`)
- Shared: `lowercase/` (e.g., `shared/`, `docker/`, `test/`)

**Functions/Methods:**
- Python: `snake_case()` (e.g., `get_locations()`, `update_locations()`)
- TypeScript: `camelCase()` (e.g., `useStopETAs()`, `loadConfig()`)

**Classes:**
- Python: `PascalCase` (e.g., `Vehicle`, `VehicleLocation`, `GeofenceEvent`)
- TypeScript: `PascalCase` (e.g., `ErrorBoundary`, `DataBoard`)

**Constants:**
- Python: `SCREAMING_SNAKE_CASE` (e.g., `LOG_LEVEL`, `DATABASE_URL`)
- TypeScript/JavaScript: `SCREAMING_SNAKE_CASE` (e.g., `TIME_FORMAT`)

**Database Tables:**
- `snake_case` (e.g., `vehicles`, `vehicle_locations`, `geofence_events`)

## Where to Add New Code

**New API Endpoint:**
- Implementation: `backend/fastapi/routes.py` (add `@router.get()` or `@router.post()`)
- Shared utils: `backend/fastapi/utils.py` (add helper functions if needed)
- Models: Extend `backend/models.py` if new database tables needed

**New React Component:**
- Feature-specific: `frontend/src/{feature}/components/` (e.g., `locations/components/`, `schedule/`)
- Shared component: `frontend/src/components/`
- Style file: Adjacent `styles/ComponentName.css` (e.g., `components/styles/Navigation.css`)
- Type definitions: `frontend/src/types/{featureName}.ts`

**New React Hook:**
- Location: `frontend/src/hooks/useFeatureName.ts`
- Pattern: Named export function starting with `use`, returns state and methods
- Usage: Import in components that need data fetching: `const { data } = useFeatureName()`

**New Utility Function:**
- Backend shared: `backend/utils.py` (database/domain logic)
- Backend FastAPI-specific: `backend/fastapi/utils.py` (serialization, response formatting)
- Frontend: `frontend/src/utils/{featureName}.ts` or `frontend/src/utils/helpers.ts`
- Shared Python: `shared/stops.py` or `shared/schedules.py` (domain algorithms)

**New Database Table:**
- Model definition: Add class to `backend/models.py`
- Migration: Create via `uv run alembic -c backend/alembic.ini revision --autogenerate -m "description"`
- Relationships: Use SQLAlchemy relationship() with lazy="raise" to prevent N+1 queries

**New Data Type/Interface:**
- TypeScript: `frontend/src/types/{name}.ts` (export interface)
- Python: Add Pydantic model to `backend/fastapi/utils.py` or specific route

**New Background Task:**
- Location: `backend/worker/worker.py` or new file in `backend/worker/`
- Pattern: Async function called from `run_worker()` event loop
- Access: Use same session_factory and Redis as main polling loop

## Special Directories

**frontend/src/shared/:**
- Purpose: Static JSON data shared with backend
- Generated: No (manually maintained or copied from `../../shared/`)
- Committed: Yes (included in frontend bundle)
- Note: Symlink or copy from project root `shared/` directory

**backend/alembic/versions/:**
- Purpose: Database migration scripts (auto-generated)
- Generated: Yes (via `alembic revision --autogenerate`)
- Committed: Yes (tracks schema changes)
- Pattern: Each file numbered with timestamp prefix and descriptive name

**ml/cache/:**
- Purpose: Cached ML model predictions
- Generated: Yes (generated by ML pipeline during worker execution)
- Committed: No (in `.gitignore`)
- Usage: Loaded by `backend/cache_dataframe.py` to provide preprocessed route data

**docker/ (non-Dockerfile files):**
- Purpose: Container configuration (nginx.conf, etc)
- Generated: No
- Committed: Yes
- Note: Frontend nginx.conf handles SPA routing and proxies /api to backend

**test/server/:**
- Purpose: Local development mock for Samsara API
- Generated: No (provides fixed, realistic vehicle movement)
- Committed: Yes
- Usage: Only used when `DEPLOY_MODE=development`

---

*Structure analysis: 2026-04-05*
