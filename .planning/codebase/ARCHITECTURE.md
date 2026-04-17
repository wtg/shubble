# Architecture

**Analysis Date:** 2026-04-05

## Pattern Overview

**Overall:** Three-tier distributed system with async event-driven backend and interactive frontend

**Key Characteristics:**
- Client-server architecture with React SPA frontend and FastAPI backend
- Background worker process polling external APIs independently
- Redis caching layer for real-time data freshness
- PostgreSQL as source of truth for all transactional data
- Webhook integration for external event processing (Samsara geofence events)
- Async-first design throughout (asyncio, SQLAlchemy async, FastAPI)

## Layers

**Presentation Layer (Frontend):**
- Purpose: Real-time shuttle tracking UI with interactive map and schedule
- Location: `frontend/src/`
- Contains: React components, hooks, type definitions, utility functions
- Depends on: Backend API endpoints (`/api/*`), shared JSON data (routes, schedules)
- Used by: Web browsers, accessed via Vite dev server or Nginx in production

**API Layer (FastAPI Backend):**
- Purpose: HTTP request handling, data transformation, caching, webhook processing
- Location: `backend/fastapi/`
- Contains: Route handlers, request/response serialization, cache decorators
- Depends on: Database (async SQLAlchemy), Redis, shared utilities
- Used by: Frontend, external webhooks (Samsara API)

**Data Access Layer:**
- Purpose: Async database operations, ORM model definitions, connection pooling
- Location: `backend/database.py`, `backend/models.py`
- Contains: SQLAlchemy Base, async engine/session factory, all ORM models
- Depends on: PostgreSQL 17
- Used by: FastAPI routes, background worker

**Worker/Background Processing Layer:**
- Purpose: Periodic polling of external APIs, asynchronous data ingestion
- Location: `backend/worker/`
- Contains: Long-running async tasks, Samsara API client, location update logic
- Depends on: Database, Redis, external API (Samsara or mock test server)
- Used by: Container orchestration (docker-compose) as separate service

**Shared Utilities Layer:**
- Purpose: Cross-layer utilities, route matching, schedule processing
- Location: `backend/config.py`, `backend/utils.py`, `backend/time_utils.py`, `shared/`
- Contains: Configuration, timezone handling, geofence queries, shared JSON data
- Depends on: Pydantic settings, database models
- Used by: All other layers

## Data Flow

**Live Location Update Flow:**

1. **Worker polls Samsara API** (`backend/worker/worker.py`)
   - Every 30 seconds (configurable), worker calls Samsara `/fleet/vehicles/stats`
   - Filters by vehicles currently in geofence (using cached query)
   - Paginated response handling with `after` token

2. **Worker inserts to PostgreSQL**
   - Writes `VehicleLocation` records with timestamp, lat, lon, speed, heading
   - Unique constraint on (vehicle_id, timestamp) prevents duplicates
   - Indexes on (vehicle_id, timestamp desc) for fast lookups

3. **Frontend fetches via `/api/locations` endpoint**
   - GET request triggers cache decorator with 15s soft TTL, 300s hard TTL
   - Cache miss queries database for latest location per vehicle in geofence
   - Joins with `DriverVehicleAssignment` to attach driver info
   - Returns dict with location, vehicle, driver, timestamp, data latency

4. **Map component renders vehicles**
   - LiveLocationMapKit component receives location data
   - Uses MapKit JS to display markers with custom shuttle icons
   - Updates in near-real-time as new API responses arrive

**Webhook Event Flow (Geofence Events):**

1. **Samsara pushes geofence event** to `POST /api/webhook`
   - HMAC-SHA256 signature verification using `SAMSARA_SECRET`
   - Event contains: vehicle_id, event_type (geofenceEntry/Exit), timestamp, address

2. **Backend stores in `geofence_events` table**
   - Indexed by (vehicle_id, event_time) for fast historical lookups
   - Tracks service area entry/exit for fleet management

3. **Worker queries geofence status**
   - `get_vehicles_in_geofence()` returns only vehicles with latest "geofenceEntry" event today
   - Cached for 15 minutes to reduce database load

**ETA Calculation Flow:**

1. **Backend generates predictions** (`backend/worker/data.py`)
   - ML pipeline processes historical location data
   - Generates LSTM/OFFSET predictions for ETA per stop
   - Stores in `etas` table indexed by (vehicle_id, timestamp)

2. **Frontend fetches via `/api/etas` endpoint**
   - Returns per-stop ETA in JSON format: `{stop_key: {eta: ISO-string, route: name}}`
   - Hooks automatically format timestamps for display
   - Used by both Schedule and Map components

**Schedule Data Flow:**

1. **Static schedule loaded from JSON**
   - `shared/schedule.json` contains day/route/time schedule
   - `shared/aggregated_schedule.json` compiled version for quick filtering
   - Frontend loads at startup, filters by active routes

2. **Routes and polylines loaded**
   - `shared/routes.json` contains route definitions, colors, polylines, stops
   - Both frontend (map rendering) and backend (route matching) use this data

3. **Route matching via Haversine**
   - `shared/stops.py` calculates closest route polyline to GPS point
   - Uses vectorized numpy operations for efficiency
   - Returns route name if unambiguous, None if multiple routes too close

**State Management:**

- **Frontend**: React hooks for local UI state, localStorage for selected route
- **Backend**: Redis for cache (locations, geofence queries), PostgreSQL for persistence
- **Worker**: In-memory state during runtime, references database/Redis for querying
- **Timezone handling**: All timestamps stored in UTC, conversions at layer boundaries via `time_utils.py`

## Key Abstractions

**Vehicle Location Abstraction:**
- Purpose: Represent current position of a shuttle in service
- Examples: `backend/models.py:VehicleLocation`, `frontend/src/types/vehicleLocation.ts`
- Pattern: ORM model on backend, TypeScript interface on frontend, shared through JSON API

**Geofence Event Abstraction:**
- Purpose: Track when shuttles enter/exit service area boundary
- Examples: `backend/models.py:GeofenceEvent`
- Pattern: Webhook-triggered database writes, used for filtering active vehicles

**ETA Abstraction:**
- Purpose: Predicted arrival time at specific stops
- Examples: `backend/models.py:ETA`, `frontend/src/hooks/useStopETAs`
- Pattern: ML-generated predictions stored in database, served via REST endpoint, cached

**Route/Schedule Abstraction:**
- Purpose: Static route definitions and service schedule
- Examples: `shared/routes.json`, `shared/schedule.json`, `shared/stops.py`
- Pattern: Static JSON files loaded at startup, shared between frontend and backend

**Driver Assignment Abstraction:**
- Purpose: Link drivers to vehicles over time periods
- Examples: `backend/models.py:DriverVehicleAssignment`
- Pattern: Temporal relationship tracking start/end times, queried for current assignments

## Entry Points

**Backend Entry Point (`shubble.py`):**
- Location: `/c/Users/Jzgam/OneDrive/Documents/GitHub/shubble/shubble.py`
- Triggers: `uvicorn shubble:app --reload` (local) or container startup
- Responsibilities: Exports FastAPI app instance for ASGI servers
- Initialization: Triggers `backend/fastapi/__init__.py:create_app()` which sets up lifespan, CORS, middleware, routes

**Worker Entry Point (`backend/worker/__main__.py`):**
- Location: `backend/worker/__main__.py`
- Triggers: `python -m backend.worker` or Docker worker container
- Responsibilities: Runs async event loop with periodic polling tasks
- Initialization: Connects to database/Redis, starts infinite loop calling `update_locations()`

**Frontend Entry Point (`frontend/src/main.tsx`):**
- Location: `frontend/src/main.tsx`
- Triggers: `vite dev` (local) or Nginx serving compiled assets (production)
- Responsibilities: Loads config, renders React app
- Initialization: Calls `loadConfig()` to fetch runtime config, then renders `App.tsx` with Router

## Error Handling

**Strategy:** Graceful degradation with logging and user-facing banners

**Patterns:**

- **Backend API errors**: Try-catch blocks log exceptions, return HTTP error status
  - Webhook processing: Logs and continues on error, doesn't fail entire request
  - Example: `backend/fastapi/routes.py:202-331` webhook endpoint catches and logs all exceptions
  
- **Frontend component errors**: ErrorBoundary catches unhandled errors, displays banner
  - Location: `frontend/src/components/ErrorBoundary.tsx`
  - Shows "Something went wrong" message with reload button
  
- **Database/network errors**: Async operations timeout and fail gracefully
  - SQLAlchemy connection pooling with pre-ping prevents stale connections
  - API client timeouts set to 30s
  
- **Missing data**: Endpoints return empty/null gracefully
  - Location endpoint returns empty dict if no vehicles in geofence
  - Schedule endpoint handles missing schedules with 404
  - Frontend displays placeholder UI for missing data

## Cross-Cutting Concerns

**Logging:** Structured logging with component-level configuration
- Each module imports logger: `logger = logging.getLogger(__name__)`
- Log levels configurable per component via `settings.get_log_level(component)`
- Backend, worker, and ML pipeline can have different log levels

**Validation:** Multi-layer validation
- Pydantic settings validation in `backend/config.py`
- Database constraints (unique, foreign key, check) in models
- Frontend TypeScript interfaces for compile-time checking
- API response validation with TypeScript types in frontend

**Authentication:** Webhook signature verification only
- Samsara API calls use Bearer token in Authorization header
- Incoming webhooks validated with HMAC-SHA256 using `SAMSARA_SECRET`
- No user authentication system (public-facing tracker)

**Caching:** Multi-level with TTL management
- Redis cache via decorator in `backend/cache.py`
- Soft TTL (15s): Cache hit returns stale data while refresh happens in background
- Hard TTL (300s): Forces database query if past hard TTL
- Namespaces separate cache keys: "locations", "geofence_vehicles", "smart_closest_point"
- Frontend hooks implement 30s polling with exponential backoff on errors

---

*Architecture analysis: 2026-04-05*
