# External Integrations

**Analysis Date:** 2026-04-05

## APIs & External Services

**Samsara Fleet Management API:**
- Purpose: Real-time GPS vehicle tracking, geofence events, driver assignments
- SDK/Client: httpx (async HTTP client)
- Authentication: Bearer token via `API_KEY` environment variable
- Endpoints used:
  - `GET https://api.samsara.com/fleet/vehicles/stats` - Vehicle GPS positions
  - Webhooks: `POST /api/webhook/geofence` - Geofence enter/exit events
  - Webhooks: `POST /api/webhook/stats` - Location updates (fallback)
- Webhook verification: HMAC-SHA256 using `SAMSARA_SECRET` (base64 decoded)
- Implementation: `backend/worker/worker.py` polls fleet API; `backend/fastapi/routes.py` receives webhooks
- Mock server: `test/server/` provides fake Samsara API for development (port 4000)
- Environment detection: Dev mode uses mock (port 4000), production uses real API (api.samsara.com)

**Apple MapKit JS:**
- Purpose: Interactive map rendering, route visualization, directions
- Authentication: Token-based (`MAPKIT_KEY` environment variable)
- Initialization: `window.mapkit.init({ token: ... })` in `frontend/src/mapkit/MapKitCanvas.tsx`
- Features used:
  - `mapkit.Map` - Map canvas initialization
  - `mapkit.Directions` - Route polyline generation between stops
  - `mapkit.Coordinate` - GPS coordinate representation
  - `mapkit.ImageAnnotation` - Shuttle vehicle markers
  - `mapkit.CircleOverlay` - Stop location indicators
  - `mapkit.Polyline` - Route path rendering
- Type definitions: `@types/apple-mapkit-js-browser` NPM package
- Browser requirement: Must be loaded in `<head>` before MapKit code runs (handled in HTML template)

## Data Storage

**Databases:**
- **PostgreSQL 17**
  - Connection: `DATABASE_URL` environment variable (postgresql+asyncpg:// scheme)
  - Host: `postgres` service in docker-compose (port 5432)
  - Client/ORM: SQLAlchemy 2.0.41+ async (asyncpg driver)
  - Persistence: Docker named volume `postgres_data`
  - Tables (5 core):
    - `vehicles` - Shuttle metadata (Samsara vehicle IDs)
    - `vehicle_locations` - GPS points with timestamp uniqueness
    - `geofence_events` - Service area enter/exit events
    - `drivers` - Driver metadata
    - `driver_vehicle_assignments` - Historical driver-vehicle pairings
  - Additional tables: Route, Stop, Polyline, BusSchedule, ETA, PredictedLocation (etc.)
  - Migrations: Managed via Alembic (`backend/alembic/`)

**File Storage:**
- Local filesystem only (no cloud storage integration)
- Shared static files: `shared/` directory (mounted in containers)
  - `shared/routes.json` - Route polylines, stops, colors (39.5 KB)
  - `shared/schedule.json` - Schedule data (26.5 KB)
  - `shared/aggregated_schedule.json` - Compiled schedule (16.7 KB)
  - `shared/announcements.json` - System announcements

**Caching:**
- **Redis 7**
  - Connection: `REDIS_URL` environment variable
  - Host: `redis` service in docker-compose (port 6379)
  - Client: aioredis (async Redis from `redis` package)
  - Persistence: AOF (Append-Only File) enabled, volume `redis_data`
  - Usage patterns:
    - Soft/hard TTL caching via custom `backend/cache.py` decorator
    - Cache namespace: `shubble-cache` prefix
    - Key examples: `shubble-cache:locations`, `shubble-cache:etas`
    - Soft TTL: Data returned but marked stale (e.g., 15s for locations)
    - Hard TTL: Data deleted from cache (e.g., 300s for locations)
  - Stores:
    - Vehicle locations (60s soft, 300s hard TTL)
    - ETAs per stop (15s soft, 300s hard TTL)
    - Geofence vehicle query results (900s TTL)
    - Per-stop ETA details computed by worker: `shubble:per_stop_etas_live`

## Authentication & Identity

**Auth Provider:**
- Custom authentication (no third-party OAuth/SAML)
- No user login system currently implemented
- Samsara API key authentication: `Authorization: Bearer {API_KEY}` header
- Webhook verification: HMAC-SHA256 using decoded `SAMSARA_SECRET`
  - Signature location: `Authorization` header with `Bearer` scheme
  - Verification code: `backend/fastapi/routes.py` (webhook endpoints)

## Monitoring & Observability

**Error Tracking:**
- Not configured (no Sentry, LogRocket, etc.)
- Application logs written to stdout (captured by container runtime)

**Logs:**
- Standard Python logging to stdout
  - Format: `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`
  - Components: FastAPI, Worker, ML pipeline (separate log level control)
  - Log levels: Configurable per component (FASTAPI_LOG_LEVEL, WORKER_LOG_LEVEL, ML_LOG_LEVEL)
  - Default: `LOG_LEVEL` environment variable (info)

**Health Checks:**
- FastAPI backend: HTTP GET `/api/locations` (checks database connectivity)
- PostgreSQL: Native pg_isready command
- Redis: redis-cli ping command
- Worker: Process running check (no HTTP endpoint)

## CI/CD & Deployment

**Hosting:**
- Docker containers (production-ready)
- Multi-container setup: backend, worker, frontend, postgres, redis
- Example deployment: Dokku (simple PaaS on VPS)

**CI Pipeline:**
- GitHub Actions workflows (`.github/workflows/`)
- Pipeline stages:
  - Build validation (Docker image builds)
  - Schedule data linting (YAML/JSON validation)
  - Deployment pipelines (staging, production, Dokku)

**Deployment Targets:**
- Development: Docker Compose (local or Docker Desktop)
- Staging: TBD (typically cloud provider)
- Production: TBD (typically cloud provider with persistent volumes)

## Environment Configuration

**Required env vars (production):**
- `DATABASE_URL` - PostgreSQL connection string (required)
- `REDIS_URL` - Redis connection string (required)
- `API_KEY` - Samsara API key (required for production, empty for dev/test)
- `SAMSARA_SECRET` - Base64-encoded webhook secret (required for prod webhooks)
- `MAPKIT_KEY` - Apple MapKit JS token (required for map rendering)

**Optional env vars:**
- `DEPLOY_MODE` - One of: development, staging, production (default: development)
- `DEBUG` - Enable debug mode (default: true)
- `LOG_LEVEL` - Global log level (default: info)
- `FRONTEND_URLS` - CORS whitelist, comma-separated (default: http://localhost:3000)
- `STATIC_ETAS` - Use static ETA predictions (default: false)
- `MAPKIT_KEY` - Can also be passed via Vite env var `VITE_MAPKIT_KEY`

**Secrets location:**
- Development: `.env` file (Git-ignored, use `.env.example` as template)
- Container secrets: Injected via environment variables at runtime
- No secret files committed to Git (proper .gitignore)

## Webhooks & Callbacks

**Incoming Webhooks:**
- `POST /api/webhook/geofence` - Samsara geofence events (enter/exit)
  - Payload: Geofence event data (vehicle_id, event_type, timestamp, location)
  - Signature verification: HMAC-SHA256 using `SAMSARA_SECRET`
  - Storage: Inserts to `geofence_events` table
  - Rate: Event-based (fired by Samsara when vehicle enters/exits)

- `POST /api/webhook/stats` - Samsara stats/location updates (fallback)
  - Payload: Vehicle location and telemetry
  - Purpose: Real-time location updates alternative to polling
  - Implementation: `backend/fastapi/routes.py` (webhook endpoints)

**Outgoing Webhooks:**
- None detected (no integration with external services via callbacks)
- Worker polling: Unidirectional pull from Samsara API, no callbacks

## Data Flow Summary

```
Samsara API (Production)
        ↓
httpx HTTP Client (backend/worker/worker.py)
        ↓
PostgreSQL (vehicle_locations table)
        ↓
Redis Cache (soft/hard TTL)
        ↓
FastAPI Endpoints (/api/locations, /api/etas, etc.)
        ↓
React Frontend (via Vite proxy)
        ↓
Apple MapKit JS (map rendering)
```

---

*Integration audit: 2026-04-05*
