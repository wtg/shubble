# Technology Stack

**Analysis Date:** 2026-04-05

## Languages

**Primary:**
- Python 3.13 - Backend, worker, test server, database migrations
- TypeScript 5.9.2 - Frontend type system for React
- JavaScript (Node.js) - Frontend build and runtime

**Secondary:**
- SQL (PostgreSQL dialect) - Database queries via SQLAlchemy ORM

## Runtime

**Environment:**
- Python 3.13 (via `docker/backend/Dockerfile.backend.dev`)
- Node.js 24-alpine (via `docker/frontend/Dockerfile.frontend.dev`)

**Package Manager:**
- **Backend:** uv (Astral, replaces pip/poetry) - Version 0.9+
  - Lockfile: `uv.lock` (frozen/locked)
  - Dependency file: `pyproject.toml`

- **Frontend:** npm (Node Package Manager)
  - Lockfile: `package-lock.json` (typically present)
  - Dependency file: `frontend/package.json`

## Frameworks

**Core:**
- **FastAPI** 0.115.0+ - Python backend web framework, async-first
  - Used in: `backend/fastapi/` for REST API endpoints
  - Includes automatic OpenAPI/Swagger docs
  - CORS middleware configured in `backend/fastapi/__init__.py`
  - Entry point: `shubble.py` exports `app` instance

- **React** 19.2.4 - Frontend UI library
  - Used in: `frontend/src/` for web interface
  - JSX/TSX syntax with TypeScript

- **React Router** 7.13.1 - Client-side routing
  - Routes defined in: `frontend/src/App.tsx`

**Database:**
- **SQLAlchemy** 2.0.41+ - Python async ORM
  - Async engine: `backend/database.py` uses `create_async_engine()`
  - Session factory: async context manager via `AsyncSession`
  - Models: `backend/models.py` (5 core tables)
  - Connection pooling with pre-ping health checks

- **Alembic** 1.14.0+ - Database migration tool
  - Async migrations: `backend/alembic/env.py`
  - Config: `backend/alembic.ini`
  - Migrations live in: `backend/alembic/versions/`

**Testing:**
- **pytest** - Python unit/integration testing framework
- **pytest-asyncio** - Async test support

**Build/Dev Tools:**
- **Vite** 7.3.1 - Frontend build tool (React)
  - Config: `frontend/vite.config.ts`
  - Dev server runs on port 3000 (configurable)
  - Proxy routing to backend: `/api` → `http://localhost:8000`

- **TypeScript** 5.9.2 - Static type checking for frontend
  - Config: `frontend/tsconfig.json` and `frontend/tsconfig.app.json`

- **ESLint** 9.39.4 - JavaScript/TypeScript linting
  - Config: `frontend/eslint.config.js` (flat config format)
  - Plugins: TypeScript, React, React Hooks, JSX A11y
  - Scope: `src/**/*.{ts,tsx}`

## Key Dependencies

**Critical (Backend):**
- **httpx** 0.28.1+ - Async HTTP client for Samsara API calls
  - Used in: `backend/worker/worker.py` for polling vehicle GPS data

- **asyncpg** 0.30.0+ - PostgreSQL async driver
  - Used with SQLAlchemy for async database connections
  - Driver prefix: `postgresql+asyncpg://`

- **redis** - Async Redis client for caching
  - Used via: `backend/cache.py` for soft/hard TTL caching
  - Client: `aioredis` (redis package async module)

- **pydantic** 2.10.0+ - Python data validation
  - Models: `backend/config.py` uses `BaseSettings` for env var validation
  - Request/response schemas throughout `backend/fastapi/routes.py`

- **pydantic-settings** 2.7.0+ - Environment configuration management
  - Settings class: `backend/config.py` reads `.env` files

- **python-dotenv** 1.1.1+ - Load `.env` files into environment
  - Loaded by Pydantic settings

- **uvicorn[standard]** 0.34.0+ - ASGI server
  - Runs: `shubble.py` app instance
  - Workers: 2 (configurable, dev uses `--reload`)
  - Host: `0.0.0.0`, Port: `8000`

- **brotli-asgi** - Response compression middleware
  - Configured in: `backend/fastapi/__init__.py` for responses >500 bytes

**Data Processing:**
- **numpy** - Numerical computing
- **pandas** 2.0.0-4.0.0 - Data manipulation (used for schedule caching)
- **scipy** - Scientific computing (schedule optimization via `linear_sum_assignment`)

**ML Dependencies (Optional Group: `ml`):**
- **torch** - PyTorch deep learning
- **scikit-learn** - Machine learning utilities
- **matplotlib** - Plotting/visualization
- **seaborn** - Statistical visualization
- **tqdm** 4.67.1+ - Progress bars
- **statsmodels** - Statistical modeling

**Frontend Dependencies:**
- **react-icons** 5.6.0 - Icon library
- **@types/apple-mapkit-js-browser** 5.78.1 - TypeScript types for MapKit
- **@vitejs/plugin-react** 5.1.4 - React Fast Refresh for Vite
- **vite-plugin-pwa** 1.2.0 - Progressive Web App support

**Development Only (Backend):**
- **ruff** - Fast Python linter/formatter
- **requests** - HTTP library for testing

## Configuration

**Environment Variables:**
Files read by code:
- `.env` (root, development) - Primary config via `pydantic-settings`
- `.env.example` - Template (available at: `.env.example`)
- Environment-specific: Set via `docker-compose.yml` environment sections

**Critical Configuration Variables** (from `backend/config.py` and `.env.example`):
- `DATABASE_URL` - PostgreSQL connection string (auto-converted to asyncpg driver)
- `REDIS_URL` - Redis cache connection
- `API_KEY` - Samsara API key for production GPS data
- `SAMSARA_SECRET` - Base64-encoded webhook signature verification secret
- `DEPLOY_MODE` - One of: `development`, `staging`, `production`
- `DEBUG` - Enable debug mode and SQL echo
- `LOG_LEVEL` - Global log level (fastapi, worker, ml can override)
- `FRONTEND_URLS` - CORS whitelist (comma-separated)
- `MAPKIT_KEY` - Apple MapKit JS authentication token

**Build Configuration:**
- `frontend/vite.config.ts` - Vite build settings, dev server proxy, PWA manifest
- `frontend/tsconfig.json` - TypeScript compiler options
- `pyproject.toml` - Python project metadata and dependency groups
- `docker-compose.yml` - Multi-service orchestration with profiles

## Platform Requirements

**Development:**
- Docker (containers for all services)
- Docker Compose (service orchestration)
- Python 3.13+ (or Docker image)
- Node.js 24+ (or Docker image)
- uv 0.9+ (Python package manager)
- npm 10+ (Node package manager)

**Production:**
- Docker runtime (builds via provided Dockerfiles)
- PostgreSQL 17 (persistent database)
- Redis 7 (persistent cache)
- External: Samsara API (GPS data) or mock test server
- External: Apple MapKit JS (map rendering)

**Operating Systems:**
- Development: Linux, macOS, Windows (via WSL2/Docker Desktop)
- Production: Linux (typical cloud/VPS deployment)

---

*Stack analysis: 2026-04-05*
