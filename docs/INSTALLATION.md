# Installation Guide

This guide explains how to set up and run the Shubble development environment.

## Architecture Overview

The codebase is organized into three main areas:

- **Frontend** - Main React application for end users
- **Backend** - FastAPI server (async), PostgreSQL database, Redis cache, and background worker
- **Test** - Mock Samsara API server and test client for development/testing

## Running Services: Docker vs Host

For each area, you have two options for running the services:

### Option 1: Dockerized (Recommended for First-Time Setup)

**Advantages:**
- Zero local setup required (no Node.js, Python, PostgreSQL, Redis installation)
- Consistent environment across all developers
- Easy to run multiple profiles without conflicts
- Isolated from your local system

**Disadvantages:**
- Slower hot reload and rebuild times
- More difficult to debug (can't easily attach debuggers)
- Changes require container rebuilds
- Less visibility into the running processes

**When to use:**
- First time setup
- Running services you're not actively developing
- Testing the full stack together
- CI/CD environments

### Option 2: Host (Recommended for Active Development)

**Advantages:**
- Instant hot reload during development
- Easy debugging with IDE integration
- Direct access to logs and processes
- Faster iteration cycle
- Can use local development tools

**Disadvantages:**
- Requires installing dependencies (Node.js, Python, PostgreSQL, Redis)
- Potential version conflicts with other projects
- Manual setup required
- Environment differences between developers

**When to use:**
- Actively developing/debugging a specific area
- Writing new features or fixing bugs
- Need to use debugging tools
- Frequent code changes

## Recommendation

**Run services on host for areas you're actively working on, and use Docker for the rest.**

For example:
- Working on the frontend? Run frontend on host, backend in Docker
- Working on the backend? Run backend on host, frontend in Docker
- Testing integration? Run everything in Docker

## Docker Setup

### Prerequisites

- Docker and Docker Compose installed
- Copy `.env.example` to `.env` and configure as needed

### Running Services

```bash
# Run only backend services (API, database, Redis, worker)
docker compose --profile backend up

# Run frontend (includes backend automatically)
docker compose --profile frontend up

# Run test services (mock Samsara API and test client)
docker compose --profile test up

# Run multiple profiles
docker compose --profile backend --profile test up

# Run everything
docker compose --profile "*" up
```

### Stopping Services

```bash
# Stop all running services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v
```

## Running on Host

### Prerequisites

**All environments:**
- Node.js 24+
- Python 3.13+
- PostgreSQL 17+
- Redis 7+

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your local database/Redis URLs
   ```

3. **Start PostgreSQL and Redis:**
   ```bash
   # Option 1: Run just database services in Docker
   docker compose up postgres redis

   # Option 2: Use local installations
   # (configure DATABASE_URL and REDIS_URL in .env accordingly)
   ```

4. **Run database migrations:**
   ```bash
   uv run alembic -c backend/alembic.ini upgrade head
   ```

5. **Start the backend server:**
   ```bash
   uv run uvicorn shubble:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Start the worker (in a separate terminal):**
   ```bash
   uv run python -m backend.worker
   ```

### Frontend Setup

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Set up environment variables:**
   ```bash
   # In .env, set (for Docker deployment):
   BACKEND_URL=http://localhost:8000
   ```

3. **Start the development server:**
   ```bash
   cd frontend
   npm run dev
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

### Test Services Setup

1. **Install test/client dependencies:**
   ```bash
   cd test/client
   npm install
   ```

2. **Set up environment variables:**
   ```bash
   # In .env, set (for Docker deployment):
   TEST_BACKEND_URL=http://localhost:4000
   ```

3. **Start the test server (in one terminal):**
   ```bash
   uv run uvicorn test.server.server:app --port 4000
   ```

4. **Start the test client (in another terminal):**
   ```bash
   cd test/client
   npm run dev
   ```

5. **Access the test services:**
   - Test Client: http://localhost:5174
   - Test Server API: http://localhost:4000

## Mixed Setup (Recommended)

The most common development setup is to run some services on host and others in Docker:

### Example: Frontend Development

```bash
# Terminal 1: Run backend in Docker
docker compose --profile backend up

# Terminal 2: Run frontend on host
cd frontend
npm run dev
```

### Example: Backend Development

```bash
# Terminal 1: Run database services in Docker
docker compose up postgres redis

# Terminal 2: Run backend on host
uv run uvicorn shubble:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3: Run worker on host
uv run python -m backend.worker

# Terminal 4 (optional): Run frontend in Docker
docker compose --profile frontend up
```

## Environment Variables

Key environment variables (see `.env.example` for full list):

### Service URLs
- `FRONTEND_URL` - Main frontend URL
- `BACKEND_URL` - Backend API URL for frontend (Docker deployment)
- `TEST_FRONTEND_URL` - Test client URL
- `TEST_BACKEND_URL` - Test server API URL for test client (Docker deployment)

### Database & Cache
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string

### Service Ports
- `FRONTEND_PORT` - Port for frontend (default: 3000)
- `BACKEND_PORT` - Port for backend API (default: 8000)
- `TEST_FRONTEND_PORT` - Port for test client (default: 5174)
- `TEST_BACKEND_PORT` - Port for test server (default: 4000)

## Troubleshooting

### Port Conflicts

If you see "port already in use" errors:

```bash
# Check what's using a port
lsof -i :8000

# Stop Docker services
docker compose down

# Change ports in .env if needed
```

### Database Issues

```bash
# Reset the database
docker compose down -v
docker compose up postgres

# Run migrations
uv run alembic -c backend/alembic.ini upgrade head
```

### Dependency Issues

```bash
# Clean install for Node.js (frontend)
cd frontend
rm -rf node_modules package-lock.json
npm install

# Clean install for Python
pip install --force-reinstall -r requirements.txt
```

## Next Steps

- See the main README for project overview
- Check individual service directories for specific documentation
- Review `.env.example` for all configuration options
