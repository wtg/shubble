# Installation Guide

This guide will help you set up Shubble for local development. You can run components **fully native**, **fully in Docker**, or **mix and match** however you prefer.

## Prerequisites

Choose what you need based on your setup approach:

### For Native Setup
- **Node.js 20+** and npm
- **Python 3.12+**
- **PostgreSQL 14+** (if running database natively)
- **Redis 7+** (if running cache natively)
- **Git**

### For Docker Setup
- **Docker Desktop** (or Docker Engine + Docker Compose)
- **Git**

### Quick Install (macOS)
```bash
# For native development
brew install node python@3.12 postgresql@16 redis git

# For Docker only
brew install --cask docker
```

### Quick Install (Ubuntu/Debian)
```bash
# For native development
sudo apt update
sudo apt install nodejs npm python3.12 postgresql redis-server git

# For Docker only
sudo apt install docker.io docker-compose
```

---

## Option 1: Fully Native Setup

**Best for:** Fast iteration, easy debugging, lower resource usage

This approach runs all services directly on your machine without Docker.

### Step 1: Clone Repository

```bash
git clone git@github.com:wtg/shubble.git
cd shubble
```

### Step 2: Set Up PostgreSQL

**macOS (Homebrew):**
```bash
brew install postgresql@16
brew services start postgresql@16

# Create database and user
psql postgres
```

**Ubuntu/Debian:**
```bash
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql

# Create database and user
sudo -u postgres psql
```

**In psql (both platforms):**
```sql
CREATE DATABASE shubble;
CREATE USER shubble WITH PASSWORD 'shubble';
GRANT ALL PRIVILEGES ON DATABASE shubble TO shubble;
\q
```

### Step 3: Set Up Redis

**macOS (Homebrew):**
```bash
brew install redis
brew services start redis
```

**Ubuntu/Debian:**
```bash
sudo apt install redis-server
sudo systemctl start redis-server
```

**Verify Redis:**
```bash
redis-cli ping  # Should return: PONG
```

### Step 4: Set Up Python Backend

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # or use your favorite editor
```

**Edit `.env` for native setup:**
```bash
# Service URLs
FRONTEND_URL=http://localhost:5173
BACKEND_URL=http://localhost:5001

# Database & Cache (use localhost for native)
DATABASE_URL=postgresql://shubble:shubble@localhost:5432/shubble
REDIS_URL=redis://localhost:6379/0

# Flask Config
FLASK_ENV=development
FLASK_DEBUG=true
LOG_LEVEL=INFO

# Samsara API (leave empty to use mock server)
API_KEY=
SAMSARA_SECRET=
```

### Step 5: Set Up Frontend

```bash
# Install dependencies
npm install
```

### Step 6: Start All Services

You'll need **3 terminal windows**:

**Terminal 1 - Backend:**
```bash
source venv/bin/activate
flask --app backend:create_app run
```
Backend will be available at http://localhost:5001

**Terminal 2 - Frontend:**
```bash
npm run dev
```
Frontend will be available at http://localhost:5173

**Terminal 3 - Worker:**
```bash
source venv/bin/activate
python -m backend.worker
```
Worker runs in background (no HTTP interface)

### Step 7: Verify Installation

Open http://localhost:5173 in your browser. You should see the Shubble map interface.

**To test with mock data**, see [Option 3: Mixed Setups](#option-3-mixed-setups-recommended) below for running the test server.

---

## Option 2: Fully Docker Setup

**Best for:** Isolated environment, no local dependencies, production-like testing

This approach runs everything in Docker containers.

### Step 1: Clone Repository

```bash
git clone git@github.com:wtg/shubble.git
cd shubble
```

### Step 2: Configure Environment

```bash
# Copy example environment file
cp .env.example .env
nano .env  # or use your favorite editor
```

**Edit `.env` for Docker setup:**
```bash
# Service URLs
FRONTEND_URL=http://localhost:5173
BACKEND_URL=http://localhost:5001

# Database & Cache (use service names for Docker)
DATABASE_URL=postgresql://shubble:shubble@postgres:5432/shubble
REDIS_URL=redis://redis:6379/0

# Flask Config
FLASK_ENV=development
FLASK_DEBUG=true
LOG_LEVEL=INFO

# Samsara API (leave empty to use mock server)
API_KEY=
SAMSARA_SECRET=
```

### Step 3: Start All Services

```bash
# Start main application (frontend, backend, worker, postgres, redis)
docker-compose up -d

# View logs
docker-compose logs -f
```

**OR with Mock Samsara API for development:**
```bash
# Start everything including test-server and test-client
docker-compose --profile dev up -d

# View logs
docker-compose logs -f
```

### Step 4: Verify Installation

**Main Application:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:5001

**Mock Services (if using --profile dev):**
- Test Server: http://localhost:4000
- Test Client: http://localhost:5174

Test the backend:
```bash
curl http://localhost:5001/api/locations
```

---

## Option 3: Mixed Setups (Recommended)

**The real power:** Mix Docker and native however you like!

This flexibility lets you run infrastructure in Docker while developing components natively for fast iteration.

### Common Mixed Setup Examples

#### Example A: Infrastructure in Docker + Application Native

**Best for:** Most developers - get databases without hassle, develop with fast reloads

**Start infrastructure:**
```bash
# Start only PostgreSQL and Redis in Docker
docker-compose up -d postgres redis
```

**Configure `.env`:**
```bash
# Use localhost since you're connecting FROM native TO Docker
DATABASE_URL=postgresql://shubble:shubble@localhost:5432/shubble
REDIS_URL=redis://localhost:6379/0
BACKEND_URL=http://localhost:5001
API_KEY=
```

**Start application natively (3 terminals):**
```bash
# Terminal 1: Backend
source venv/bin/activate
flask --app backend:create_app run

# Terminal 2: Frontend
npm run dev

# Terminal 3: Worker
source venv/bin/activate
python -m backend.worker
```

**Access:**
- Frontend: http://localhost:5173
- Backend: http://localhost:5001

---

#### Example B: Backend in Docker + Frontend Native

**Best for:** Frontend developers - no Python setup needed, fast frontend iteration

**Start backend services:**
```bash
# Start postgres, redis, backend, worker in Docker
docker-compose up -d postgres redis backend worker
```

**Start frontend natively:**
```bash
npm install
npm run dev
```

**Access:**
- Frontend: http://localhost:5173 (native Vite dev server)
- Backend: http://localhost:5001 (Docker container)

**Benefits:**
- No Python environment needed
- Instant frontend HMR
- Backend in isolated container

---

#### Example C: Everything Docker + Test Server Native

**Best for:** Testing mock API changes while app runs isolated

**Start main application:**
```bash
docker-compose up -d
```

**Start test server natively:**
```bash
cd testing/test-server
python server.py  # Runs on http://localhost:4000
```

**Access:**
- Main app: http://localhost:5173
- Test server: http://localhost:4000

---

#### Example D: Native Development + Test Suite in Docker

**Best for:** Developers who want quick app development with isolated test environment

**Start test services only:**
```bash
docker-compose --profile dev up -d test-server test-client
```

**Start main app natively (3 terminals):**
```bash
# Terminal 1: Infrastructure
docker-compose up -d postgres redis

# Terminal 2: Backend
source venv/bin/activate
flask --app backend:create_app run

# Terminal 3: Frontend
npm run dev

# Terminal 4: Worker
source venv/bin/activate
python -m backend.worker
```

**Access:**
- Main app frontend: http://localhost:5173 (native)
- Main app backend: http://localhost:5001 (native)
- Test server: http://localhost:4000 (Docker)
- Test client: http://localhost:5174 (Docker)

---

### Mix and Match Reference Table

| Component | Native Command | Docker Command | Port |
|-----------|---------------|----------------|------|
| **PostgreSQL** | `brew services start postgresql@16` | `docker-compose up -d postgres` | 5432 |
| **Redis** | `brew services start redis` | `docker-compose up -d redis` | 6379 |
| **Backend** | `flask --app backend:create_app run` | `docker-compose up -d backend` | 5001 |
| **Worker** | `python -m backend.worker` | `docker-compose up -d worker` | (none) |
| **Frontend** | `npm run dev` | `docker-compose up -d frontend` | 5173 |
| **Test Server** | `cd testing/test-server && python server.py` | `docker-compose --profile dev up -d test-server` | 4000 |
| **Test Client** | `cd testing/test-client && npm run dev` | `docker-compose --profile dev up -d test-client` | 5174 |

### Environment Variable Rules for Mixed Setups

**Key principle:** When connecting FROM native TO Docker services (or vice versa), use `localhost`. When connecting FROM Docker TO Docker, use service names.

**Connecting from Native to Docker services:**
```bash
# Native app → Docker database
DATABASE_URL=postgresql://shubble:shubble@localhost:5432/shubble
REDIS_URL=redis://localhost:6379/0
```

**Connecting from Docker to Docker services:**
```bash
# Docker app → Docker database
DATABASE_URL=postgresql://shubble:shubble@postgres:5432/shubble
REDIS_URL=redis://redis:6379/0
```

**Backend URL for frontend:**
```bash
# Both native OR both Docker
BACKEND_URL=http://localhost:5001

# Mixed: doesn't matter, frontend uses proxy or localhost
BACKEND_URL=http://localhost:5001
```

---

## Verification Steps

After setup, verify everything works:

### 1. Check Backend Health

```bash
curl http://localhost:5001/api/locations
```

Expected: JSON response (may be empty array if no shuttles active)

### 2. Check Frontend

Open http://localhost:5173 in browser. You should see:
- Map interface loads
- No console errors (check browser DevTools)

### 3. Check Database Connection

**Native:**
```bash
psql postgresql://shubble:shubble@localhost:5432/shubble -c "SELECT 1"
```

**Docker:**
```bash
docker-compose exec postgres psql -U shubble -d shubble -c "SELECT 1"
```

### 4. Check Redis Connection

**Native:**
```bash
redis-cli -p 6379 ping
```

**Docker:**
```bash
docker-compose exec redis redis-cli ping
```

### 5. Check Worker Logs

**Native:**
Check Terminal 3 output - should see polling logs every 5 seconds

**Docker:**
```bash
docker-compose logs -f worker
```

### 6. Test with Mock Data (Optional)

Start test server and create a mock shuttle:

**Native test server:**
```bash
cd testing/test-server
python server.py
```

**Create test shuttle:**
```bash
curl -X POST http://localhost:4000/api/shuttles \
  -H "Content-Type: application/json" \
  -d '{"route": "west"}'
```

Check if it appears in main app: http://localhost:5173

---

## Troubleshooting

### Port Already in Use

**Check what's using a port:**
```bash
lsof -i :5001  # Backend
lsof -i :5173  # Frontend
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis
lsof -i :4000  # Test server
lsof -i :5174  # Test client
```

**Fix:** Kill the process or change the port in configuration

### Database Connection Errors

**Check if PostgreSQL is running:**

Native:
```bash
# macOS
brew services list | grep postgresql

# Linux
sudo systemctl status postgresql
```

Docker:
```bash
docker-compose ps postgres
```

**Test connection:**
```bash
psql -U shubble -d shubble -h localhost
# Password: shubble
```

**Common fixes:**
- Verify DATABASE_URL in `.env` matches your setup (localhost vs postgres)
- Check PostgreSQL is actually running
- Verify user/password are correct

### Redis Connection Errors

**Check if Redis is running:**

Native:
```bash
# macOS
brew services list | grep redis

# Linux
sudo systemctl status redis-server
```

Docker:
```bash
docker-compose ps redis
```

**Test connection:**
```bash
redis-cli -p 6379 ping
# Should return: PONG
```

**Common fixes:**
- Verify REDIS_URL in `.env` (localhost vs redis)
- Check Redis is running
- Verify port 6379 is not blocked

### Frontend Can't Reach Backend

**Symptoms:**
- Network errors in browser console
- "Failed to fetch" errors
- No data loading on map

**Checks:**
```bash
# Test backend directly
curl http://localhost:5001/api/locations

# Check backend logs for errors
# Native: Check Terminal 1 output
# Docker: docker-compose logs backend
```

**Common fixes:**
- Verify backend is running on port 5001
- Check BACKEND_URL in `.env` or Vite proxy config
- Check browser console for CORS errors
- Verify firewall isn't blocking localhost connections

### Worker Not Polling

**Symptoms:**
- No location updates
- Worker logs show errors or no activity

**Checks:**

Native:
```bash
# Check worker terminal output for errors
# Ensure virtual environment is activated
source venv/bin/activate
echo $VIRTUAL_ENV  # Should show path to venv
```

Docker:
```bash
# Check worker logs
docker-compose logs worker

# Restart worker
docker-compose restart worker
```

**Common fixes:**
- Verify API_KEY in `.env` (empty = uses mock server)
- Check DATABASE_URL and REDIS_URL are correct
- Ensure no vehicles in geofence if using real API
- Check worker has network access to Samsara API (if using real credentials)

### Module Not Found (Python)

**Error:** `ModuleNotFoundError: No module named 'flask'` (or similar)

**Fix:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep -i flask
```

### Node Module Errors

**Error:** Module not found or package issues

**Fix:**
```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install

# If still issues, clear npm cache
npm cache clean --force
npm install
```

### Docker Build Fails

**Common issues:**

**Out of disk space:**
```bash
# Check disk usage
docker system df

# Clean up
docker system prune -a --volumes
```

**Network issues:**
```bash
# Restart Docker daemon
# macOS: Docker Desktop → Restart
# Linux: sudo systemctl restart docker
```

**Permission errors (Linux):**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

### Worker Shows No Logs

**Likely cause:** No vehicles in geofence, so worker has nothing to poll

**Check geofence events:**

Native:
```bash
psql postgresql://shubble:shubble@localhost:5432/shubble
SELECT * FROM geofence_events ORDER BY event_time DESC LIMIT 10;
\q
```

Docker:
```bash
docker-compose exec postgres psql -U shubble -d shubble
SELECT * FROM geofence_events ORDER BY event_time DESC LIMIT 10;
\q
```

**Solution:** Use test server to generate mock events (see Mock Data section above)

### Changes Not Reflecting

**Frontend changes not showing:**

Native:
- Vite HMR should reload automatically
- Try hard refresh: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows/Linux)
- Check terminal for HMR errors

Docker:
- Frontend container serves built assets - need to rebuild:
  ```bash
  docker-compose up -d --build frontend
  ```

**Backend changes not showing:**

Native:
- Flask auto-reloads in debug mode
- If not reloading, restart backend manually

Docker:
- Need to restart container:
  ```bash
  docker-compose restart backend
  ```

**Worker changes not showing:**

Native:
- Stop worker (Ctrl+C) and restart:
  ```bash
  python -m backend.worker
  ```

Docker:
- Restart container:
  ```bash
  docker-compose restart worker
  ```

---

## Next Steps

Now that you have Shubble running locally:

1. **Read the architecture:** [docs/ARCHITECTURE.md](ARCHITECTURE.md) - Understand how the system works
2. **Set up testing:** [testing/README.md](../testing/README.md) - Learn about the test suite and mock server
3. **Explore the codebase:** [CLAUDE.md](../CLAUDE.md) - Quick reference for common development tasks
4. **Make changes:** Edit code and see your changes live!

### Development Workflow

**For fast iteration (recommended):**
- Use fully native setup OR infrastructure in Docker + application native
- Frontend changes: Just save files (Vite HMR)
- Backend changes: Just save files (Flask auto-reload)
- Worker changes: Restart worker process

**For isolated testing:**
- Use fully Docker setup
- Rebuild containers after code changes
- Closer to production environment

### Using the Mock Samsara API

If you don't have Samsara API credentials, use the test server for development:

**Start test server:**
```bash
# Native
cd testing/test-server
python server.py

# Docker
docker-compose --profile dev up -d test-server test-client
```

**Use test client UI:**
- Native: http://localhost:5173 (if running test-client native)
- Docker: http://localhost:5174

**Or use API directly:**
```bash
# Create a shuttle on west route
curl -X POST http://localhost:4000/api/shuttles \
  -H "Content-Type: application/json" \
  -d '{"route": "west"}'

# Trigger state change
curl -X POST http://localhost:4000/api/shuttles/1/set-next-state
```

See [testing/README.md](../testing/README.md) for complete test server documentation.

### Running Tests

Before committing changes, run tests:

**Backend tests (pytest):**
```bash
# Native
source venv/bin/activate
pytest

# Docker
docker-compose exec backend pytest
```

**Frontend tests (vitest):**
```bash
# Native
npm test

# Docker
docker-compose exec frontend npm test
```

See [docs/TESTING.md](TESTING.md) for comprehensive testing documentation.

---

## Getting Help

If you encounter issues not covered here:

1. Check [PORTS.md](../PORTS.md) for port reference
2. Check [testing/TROUBLESHOOTING.md](../testing/TROUBLESHOOTING.md) for test-specific issues
3. Search existing [GitHub issues](https://github.com/wtg/shubble/issues)
4. Open a new issue with:
   - Your setup type (native/Docker/mixed)
   - Operating system
   - Error messages
   - Steps to reproduce
5. Contact [Joel McCandless](mailto:mail@joelmccandless.com)

---

## Quick Reference

### Full Native Setup (3 terminals)
```bash
# Terminal 1: Backend
source venv/bin/activate
flask --app backend:create_app run

# Terminal 2: Frontend
npm run dev

# Terminal 3: Worker
source venv/bin/activate
python -m backend.worker
```

### Full Docker Setup (1 command)
```bash
docker-compose up -d
```

### Mixed: Infrastructure Docker + App Native (3 terminals)
```bash
# Terminal 0: Infrastructure
docker-compose up -d postgres redis

# Terminal 1: Backend
source venv/bin/activate
flask --app backend:create_app run

# Terminal 2: Frontend
npm run dev

# Terminal 3: Worker
source venv/bin/activate
python -m backend.worker
```

### Port Reference
- **Frontend:** http://localhost:5173
- **Backend:** http://localhost:5001
- **PostgreSQL:** localhost:5432
- **Redis:** localhost:6379
- **Test Server:** http://localhost:4000
- **Test Client:** http://localhost:5174
