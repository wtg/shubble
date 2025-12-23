# Shubble Testing Guide

Comprehensive guide to testing the Shubble shuttle tracking application.

---

## Table of Contents

1. [Overview of Testing](#overview-of-testing)
2. [Frontend Testing (Vitest)](#frontend-testing-vitest)
3. [Backend Testing (Pytest)](#backend-testing-pytest)
4. [Mock Samsara API (Development Tool)](#mock-samsara-api-development-tool)
5. [Common Testing Workflows](#common-testing-workflows)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

---

## Overview of Testing

Shubble has multiple testing approaches:

### Automated Test Suites
- **Frontend Tests (Vitest)** - Component tests, integration tests, and utility tests for React frontend
- **Backend Tests (Pytest)** - Unit tests, integration tests, and end-to-end workflow tests for Flask backend

### Development Tools
- **Mock Samsara API (Test Server)** - Simulates real Samsara GPS API for local development without credentials
- **Test Client UI** - Web interface for controlling and managing simulated shuttles

### Test Organization

```
testing/
├── tests/                  # Automated pytest tests
│   ├── conftest.py        # Test fixtures and configuration
│   ├── test_api_endpoints.py
│   ├── test_models.py
│   ├── test_worker.py
│   └── integration/       # Integration tests
│       └── test_shuttle_workflow.py
│
├── test-server/           # Mock Samsara API (dev tool)
│   ├── server.py
│   └── shuttle.py
│
├── test-client/           # UI for controlling mock shuttles (dev tool)
│   ├── src/
│   ├── public/
│   └── vite.config.js
│
├── README.md              # This file
└── TROUBLESHOOTING.md     # Common issues and solutions

frontend/src/test/         # Automated vitest tests
├── setup.ts              # Test setup and global mocks
├── components/           # Component tests
├── integration/          # Integration tests
├── api/                  # API tests
└── utils/                # Utility tests
```

---

## Frontend Testing (Vitest)

Shubble uses **Vitest** for frontend testing - a modern, fast test runner built for Vite projects.

### Running Frontend Tests

**Note:** Frontend tests run natively only (not in Docker).

```bash
# Run all tests once
npm test

# Run tests in watch mode (auto-rerun on file changes)
npm test -- --watch

# Run tests with interactive UI
npm run test:ui

# Run tests with coverage report
npm run test:coverage
```

### Test Commands Reference

| Command | Description |
|---------|-------------|
| `npm test` | Run all tests once |
| `npm test -- --watch` | Watch mode (auto-rerun on changes) |
| `npm run test:ui` | Open interactive test UI in browser |
| `npm run test:coverage` | Generate coverage report |

### Writing Frontend Tests

Frontend tests are located in `frontend/src/test/` and follow the pattern `*.test.tsx` or `*.test.ts`.

**Example Component Test:**

```typescript
// frontend/src/test/components/MyComponent.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import MyComponent from '../../components/MyComponent';

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });

  it('handles user interaction', async () => {
    const { user } = render(<MyComponent />);
    await user.click(screen.getByRole('button'));
    expect(screen.getByText('Clicked')).toBeInTheDocument();
  });
});
```

### Test File Locations

- **Test configuration**: `vite.config.ts` (in root package.json)
- **Test setup file**: `frontend/src/test/setup.ts`
- **Component tests**: `frontend/src/test/components/`
- **Integration tests**: `frontend/src/test/integration/`
- **API tests**: `frontend/src/test/api/`
- **Utility tests**: `frontend/src/test/utils/`

### Best Practices

1. **Test user interactions, not implementation details**
2. **Use accessibility-friendly queries**: `screen.getByRole()`, `screen.getByLabelText()`
3. **Mock external dependencies**: API calls, MapKit, browser APIs
4. **Keep tests fast and independent**
5. **Use descriptive test names** that explain what's being tested

### Coverage Reports

After running `npm run test:coverage`, open `frontend/coverage/index.html` in your browser to view the coverage report.

---

## Backend Testing (Pytest)

Shubble uses **pytest** for backend testing - Python's standard testing framework.

### Running Backend Tests

**Note:** Backend tests run natively only (not in Docker).

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest testing/tests/test_models.py

# Run specific test
pytest testing/tests/test_models.py::test_vehicle_creation

# Run tests by marker
pytest -m unit              # Only unit tests
pytest -m integration       # Only integration tests
pytest -m "not slow"        # Exclude slow tests
```

### Test Commands Reference

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests |
| `pytest -v` | Verbose output |
| `pytest --cov=backend` | Run with coverage |
| `pytest --cov=backend --cov-report=html` | Generate HTML coverage report |
| `pytest -m unit` | Run only unit tests |
| `pytest -m integration` | Run only integration tests |
| `pytest testing/tests/test_models.py` | Run specific file |
| `pytest -k "test_name"` | Run tests matching name pattern |

### Writing Backend Tests

Backend tests are located in `testing/tests/` and follow the pattern `test_*.py`.

**Example Unit Test:**

```python
# testing/tests/test_models.py
import pytest
from backend.models import Vehicle

@pytest.mark.unit
def test_vehicle_creation():
    """Test Vehicle model instantiation"""
    vehicle = Vehicle(
        id="test_1",
        name="Test Shuttle",
        license_plate="ABC123"
    )
    assert vehicle.id == "test_1"
    assert vehicle.name == "Test Shuttle"
    assert vehicle.license_plate == "ABC123"
```

**Example Integration Test:**

```python
# testing/tests/integration/test_shuttle_workflow.py
import pytest
from backend.models import Vehicle, GeofenceEvent

@pytest.mark.integration
def test_geofence_entry_workflow(client, db_session):
    """Test full geofence entry workflow"""
    # Create vehicle
    vehicle = Vehicle(id="test_1", name="Shuttle 1")
    db_session.add(vehicle)
    db_session.commit()

    # Simulate geofence entry webhook
    response = client.post('/api/webhook', json={
        'eventType': 'geofenceEntry',
        'vehicle': {'id': 'test_1'},
        'eventTime': '2025-01-15T10:00:00Z'
    })

    assert response.status_code == 200
    assert GeofenceEvent.query.count() == 1
```

### Test Markers

Use markers to categorize tests (defined in `pytest.ini`):

- `@pytest.mark.unit` - Fast unit tests (no external dependencies)
- `@pytest.mark.integration` - Integration tests requiring DB/Redis
- `@pytest.mark.slow` - Slow running tests (skip during quick checks)

### Test Fixtures

Common fixtures are defined in `testing/tests/conftest.py`:

- **`app`** - Flask application instance with test configuration
- **`client`** - Test client for making HTTP requests
- **`db_session`** - Database session with automatic rollback after each test

**Using Fixtures:**

```python
def test_api_endpoint(client):
    """Test using the client fixture"""
    response = client.get('/api/routes')
    assert response.status_code == 200
    assert 'routes' in response.json

def test_database_operation(db_session):
    """Test using the db_session fixture"""
    vehicle = Vehicle(id="test_1", name="Test")
    db_session.add(vehicle)
    db_session.commit()

    assert Vehicle.query.count() == 1
```

### Test Configuration

Backend test configuration is in `pytest.ini`:

```ini
[pytest]
pythonpath = .
testpaths = testing/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
```

### Test Database & Cache

Tests use in-memory databases to avoid affecting production data:

- **Database**: SQLite in-memory (`:memory:`)
- **Cache**: SimpleCache (no Redis required)

Configuration is set in `testing/tests/conftest.py`:

```python
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
app.config['CACHE_TYPE'] = 'SimpleCache'
```

### Coverage Reports

After running `pytest --cov=backend --cov-report=html`, open `htmlcov/index.html` in your browser to view the coverage report.

---

## Mock Samsara API (Development Tool)

The Mock Samsara API allows local development and testing without real GPS credentials or live vehicles.

**Important:** This is a **development tool**, not an automated test suite.

### What is the Mock Samsara API?

The mock API consists of two services:

1. **Test Server** - Flask backend that simulates Samsara GPS API endpoints
2. **Test Client** - React UI for controlling simulated shuttles

### Why Use the Mock API?

- **No credentials needed** - Develop without Samsara API key
- **Controlled testing** - Trigger specific scenarios and edge cases
- **Offline development** - Work without internet connection
- **Predictable behavior** - Simulate exact shuttle movements and states

---

## Test Server Setup

The test server simulates the Samsara GPS API with mock vehicle data.

### Port Configuration

- **Native**: `http://localhost:4000`
- **Docker**: `http://localhost:4000` (from host), `http://test-server:4000` (from Docker network)

### Running Test Server - Native

```bash
# Navigate to test server directory
cd testing/test-server

# Install dependencies (if needed)
pip install Flask Flask-Cors Flask-SQLAlchemy

# Start the test server
python server.py

# Server runs on http://localhost:4000
```

**What it provides:**
- Mock Samsara API endpoints
- Simulated vehicle movement along routes
- Fake geofence events
- Mock GPS data
- API endpoints to control shuttle states

### Running Test Server - Docker

```bash
# Start with Docker Compose (includes test-server and test-client)
docker-compose --profile dev up -d

# View logs
docker-compose logs -f test-server

# Server runs on http://localhost:4000
```

### Test Server API Endpoints

The test server provides these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/routes` | GET | Get route data (from `data/routes.json`) |
| `/api/schedule` | GET | Get schedule data (from `data/aggregated_schedule.json`) |
| `/api/shuttles` | GET | List all simulated shuttles |
| `/api/shuttles` | POST | Create new shuttle |
| `/api/shuttles/<id>` | GET | Get shuttle details |
| `/api/shuttles/<id>` | DELETE | Remove shuttle |
| `/api/shuttles/<id>/state` | POST | Update shuttle state (entering, looping, exiting) |
| `/api/feed/locations` | GET | Samsara-compatible locations feed |
| `/locations/history` | GET | Samsara-compatible location history |

---

## Test Client Setup

The test client is a React web UI for controlling and monitoring simulated shuttles.

### Port Configuration

- **Native**: `http://localhost:5174` (Vite dev server)
- **Docker**: `http://localhost:5174` (nginx)

### Running Test Client - Native

```bash
# Navigate to test client directory
cd testing/test-client

# Install dependencies
npm install

# Start the test client
npm run dev

# Client runs on http://localhost:5174
```

**Requirements:**
- Test server must be running on `http://localhost:4000`
- Vite proxy automatically forwards `/api/*` requests to test server

### Running Test Client - Docker

```bash
# Start with Docker Compose (includes test-server and test-client)
docker-compose --profile dev up -d

# View logs
docker-compose logs -f test-client

# Client runs on http://localhost:5174
```

**Requirements:**
- Test server container must be running
- Nginx proxy automatically forwards `/api/*` requests to test server

### Test Client Features

The test client UI provides:

- **Shuttle Management** - Create, view, and delete simulated shuttles
- **Visual Interface** - See shuttle positions and states in real-time
- **State Control** - Trigger state changes (entering, looping, exiting)
- **Automated Scenarios** - Run test scenarios from JSON files
- **Event Monitoring** - Watch geofence events and location updates
- **Debugging Tools** - Inspect API requests and responses

### Using the Test Client

1. **Access the UI**: Open `http://localhost:5174` in your browser
2. **Create shuttles**: Click "Add Shuttle" and select a route
3. **Control movement**: Use buttons to start/stop shuttle movement
4. **Monitor events**: Watch geofence entry/exit events
5. **Debug issues**: Check network tab for API requests

---

## Integration with Main App

To use the mock API with the main Shubble application:

### Step 1: Start Mock Services

**Native:**
```bash
# Terminal 1: Start test server
cd testing/test-server
python server.py

# Terminal 2: Start test client (optional)
cd testing/test-client
npm run dev
```

**Docker:**
```bash
docker-compose --profile dev up -d
```

### Step 2: Configure Main App

In your `.env` file:

```bash
FLASK_ENV=development

# Leave API_KEY empty to use mock server
API_KEY=

# Backend will automatically use test server at http://localhost:4000
```

### Step 3: Start Main App

```bash
# Terminal 3: Start backend (will use mock server)
flask --app backend:create_app run --port 5001

# Terminal 4: Start frontend
npm run dev

# Terminal 5: Start worker
python -m backend.worker
```

### How It Works

When `API_KEY` is empty in development mode:
- Backend automatically redirects API calls to `http://localhost:4000` (test server)
- Worker fetches GPS data from mock shuttles instead of real vehicles
- Geofence webhooks come from test server instead of Samsara
- All other functionality works normally

---

## Common Testing Workflows

### 1. Run Unit Tests Before Committing

Quick sanity check before committing code:

```bash
# Quick check (both frontend and backend)
npm test && pytest

# More thorough check with coverage
npm run test:coverage && pytest --cov=backend
```

### 2. Test a New Feature End-to-End

**Using Mock API for manual testing:**

```bash
# 1. Start test server and client
docker-compose --profile dev up -d

# 2. Configure main app to use mock server
# Edit .env: API_KEY=

# 3. Start main app
flask --app backend:create_app run --port 5001 &
npm run dev &
python -m backend.worker &

# 4. Create test shuttles in test client
# Open http://localhost:5174

# 5. Watch shuttles in main app
# Open http://localhost:5173

# 6. Verify behavior matches requirements
```

### 3. Debug a Failing Test

**Backend test debugging:**

```bash
# Run single test with verbose output and debug prints
pytest -v -s testing/tests/test_models.py::test_vehicle_creation

# Use debugger
pytest --pdb testing/tests/test_models.py::test_vehicle_creation

# Check test logs
pytest --log-cli-level=DEBUG testing/tests/test_models.py
```

**Frontend test debugging:**

```bash
# Run single test file in watch mode
npm test -- components/MyComponent.test.tsx --watch

# Use UI mode for visual debugging
npm run test:ui

# Enable console logs in tests
# Add console.log() statements in test files
```

### 4. Test the Full Shuttle Tracking Workflow

**End-to-end integration test:**

```bash
# Run integration tests
pytest -m integration -v

# Or test manually with mock API:
# 1. Start mock services
docker-compose --profile dev up -d

# 2. Create shuttle in test client
# - Navigate to http://localhost:5174
# - Click "Add Shuttle" → Select "North Route" → Click "Entering"

# 3. Verify in main app
# - Navigate to http://localhost:5173
# - Should see shuttle appear on map
# - Should see route assignment in schedule

# 4. Test state transitions
# - In test client, click "Looping" → shuttle should loop route
# - In test client, click "Exiting" → shuttle should disappear

# 5. Check database
docker-compose exec postgres psql -U shubble -d shubble
# SELECT * FROM vehicle_locations ORDER BY timestamp DESC LIMIT 10;
```

### 5. Validate Schedule Matching Algorithm

```bash
# Test schedule matching with mock data
python -c "
from backend import create_app
from data.schedules import match_shuttles_to_schedules
app = create_app()
with app.app_context():
    results = match_shuttles_to_schedules()
    print(results)
"
```

### 6. Load Testing with Multiple Shuttles

**Using test client to simulate load:**

1. Open test client: `http://localhost:5174`
2. Create 10+ shuttles on different routes
3. Set all to "Looping" state
4. Monitor backend logs for performance
5. Check Redis cache hit rates
6. Verify database query performance

---

## Configuration

### Frontend Test Configuration

No special configuration needed - tests use settings from `vite.config.ts`.

**Environment variables** (optional):
```bash
# None required for frontend tests
```

### Backend Test Configuration

Tests automatically use test configuration from `testing/tests/conftest.py`:

```python
# Database: SQLite in-memory
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'

# Flask environment
os.environ['FLASK_ENV'] = 'testing'

# Cache: SimpleCache (no Redis)
app.config['CACHE_TYPE'] = 'SimpleCache'
```

**No `.env` file needed** - tests are self-contained.

### Test Server Configuration

**Environment variables** (optional):

```bash
# Frontend URLs for CORS
FRONTEND_URL=http://localhost:5173
TEST_FRONTEND_URL=http://localhost:5174

# Database (optional - test server can run without DB)
DATABASE_URL=postgresql://shubble:shubble@localhost:5432/shubble

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
```

### Test Client Configuration

Create `testing/test-client/.env`:

```bash
# API Base URL (Optional - leave empty to use proxy)
VITE_TEST_BACKEND_URL=

# If empty, Vite proxies /api/* to http://localhost:4000
# If set, makes direct requests to specified URL
```

**Proxy configuration** (in `testing/test-client/vite.config.js`):

```javascript
server: {
  port: 5174,
  proxy: {
    '/api': {
      target: process.env.VITE_API_URL || 'http://localhost:4000',
      changeOrigin: true,
    },
  },
}
```

---

## Troubleshooting

### Frontend Tests

**Issue: Module not found errors**

```bash
# Solution: Install dependencies
npm install
```

**Issue: Tests timeout**

```bash
# Solution 1: Increase timeout in test file
import { describe, it, expect } from 'vitest';
it('slow test', async () => {
  // ...
}, { timeout: 10000 }); // 10 seconds

# Solution 2: Check for infinite loops in code
# Solution 3: Verify mocks are properly set up
```

**Issue: MapKit errors in tests**

```bash
# Solution: Check frontend/src/test/setup.ts has MapKit mocks
global.mapkit = {
  init: vi.fn(),
  // ... other MapKit mocks
};
```

### Backend Tests

**Issue: Database connection errors**

```bash
# Solution: Tests should use SQLite in-memory
# Check testing/tests/conftest.py:
# os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
```

**Issue: Redis connection errors**

```bash
# Solution: Tests should use SimpleCache, not Redis
# Check testing/tests/conftest.py:
# app.config['CACHE_TYPE'] = 'SimpleCache'
```

**Issue: Import errors (`ModuleNotFoundError: No module named 'backend'`)**

```bash
# Solution 1: Ensure you're in virtual environment
source venv/bin/activate
pip install -r requirements.txt

# Solution 2: Check pytest.ini pythonpath setting
# pythonpath = .

# Solution 3: Run from project root
cd /Users/joel/eclipse-workspace/shuttletracker-new
pytest
```

**Issue: Foreign key constraint errors**

```bash
# Solution: Create parent records before child records
vehicle = Vehicle(id="test_1", name="Test")
db_session.add(vehicle)
db_session.commit()

# THEN create child record
event = GeofenceEvent(vehicle_id="test_1", ...)
db_session.add(event)
db_session.commit()
```

### Mock API Issues

**Issue: Test client can't reach test server**

See `testing/TROUBLESHOOTING.md` for detailed solutions.

**Quick diagnostic:**

```bash
# 1. Check if test server is running
curl http://localhost:4000/api/routes

# 2. Check if test client is running
curl http://localhost:5174

# 3. Check proxy is working
curl http://localhost:5174/api/routes

# 4. Check Docker services (if using Docker)
docker-compose --profile dev ps
docker-compose logs test-server
docker-compose logs test-client
```

**Common mistakes:**

- ❌ Accessing `http://localhost:4000` (that's the API, not the UI)
- ✅ Use `http://localhost:5174` for both native and Docker

**Issue: CORS errors**

```bash
# Solution: Ensure Flask-Cors is installed
pip install Flask-Cors

# Restart test server after installing
cd testing/test-server
python server.py
```

**Issue: Shuttles not moving in test client**

```bash
# 1. Check test server logs for errors
docker-compose logs -f test-server  # Docker
# OR check terminal output (native)

# 2. Verify shuttle state is "looping"
curl http://localhost:4000/api/shuttles

# 3. Check for JavaScript errors in browser console (F12)
```

---

## Quick Reference

### Test Commands Summary

```bash
# Frontend Tests (Native only)
npm test                      # Run all tests
npm test -- --watch          # Watch mode
npm run test:ui              # Interactive UI
npm run test:coverage        # With coverage

# Backend Tests (Native only)
pytest                       # Run all tests
pytest -v                    # Verbose
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest --cov=backend        # With coverage

# Mock API - Native
cd testing/test-server && python server.py          # Test server (port 4000)
cd testing/test-client && npm run dev               # Test client (port 5174)

# Mock API - Docker
docker-compose --profile dev up -d                  # Both services
docker-compose logs -f test-server                  # Server logs
docker-compose logs -f test-client                  # Client logs
```

### Port Reference

| Service | Native | Docker |
|---------|--------|--------|
| Main Backend | 5001 | 5001 |
| Main Frontend | 5173 | 5173 |
| Test Server | 4000 | 4000 |
| Test Client | 5174 | 5174 |
| PostgreSQL | 5432 | 5432 |
| Redis | 6379 | 6379 |

### File Locations

| Type | Location |
|------|----------|
| Frontend tests | `frontend/src/test/` |
| Backend tests | `testing/tests/` |
| Test server | `testing/test-server/` |
| Test client | `testing/test-client/` |
| Test config (pytest) | `pytest.ini` |
| Test config (vitest) | `vite.config.ts` (in package.json) |
| Test fixtures | `testing/tests/conftest.py` |

---

## Related Documentation

- [../docs/TESTING.md](../docs/TESTING.md) - Original testing documentation (deprecated, use this file instead)
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Detailed troubleshooting for mock API
- [../docs/INSTALLATION.md](../docs/INSTALLATION.md) - Development setup
- [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) - Code architecture
- [../README.md](../README.md) - Project overview
- [../CLAUDE.md](../CLAUDE.md) - Architecture guide for AI assistance
