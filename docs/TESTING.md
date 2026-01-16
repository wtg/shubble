# Testing Guide

This guide explains how to test Shubble without access to the real Samsara API.

## Overview

Shubble includes a **test server** that simulates the Samsara Fleet API and a **test client** UI for controlling simulated shuttles. This allows you to:

- Develop and test without Samsara API credentials
- Simulate shuttle movements along real routes
- Test geofence entry/exit events
- Populate the database with realistic test data

## Quick Start with Docker

The easiest way to run the test environment is with Docker Compose:

```bash
# Start all test services
docker compose --profile test --profile backend up
```

This starts:
- **PostgreSQL** (port 5432) - Database
- **Redis** (port 6379) - Cache
- **Backend** (port 8000) - Main API server
- **Worker** - Background GPS poller
- **Test Server** (port 4000) - Mock Samsara API
- **Test Client** (port 5174) - Control UI

Access the test client at: http://localhost:5174

## Running on Host

### Prerequisites

- Python 3.13+
- Node.js 24+
- PostgreSQL 17+ (or via Docker)
- Redis 7+ (or via Docker)

### Step 1: Start Database Services

```bash
# Option A: Use Docker for database services only
docker compose up postgres redis

# Option B: Use local installations
# (configure DATABASE_URL and REDIS_URL in .env)
```

### Step 2: Start the Backend

```bash
# Terminal 1: API server
uvicorn shubble:app --reload --port 8000

# Terminal 2: Worker (polls test server instead of Samsara)
python -m backend.worker
```

### Step 3: Start the Test Server

```bash
# Terminal 3: Mock Samsara API
cd test-server
uvicorn server:app --port 4000
```

### Step 4: Start the Test Client (Optional)

```bash
# Terminal 4: Test control UI
cd test-client
npm install
npm run dev
```

Access at: http://localhost:5173

## Using the Test Client

The test client provides a web UI for controlling simulated shuttles.

### Adding Shuttles

1. Open the test client in your browser
2. Click "Add Shuttle" to create a new simulated shuttle
3. The shuttle will be assigned a unique ID

### Controlling Shuttles

For each shuttle, you can:

- **Set State** - Change between running, stopped, at_stop, out_of_service
- **Assign Route** - Select which route the shuttle follows
- **Remove** - Delete the shuttle from simulation

### Shuttle States

| State | Description |
|-------|-------------|
| `running` | Moving along assigned route |
| `stopped` | Stationary (not at a designated stop) |
| `at_stop` | At a bus stop |
| `out_of_service` | Outside geofence, not tracked |

### Viewing Data

- **Events** - View geofence entry/exit events
- **Routes** - See available routes
- **Clear Events** - Reset test data

## Using the Test Server API Directly

You can also control shuttles via API calls:

```bash
# Add a new shuttle
curl -X POST http://localhost:4000/api/shuttles

# List all shuttles
curl http://localhost:4000/api/shuttles

# Set shuttle state
curl -X POST http://localhost:4000/api/shuttles/1/set-next-state \
  -H "Content-Type: application/json" \
  -d '{"state": "running", "route": "East Route"}'

# Get vehicle locations (Samsara API format)
curl http://localhost:4000/fleet/vehicles/stats

# View today's events
curl http://localhost:4000/api/events/today

# Clear events
curl -X DELETE http://localhost:4000/api/events/today
```

## How It Works

### Data Flow

```
┌─────────────────┐     ┌─────────────────┐
│   Test Client   │────▶│   Test Server   │
│  (Control UI)   │     │ (Mock Samsara)  │
└─────────────────┘     └────────┬────────┘
                                 │
                                 │ Polls every 5s
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│    Frontend     │◀────│     Backend     │
│  (React App)    │     │   (FastAPI)     │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   PostgreSQL    │
                        │    Database     │
                        └─────────────────┘
```

### Route Simulation

Shuttles move along real route polylines from `shared/routes.json`:

1. When a shuttle is set to "running", it starts at the beginning of the assigned route
2. Position updates follow the polyline coordinates
3. Speed varies based on state
4. Geofence events are generated when shuttles enter/exit the campus area

## Testing Scenarios

### Scenario 1: Basic Functionality

1. Start the test environment
2. Add 2-3 shuttles
3. Assign different routes to each
4. Set all to "running"
5. Open the main frontend and verify shuttles appear on the map

### Scenario 2: Geofence Events

1. Add a shuttle and set to "running"
2. Wait for it to enter the geofence (check events)
3. Set to "out_of_service"
4. Verify geofence exit event
5. Check that shuttle disappears from the main frontend

### Scenario 3: Multiple Routes

1. Add shuttles for each available route
2. Verify each shuttle follows its assigned route
3. Check that route colors match on the frontend

### Scenario 4: Data Persistence

1. Add shuttles and let them run
2. Stop the test server
3. Restart the test server
4. Verify shuttles resume from their last positions

## Troubleshooting

### Shuttles Not Appearing on Frontend

1. Check that the backend worker is running
2. Verify test server is running on port 4000
3. Check backend logs for errors
4. Ensure shuttles are in "running" state and inside geofence

### Test Client Won't Connect

1. Verify test server is running: `curl http://localhost:4000/api/shuttles`
2. Check CORS settings if running on different ports
3. Check browser console for errors

### Database Errors

```bash
# Reset database
docker compose down -v
docker compose up postgres redis
alembic -c backend/alembic.ini upgrade head
```

### Port Conflicts

```bash
# Check what's using a port
lsof -i :4000
lsof -i :5173

# Change ports in docker-compose.yml or .env
TEST_BACKEND_PORT=4001
TEST_FRONTEND_PORT=5175
```

## Automated Testing

For automated testing scenarios, see the `AutoTest.js` module in `test-client/src/`. This provides utilities for:

- Programmatically adding/removing shuttles
- Simulating realistic shuttle behavior over time
- Running reproducible test scenarios

## Next Steps

- See [test-server/README.md](../test-server/README.md) for test server details
- See [test-client/README.md](../test-client/README.md) for test client details
- See [INSTALLATION.md](INSTALLATION.md) for full setup instructions
