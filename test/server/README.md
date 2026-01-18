# Shubble Test Server

Mock Samsara API server for development and testing.

## Purpose

The test server simulates the Samsara Fleet API, providing:

- Mock GPS data for simulated shuttles
- Geofence entry/exit events
- Shuttle state management via action queues
- Route simulation along real polylines

This allows development without access to the real Samsara API or live shuttle data.

## Project Structure

```
test/server/
├── __init__.py       # Package init
├── server.py         # FastAPI application entry point
├── shuttle.py        # Shuttle simulation logic
├── shuttles.py       # Shuttle management routes
├── events.py         # Event management routes
└── mock_samsara.py   # Mock Samsara API routes
```

## API Endpoints

### Shuttle Management

```
GET  /api/shuttles                    # List all shuttles
POST /api/shuttles                    # Add a new shuttle
POST /api/shuttles/{id}/queue         # Queue actions for a shuttle
GET  /api/shuttles/{id}/queue         # Get shuttle's action queue
DELETE /api/shuttles/{id}/queue       # Clear shuttle's pending actions
```

### Mock Samsara API

```
GET /fleet/vehicles/stats             # Vehicle locations (Samsara format)
```

### Events

```
GET    /api/events/counts             # Get event counts
DELETE /api/events                    # Clear events
```

### Routes

```
GET /api/routes                       # Get available routes
```

## Action Types

Shuttles process queued actions:

| Action | Description | Required Fields |
|--------|-------------|-----------------|
| `entering` | Enter service area (triggers geofence entry) | - |
| `looping` | Loop on a route | `route` |
| `on_break` | Pause for a duration | `duration` |
| `exiting` | Exit service area (triggers geofence exit) | - |

## Configuration

Environment variables:

```bash
# Database (for persisting test data)
DATABASE_URL=postgresql://shubble:shubble@localhost:5432/shubble

# Redis
REDIS_URL=redis://localhost:6379/0

# CORS
TEST_FRONTEND_URL=http://localhost:5174
```

## Docker

```bash
# Build
docker build -f docker/Dockerfile.test-server -t shubble-test-server .

# Run
docker run -p 4000:4000 \
  -e DATABASE_URL=postgresql://... \
  shubble-test-server
```

## Usage with Backend

Configure the backend to use the test server:

```bash
# In backend environment
export API_KEY=test
export SAMSARA_API_URL=http://localhost:4000
```

The backend worker will poll the test server instead of the real Samsara API.

## Usage with Docker Compose

```bash
# Start complete test environment
docker compose --profile test --profile backend up

# Services started:
# - postgres (port 5432)
# - redis (port 6379)
# - test-server (port 4000)
# - test-client (port 5174)
# - backend (port 8000)
```

## Example: Add and Control a Shuttle

```bash
# Add a new shuttle
curl -X POST http://localhost:4000/api/shuttles

# Queue actions for the shuttle
curl -X POST http://localhost:4000/api/shuttles/000000000000001/queue \
  -H "Content-Type: application/json" \
  -d '{"actions": [
    {"action": "entering"},
    {"action": "looping", "route": "NORTH"},
    {"action": "exiting"}
  ]}'

# Get shuttle positions (Samsara API format)
curl http://localhost:4000/fleet/vehicles/stats

# Clear test data
curl -X DELETE http://localhost:4000/api/events
```

## Integration with Test Client

The test client (`test/client/`) provides a web UI for controlling shuttles. Start both services:

```bash
# Terminal 1: Test server
cd test/server
uvicorn server:app --port 4000

# Terminal 2: Test client
cd test/client
npm run dev
```

Access the test client at http://localhost:5174
