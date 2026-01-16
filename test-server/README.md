# Shubble Test Server

Mock Samsara API server for development and testing.

## Purpose

The test server simulates the Samsara Fleet API, providing:

- Mock GPS data for simulated shuttles
- Geofence entry/exit events
- Shuttle state management
- Route simulation along real polylines

This allows development without access to the real Samsara API or live shuttle data.

## Project Structure

```
test-server/
├── __init__.py       # Package init
├── server.py         # FastAPI application and routes
└── shuttle.py        # Shuttle simulation logic
```

## API Endpoints

### Shuttle Management

```
GET  /api/shuttles                    # List all shuttles
POST /api/shuttles                    # Add a new shuttle
POST /api/shuttles/{id}/set-next-state  # Set shuttle state/route
```

### Mock Samsara API

```
GET /fleet/vehicles/stats             # Vehicle locations (Samsara format)
```

### Events

```
GET    /api/events/today              # Get today's geofence events
DELETE /api/events/today              # Clear today's events
```

### Routes

```
GET /api/routes                       # Get available routes
```

## Shuttle States

Shuttles can be in one of several states:

```python
class ShuttleState:
    STOPPED = "stopped"           # Stationary
    RUNNING = "running"           # Moving along route
    AT_STOP = "at_stop"          # At a bus stop
    OUT_OF_SERVICE = "out_of_service"  # Not in geofence
```

## Simulation

### Route Following

Shuttles move along real route polylines from `shared/routes.json`:

1. Shuttle is assigned a route
2. Position updates follow polyline coordinates
3. Speed varies based on state (stopped, running)
4. Geofence events generated on entry/exit

### Position Updates

The server generates realistic GPS data:

```python
{
    "latitude": 42.7284,
    "longitude": -73.6788,
    "heading_degrees": 180,
    "speed_mph": 25,
    "timestamp": "2025-01-15T12:00:00Z"
}
```

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
docker-compose --profile test --profile backend up

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

# Set shuttle to run on East Route
curl -X POST http://localhost:4000/api/shuttles/1/set-next-state \
  -H "Content-Type: application/json" \
  -d '{"state": "running", "route": "East Route"}'

# Get shuttle positions (Samsara API format)
curl http://localhost:4000/fleet/vehicles/stats

# Clear test data
curl -X DELETE http://localhost:4000/api/events/today
```

## Integration with Test Client

The test client (test-client/) provides a web UI for controlling shuttles. Start both services:

```bash
# Terminal 1: Test server
uvicorn test-server.server:app --port 4000

# Terminal 2: Test client
cd test-client && npm run dev
```

Access the test client at http://localhost:5173
