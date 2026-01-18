# Test Environment

This directory contains the test environment for simulating shuttle tracking without requiring the real Samsara API.

## Components

### server/

Mock backend that simulates the Samsara API and shuttle behavior. Runs on port 4000.

- Simulates shuttle movement along routes
- Processes action queues (entering, looping, on_break, exiting)
- Sends geofence webhooks to the main backend
- Provides API endpoints for managing test shuttles

### client/

React frontend for controlling test shuttles. Runs on port 5174.

- Add/remove test shuttles
- Queue actions for shuttles
- View action queue and status
- Load test files to batch-create shuttles with predefined actions

### files/

Example test files that define shuttle action sequences. See `files/README.md` for the format specification.

## Running

### With Docker Compose

```bash
# Start the test environment (includes postgres, redis, backend, test-server, test-client)
docker compose --profile test --profile backend up
```

### For Development

```bash
# Terminal 1: Start the test server
cd test/server
uvicorn server:app --port 4000 --reload

# Terminal 2: Start the test client
cd test/client
npm install
npm run dev
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Test Client   │────▶│   Test Server   │
│   (port 5174)   │     │   (port 4000)   │
└─────────────────┘     └────────┬────────┘
                                 │
                                 │ webhooks
                                 ▼
                        ┌─────────────────┐
                        │  Main Backend   │
                        │   (port 8000)   │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   PostgreSQL    │
                        └─────────────────┘
```

The test server simulates shuttles and sends geofence entry/exit webhooks to the main backend, which stores them in the database. The main frontend can then display the simulated shuttle locations.
