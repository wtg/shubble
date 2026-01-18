# Shubble Test Client

React UI for controlling the mock Samsara API test server.

## Purpose

The test client provides a web interface to:

- Add simulated shuttles
- Queue actions for shuttles (entering, looping, breaks, exiting)
- View action queue and status
- Load test files to batch-create shuttles with predefined actions
- Clear test data

This is useful for development and testing without access to the real Samsara API.

## Project Structure

```
test/client/
├── src/
│   ├── main.tsx              # Entry point
│   ├── App.tsx               # Main UI component
│   ├── App.css               # Styles
│   ├── types.ts              # TypeScript types
│   ├── api/
│   │   ├── config.ts         # Runtime configuration
│   │   └── events.ts         # Event API functions
│   ├── utils/
│   │   ├── shuttles.ts       # Shuttle API functions
│   │   └── testFiles.ts      # Test file loading
│   └── components/
│       ├── Shuttle.tsx       # Shuttle view component
│       └── ShuttleAction.tsx # Queue item component
│
├── public/
│   └── config.json           # Default config for local dev
│
├── config.template.json      # Template for Docker runtime config
└── package.json
```

## Configuration

### Local Development

Edit `public/config.json`:

```json
{
  "apiBaseUrl": "http://localhost:4000"
}
```

### Docker

Environment variables are substituted at container startup:

```bash
docker run -p 5174:80 \
  -e VITE_TEST_BACKEND_URL=http://localhost:4000 \
  shubble-test-client
```

## Docker

```bash
# Build
docker build -f docker/test-client/Dockerfile.test-client -t shubble-test-client .

# Run
docker run -p 5174:80 \
  -e VITE_TEST_BACKEND_URL=http://localhost:4000 \
  shubble-test-client
```

## Usage with Docker Compose

```bash
# Start test environment
docker compose --profile test --profile backend up
```

Access at: http://localhost:5174

## Features

### Shuttle Management

- **Add Shuttle** - Create a new simulated shuttle
- **Queue Actions** - Add actions to a shuttle's queue
- **View Queue** - See pending, in-progress, and completed actions
- **Load Test File** - Import predefined action sequences from JSON files

### Action Types

| Action | Description |
|--------|-------------|
| Entering | Shuttle enters the service area |
| Looping | Shuttle loops on a route (requires route selection) |
| On Break | Shuttle pauses for a duration (requires duration in seconds) |
| Exiting | Shuttle exits the service area |

### Data Management

- **View Event Counts** - See location and geofence event counts
- **Clear Events** - Reset event data (optionally keep shuttles)
- **Clear All** - Remove all shuttles and events

## Test Files

Test files define sequences of actions for shuttles. See `test/files/README.md` for the format specification.

Example test file:

```json
{
  "shuttles": [
    {
      "id": "morning-shift",
      "events": [
        { "type": "entering" },
        { "type": "looping", "route": "NORTH" },
        { "type": "on_break", "duration": 60 },
        { "type": "exiting" }
      ]
    }
  ]
}
```
