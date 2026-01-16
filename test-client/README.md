# Shubble Test Client

React UI for controlling the mock Samsara API test server.

## Purpose

The test client provides a web interface to:

- Add/remove simulated shuttles
- Control shuttle states (running, stopped, out of service)
- Assign shuttles to routes
- View real-time shuttle positions
- Clear test data

This is useful for development and testing without access to the real Samsara API.

## Project Structure

```
test-client/
├── src/
│   ├── main.jsx        # Entry point (loads config)
│   ├── App.jsx         # Main UI component
│   ├── api.js          # API wrapper functions
│   ├── config.js       # Runtime configuration
│   ├── AutoTest.js     # Automated testing utilities
│   └── utils.js        # Helper functions
│
├── public/
│   └── config.json     # Default config for local dev
│
├── config.template.json  # Template for Docker runtime config
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

## API Integration

The test client communicates with the test server:

```javascript
import config from './config.js';

// Fetch shuttles
fetch(`${config.apiBaseUrl}/api/shuttles`);

// Add a new shuttle
fetch(`${config.apiBaseUrl}/api/shuttles`, { method: 'POST' });

// Set shuttle state
fetch(`${config.apiBaseUrl}/api/shuttles/${id}/set-next-state`, {
  method: 'POST',
  body: JSON.stringify({ state: 'running', route: 'East Route' })
});
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
# Start test environment (test-server + test-client + backend)
docker-compose --profile test --profile backend up
```

Access at: http://localhost:5174

## Features

### Shuttle Management

- **Add Shuttle** - Create a new simulated shuttle
- **Remove Shuttle** - Delete a shuttle from simulation
- **Set State** - Change shuttle state (running, stopped, etc.)
- **Assign Route** - Assign shuttle to a specific route

### Data Management

- **View Events** - See geofence entry/exit events
- **Clear Events** - Reset test data
- **View Routes** - Display available routes

### Auto Test

The `AutoTest.js` module provides automated testing capabilities for simulating realistic shuttle behavior over time.
