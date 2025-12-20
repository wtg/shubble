# Shubble - RPI Shuttle Tracker

> Making RPI Shuttles Reliable, Predictable, and Accountable with Real Time Data

Shubble is a real-time shuttle tracking application that helps RPI students track campus shuttles. It integrates with GPS tracking to show live shuttle locations, expected schedules, and route information on an interactive map.

## Features

- **Real-time Tracking**: Live GPS tracking of shuttle locations with 5-second updates
- **Route Visualization**: Interactive map showing all shuttle routes and stops
- **Schedule Matching**: Intelligent algorithm matches shuttles to their scheduled routes
- **Driver Information**: Shows which driver is operating each shuttle
- **Schedule Display**: View expected arrival times for all routes
- **Progressive Web App**: Install on mobile devices for app-like experience
- **Animated Movement**: Smooth shuttle animations between GPS updates

## Tech Stack

- **Frontend**: React 19 + TypeScript + Vite + Apple MapKit JS
- **Backend**: Flask (Python) + SQLAlchemy + PostgreSQL
- **Cache**: Redis for performance optimization
- **Worker**: Background service polling Samsara GPS API
- **Deployment**: Docker containers (Dokploy/Dokku)

## Quick Start

### Using Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/shuttletracker-new.git
cd shuttletracker-new

# 2. Set up environment
cp .env.example .env
# Edit .env with your Samsara API credentials

# 3. Start all services
docker-compose up -d

# 4. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
```

### Manual Setup

See [INSTALLATION.md](INSTALLATION.md) for detailed installation instructions.

## Documentation

- **[INSTALLATION.md](INSTALLATION.md)** - Local development setup (Docker & native)
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment guide (Dokploy)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture and development guide
- **[TESTING.md](TESTING.md)** - Testing guide (Vitest & pytest)

## Architecture Overview

```
┌─────────────┐
│   Samsara   │ GPS Provider
│   GPS API   │
└──────┬──────┘
       │ Webhooks (geofence events)
       │ Polling (location data)
       ↓
┌──────────────────────────────────────┐
│           Backend Services            │
├──────────────┬───────────────────────┤
│  Flask API   │  Background Worker    │
│  (Routes)    │  (GPS Polling)        │
└──────┬───────┴──────┬────────────────┘
       │              │
       ↓              ↓
┌─────────────────────────────────┐
│     PostgreSQL + Redis          │
│  (Storage + Cache)              │
└─────────────┬───────────────────┘
              │
              ↓
       ┌──────────────┐
       │   React UI   │
       │  (MapKit JS) │
       └──────────────┘
```

### How It Works

1. **GPS Tracking**: Samsara API sends geofence webhooks when shuttles enter/exit campus
2. **Location Polling**: Background worker polls GPS data every 5 seconds for active shuttles
3. **Schedule Matching**: Hungarian algorithm matches shuttles to scheduled routes
4. **Real-time Updates**: Frontend polls backend every 5 seconds for latest positions
5. **Map Display**: Apple MapKit displays shuttles with smooth animations

## API Endpoints

- `GET /api/locations` - Current shuttle positions with route matching
- `GET /api/routes` - Route polylines and stop coordinates
- `GET /api/schedule` - Shuttle schedules
- `GET /api/matched-schedules` - Algorithm-matched schedule assignments
- `POST /api/webhook` - Samsara geofence webhook handler
- `GET /api/today` - All location data for current day

## Development

```bash
# Start all services with Docker
docker-compose up -d

# View logs
docker-compose logs -f

# Run automated tests
npm test                    # Frontend tests (vitest)
pytest                      # Backend tests (pytest)

# Run database migrations
docker-compose exec backend flask --app server:create_app db upgrade

# Access database
docker-compose exec postgres psql -U shubble -d shubble

# Stop all services
docker-compose down
```

### Development without Samsara API

Use the mock server and test client for local development:

```bash
# Start mock Samsara API
cd test-server && python server.py

# Start test client UI (optional)
cd test-client && npm install && npm run dev
```

See [INSTALLATION.md](INSTALLATION.md) for more development commands.

## Contributing

This is an open source project under Rensselaer Polytechnic Institute's Rensselaer Center for Open Source (RCOS). We welcome all contributions, whether it's code, documentation, or design.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Getting Help

If you have questions or want help getting started, please reach out to [Joel McCandless](mailto:mail@joelmccandless.com).

## License

[Add your license here]

## Acknowledgments

- Rensselaer Center for Open Source (RCOS)
- Samsara for GPS tracking API
- Apple for MapKit JS
