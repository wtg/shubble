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

### Option 1: Native Setup (Recommended for Development)

**Faster iteration with instant reloads!**

```bash
# 1. Clone and install
git clone git@github.com:wtg/shubble.git
cd shubble

# 2. Install PostgreSQL and Redis (macOS)
brew install postgresql@16 redis
brew services start postgresql@16 redis

# 3. Set up backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your database settings

# 4. Run migrations
flask --app backend:create_app db upgrade

# 5. Start services (3 terminals)
# Terminal 1: Backend
flask --app backend:create_app run

# Terminal 2: Frontend
npm install && npm run dev

# Terminal 3: Worker
python -m backend.worker

# Access at:
# Frontend: http://localhost:5173
# Backend: http://localhost:5001
```

### Option 2: Docker Setup

```bash
# 1. Clone the repository
git clone git@github.com:wtg/shubble.git
cd shubble

# 2. Set up environment
cp .env.example .env
# Edit .env with your Samsara API credentials

# 3. Start all services
docker-compose up -d

# OR: Start with Mock Samsara API for development/testing
docker-compose --profile dev up -d

# 4. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# Mock Samsara API: http://localhost:4000 (with --profile dev)
# Test Client UI: http://localhost:4001 (with --profile dev)
```

**Why Native?**
- Instant frontend reloads with Vite HMR
- Faster backend reloads (no container rebuilds)
- Easier debugging and lower resource usage
- Direct test execution

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed setup instructions.

## Documentation

- **[docs/INSTALLATION.md](docs/INSTALLATION.md)** - Local development setup (native & Docker)
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Production deployment guide (Dokploy)
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical architecture details
- **[docs/TESTING.md](docs/TESTING.md)** - Testing guide (Vitest & pytest)
- **[CLAUDE.md](CLAUDE.md)** - Quick reference for common commands

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
│           Backend Services           │
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

### Native Development

```bash
# Run tests
npm test                    # Frontend tests (vitest)
pytest                      # Backend tests (pytest)

# Database migrations
flask --app backend:create_app db migrate -m "description"
flask --app backend:create_app db upgrade

# Build frontend for production
npm run build               # Output: frontend/dist/
```

### Docker Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Run automated tests
docker-compose exec frontend npm test
docker-compose exec backend pytest

# Run database migrations
docker-compose exec backend flask --app backend:create_app db upgrade

# Access database
docker-compose exec postgres psql -U shubble -d shubble

# Stop all services
docker-compose down
```

### Development without Samsara API

Use the mock server for local development without API credentials:

**Native:**
```bash
# Terminal 1: Start mock Samsara API
cd testing/test-server
python server.py

# Terminal 2: Start test client UI
cd testing/test-client
npm install && npm run dev

# In .env, leave API_KEY empty or remove it
# Backend will automatically use mock server in development
```

**Docker:**
```bash
# Start all services including mock Samsara API and test client
docker-compose --profile dev up -d
# Mock server: http://localhost:4000
# Test client UI: http://localhost:4001
```

The mock server provides:
- Simulated vehicle movement along routes
- Mock geofence events
- Separate test client UI for controlling shuttles

**Port Reference:**
| Service | Native Port | Docker Port |
|---------|-------------|-------------|
| Test Server API | 4000 | 4000 |
| Test Client UI | 5173 | 4001 |

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for more details.

## Project Structure

```
shubble/
├── backend/              # Flask backend application
├── frontend/             # React frontend application
├── data/                 # Static data and algorithms
├── testing/              # All testing infrastructure
│   ├── tests/           # Automated pytest tests
│   ├── test-server/     # Mock Samsara API (dev tool)
│   └── test-client/     # UI for controlling mock shuttles
├── docker/              # Docker configuration
├── docs/                # Documentation
├── migrations/          # Database migrations
└── docker-compose.yml   # Docker services configuration
```

## Contributing

This is an open source project under Rensselaer Polytechnic Institute's Rensselaer Center for Open Source (RCOS). We welcome all contributions, whether it's code, documentation, or design.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting:
   ```bash
   npm test && pytest
   npm run lint
   ```
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Tips

- **Use native setup** for faster iteration during development
- **Run tests** before committing (`npm test && pytest`)
- **Check coverage** with `npm run test:coverage && pytest --cov=backend`
- **Follow existing patterns** in the codebase
- **Update documentation** when adding features

### Getting Help

If you have questions or want help getting started:
- Check the [docs/](docs/) directory for guides
- Review [CLAUDE.md](CLAUDE.md) for common commands
- Open an issue on GitHub
- Contact [Joel McCandless](mailto:mail@joelmccandless.com)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Rensselaer Center for Open Source (RCOS)
- Samsara for GPS tracking API
- Apple for MapKit JS
