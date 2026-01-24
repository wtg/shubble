# Shubble - RPI Shuttle Tracker

Shubble is a real-time shuttle tracking application for Rensselaer Polytechnic Institute (RPI). The system provides live GPS tracking, route information, and schedules through a modern web interface.

## Features

- **Real-time tracking** - Live shuttle locations updated every 5 seconds
- **Route visualization** - Interactive map with route polylines and stops
- **Schedule information** - View shuttle schedules and expected arrival times
- **Driver assignments** - Track which driver is operating each shuttle
- **ML predictions** - ARIMA and LSTM models for ETA predictions
- **Analytics dashboard** - Historical data analysis and insights

## Tech Stack

- **Backend**: FastAPI (Python 3.13) with async support
- **Frontend**: React 19 + TypeScript + Vite
- **Database**: PostgreSQL 17 with SQLAlchemy ORM
- **Cache**: Redis 7 with fastapi-cache
- **Maps**: Apple MapKit JS
- **GPS Data**: Samsara API integration
- **ML**: ARIMA (statsmodels) and LSTM (PyTorch) models

## Project Structure

```
shubble/
├── backend/              # FastAPI backend application
│   ├── fastapi/          # API routes and app factory
│   ├── worker/           # Background GPS polling worker
│   └── alembic/          # Database migrations
├── frontend/             # React + TypeScript frontend
├── ml/                   # Machine learning pipelines
│   ├── models/           # ARIMA and LSTM implementations
│   ├── deploy/           # Model deployment utilities
│   └── data/             # Data loading and preprocessing
├── shared/               # Shared resources (routes, schedules, stops)
├── test/                 # Test environment
│   ├── server/           # Mock Samsara API for development
│   ├── client/           # Test UI for mock server
│   └── files/            # Example test files
└── docker/               # Docker configurations
    ├── backend/          # Backend Dockerfiles (dev/prod)
    ├── frontend/         # Frontend Dockerfiles (dev/prod)
    └── test-client/      # Test client Dockerfile
```

## Getting Started

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for setup instructions and [docs/TESTING.md](docs/TESTING.md) for running the test environment.

## Documentation

- [backend/README.md](backend/README.md) - Backend API documentation
- [frontend/README.md](frontend/README.md) - Frontend development guide
- [ml/README.md](ml/README.md) - Machine learning pipelines
- [test/README.md](test/README.md) - Test environment

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required
DATABASE_URL=postgresql://shubble:shubble@localhost:5432/shubble
REDIS_URL=redis://localhost:6379/0
FRONTEND_URLS=http://localhost:3000

# Samsara API (not needed with test-server)
API_KEY=sms_live_...
SAMSARA_SECRET=...

# Frontend (for Docker deployment)
BACKEND_URL=http://localhost:8000
DEPLOY_MODE=development
MAPKIT_KEY=
```

## Contributing

This is an open source project under Rensselaer Polytechnic Institute's Rensselaer Center for Open Source (RCOS). We welcome all contributions:

- **Code**: Bug fixes, new features, optimizations
- **Documentation**: Improve guides, add examples
- **Design**: UI/UX improvements, accessibility
- **Testing**: Report bugs, write tests

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed setup instructions.

## License

[License information]

## Contact

For questions or help getting started, reach out to:
- [Joel McCandless](mailto:mail@joelmccandless.com)
- RPI Web Technologies Group (WTG)

---

**Live site**: [https://shuttles.rpi.edu](https://shuttles.rpi.edu)
**Staging**: [https://staging-web-shuttles.rpi.edu](https://staging-web-shuttles.rpi.edu)
