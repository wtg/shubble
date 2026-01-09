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

## Quick Start

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed setup instructions.

```bash
# Clone and setup
git clone git@github.com:wtg/shubble.git
cd shubble
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Database setup
createdb shubble
cd backend && alembic upgrade head && cd ..

# Run backend (in two terminals)
uvicorn shubble:app --reload --port 8000   # Terminal 1: API server
python -m backend.worker                    # Terminal 2: GPS worker

# Run frontend
cd frontend && npm install && npm run dev
```

## Project Structure

```
shuttletracker-new/
├── backend/             # FastAPI backend
│   ├── flask/          # API routes and app factory
│   └── worker/         # Background GPS polling worker
├── frontend/            # React frontend
├── ml/                  # Machine learning pipelines
│   ├── pipelines.py    # Data processing pipelines
│   ├── deploy/         # Model deployment utilities
│   ├── models/         # ARIMA and LSTM implementations
│   └── cache/          # Cached datasets and models
├── shared/              # Shared resources (routes, schedules, stops)
├── alembic/             # Database migrations
└── test-server/         # Mock Samsara API for development
```

## ML Pipelines

Run complete ML training pipelines:

```bash
# ARIMA pipeline (speed prediction)
python -m ml.pipelines arima --segment --p 3 --d 0 --q 2

# LSTM pipeline (ETA prediction)
python -m ml.pipelines lstm --stops --train --epochs 20

# See all options
python -m ml.pipelines --help
```

See [ml/README.md](ml/README.md) for detailed ML documentation.

## Contributing

This is an open source project under Rensselaer Polytechnic Institute's
Rensselaer Center for Open Source (RCOS). We welcome all contributions:

- **Code**: Bug fixes, new features, optimizations
- **Documentation**: Improve guides, add examples
- **Design**: UI/UX improvements, accessibility
- **Testing**: Report bugs, write tests

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

## License

[License information]

## Contact

For questions or help getting started, reach out to:
- [Joel McCandless](mailto:mail@joelmccandless.com)
- RPI Web Technologies Group (WTG)

---

**Live site**: [https://shuttles.rpi.edu](https://shuttles.rpi.edu)
**Staging**: [https://staging-web-shuttles.rpi.edu](https://staging-web-shuttles.rpi.edu)
