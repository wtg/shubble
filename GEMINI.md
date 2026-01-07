# Shubble - Gemini Development Guide

Shubble is a real-time shuttle tracking application for Rensselaer Polytechnic Institute (RPI).

## Tech Stack
- **Backend**: FastAPI (Python 3.13) with Async SQLAlchemy 2.0 & PostgreSQL 17.
- **Frontend**: React 19 + TypeScript + Vite.
- **Cache**: Redis 7.
- **Database**: PostgreSQL 17.
- **GPS Data**: Samsara API (polling via background worker).

## Core Commands

### Backend (from root)
- **Install**: `pip install -r backend/requirements.txt`
- **Run API**: `uvicorn shubble:app --reload`
- **Run Worker**: `python -m backend.worker`
- **Migrations**: `alembic upgrade head` (from `backend/` directory)

### Frontend (from `frontend/`)
- **Install**: `npm install`
- **Dev**: `npm run dev` (Parses schedules and copies shared data to `src/`)
- **Build**: `npm run build`
- **Lint**: `npm run lint`

### Docker (from root)
- **Full System**: `docker-compose --profile backend --profile frontend up`
- **Dev with Mock API**: `docker-compose --profile test --profile backend up`

## Project Structure
- `backend/`: FastAPI application (located in `backend/flask/` - note that this is NOT a Flask app) and background worker (`backend/worker/`).
- `frontend/`: React application.
- `shared/`: Shared JSON data (routes, schedules) and Python/JS utilities used by both ends.
- `test-server/`: Mock Samsara API for local development without real API keys.
- `alembic/`: Database migration scripts.
- `ml/`: Machine learning models and pipelines for shuttle arrival prediction.

## Machine Learning
- **Pipeline**: Run `PYTHONPATH=. python3 ml/pipelines.py` to preprocess data and generate train/test splits.
- **Models**: ARIMA models are located in `ml/models/arima.py`.
- **Data**: Preprocessed data is cached in `ml/data/preprocessed_vehicle_locations.csv`. Training splits are in `ml/training/train.csv` and `test.csv`.

## Development Guidelines
- **Async Operations**: All database and network calls in the backend MUST use `async`/`await`.
- **Shared Resources**: Route polylines and schedules are in `shared/*.json`. The frontend `dev` and `build` scripts copy these to `frontend/src/`.
- **Database**: Use SQLAlchemy models in `backend/models.py`. Ensure indices are added for performance on `vehicle_id` and `timestamp`.
- **Caching**: Use the `@cache` decorator in `routes.py` for endpoints like `/api/locations` (60s TTL).
- **Timezones**: Always use `America/New_York` (Campus Time) for user-facing logic; store as UTC in the database. Use `backend/time_utils.py`.
- **Route Matching**: Use `shared/stops.py` for haversine-based route matching from GPS points.

## Testing & Validation
- **Mock Server**: Use `test-server/server.py` to simulate shuttle movement for UI testing.
- **Linting**: Run `npm run lint` in the frontend. Ensure `.yamllint` is respected for CI.
