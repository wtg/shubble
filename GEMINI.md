# Shubble - Gemini Development Guide

Shubble is a real-time shuttle tracking application for Rensselaer Polytechnic Institute (RPI).

## Tech Stack
- **Backend**: FastAPI (Python 3.13) with Async SQLAlchemy 2.0 & PostgreSQL 17.
- **Frontend**: React 19 + TypeScript + Vite.
- **Cache**: Redis 7.
- **Database**: PostgreSQL 17.
- **GPS Data**: Samsara API (polling via background worker).
- **Package Manager**: [uv](https://docs.astral.sh/uv/) for Python dependency management.

## Core Commands

### Backend (from root)
- **Install**: `uv sync` (or `uv sync --group ml` for ML dependencies)
- **Run API**: `uv run uvicorn shubble:app --reload`
- **Run Worker**: `uv run python -m backend.worker`
- **Migrations**: `uv run alembic -c backend/alembic.ini upgrade head`

### Frontend (from `frontend/`)
- **Install**: `npm install`
- **Dev**: `npm run dev` (Parses schedules and copies shared data to `src/`)
- **Build**: `npm run build`
- **Lint**: `npm run lint`

### Docker (from root)
- **Full System**: `docker-compose --profile backend --profile frontend up`
- **Dev with Mock API**: `docker-compose --profile test --profile backend up`

## Project Structure
- `backend/`: FastAPI application (`backend/fastapi/`) and background worker (`backend/worker/`).
- `frontend/`: React application.
- `shared/`: Shared JSON data (routes, schedules) and Python/JS utilities used by both ends.
- `test/`: Test environment (mock Samsara API, test client UI, example test files).
- `alembic/`: Database migration scripts.
- `ml/`: Machine learning models and pipelines for shuttle arrival prediction.

## Machine Learning
- **Pipeline**: Run `PYTHONPATH=. python3 ml/pipelines.py` to preprocess data and generate train/test splits. Supports disk-based and in-memory DataFrame processing.
- **Cache**: Managed by `ml/cache.py`. Files are stored in `ml/cache/` (shared, arima, lstm subdirectories).
- **Models**: 
  - LSTM models for stop-based ETA predictions (`ml/models/lstm.py`).
  - ARIMA models for speed and next-state forecasting (`ml/models/arima.py`).
- **Inference**: Handled by `backend/worker/data.py`, leveraging cached daily data and pre-trained models.

## Development Guidelines
- **Async Operations**: All database and network calls in the backend MUST use `async`/`await`.
- **Data Caching**:
  - Daily vehicle data is cached in Redis (`locations:{date}`) via `backend/cache_dataframe.py`.
  - The cache stores DataFrames processed through ML pipelines (with routes, segments, and stops).
  - Use `get_today_dataframe()` for cached access and `update_today_dataframe()` for automatic cache loading or incremental refreshes.
- **Real-time Inference**: 
  - The worker triggers `generate_and_save_predictions()` after fetching new GPS points. 
  - Results are saved to `etas` (JSON) and `predicted_locations` tables.
- **Shared Resources**: Route polylines and schedules are in `shared/*.json`. The frontend `dev` and `build` scripts copy these to `frontend/src/`.
- **Database**: Use SQLAlchemy models in `backend/models.py`. 
  - `VehicleLocation`: Raw GPS history.
  - `ETA`: Predicted arrival times.
  - `PredictedLocation`: ARIMA-based state forecasts.
- **Timezones**: Always use `America/New_York` (Campus Time) for user-facing logic; store as UTC in the database. Use `backend/time_utils.py`.
- **Route Matching**: Use `shared/stops.py` for haversine-based route matching from GPS points.

## Testing & Validation
- **Mock Server**: Use `test/server/server.py` to simulate shuttle movement for UI testing.
- **Linting**: Run `npm run lint` in the frontend. Ensure `.yamllint` is respected for CI.
