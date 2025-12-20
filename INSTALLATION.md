# Installation Guide

This guide covers setting up Shubble for local development using either Docker or native installation.

## Prerequisites

### For Docker Setup
- Docker Desktop (or Docker Engine + Docker Compose)
- Git

### For Native Setup
- Node.js 20+ and npm
- Python 3.12+
- PostgreSQL 14+
- Redis 7+
- Git

## Option 1: Docker Setup (Recommended)

Docker provides the easiest way to get started with all dependencies included.

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/shuttletracker-new.git
cd shuttletracker-new
```

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your credentials
nano .env
```

Required environment variables:
```bash
DATABASE_URL=postgresql://shubble:shubble@postgres:5432/shubble
REDIS_URL=redis://redis:6379/0
FLASK_ENV=development
FLASK_DEBUG=true
LOG_LEVEL=INFO

# Add your Samsara API credentials
API_KEY=your_samsara_api_key_here
SAMSARA_SECRET=your_base64_encoded_webhook_secret_here
```

### 3. Start All Services

```bash
# Start frontend, backend, worker, PostgreSQL, and Redis
docker-compose up -d

# View logs
docker-compose logs -f
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **PostgreSQL**: localhost:5432 (username: shubble, password: shubble)
- **Redis**: localhost:6379

### 5. Run Database Migrations

```bash
docker-compose exec backend flask --app server:create_app db upgrade
```

### Common Docker Commands

```bash
# View running services
docker-compose ps

# View logs
docker-compose logs -f [service_name]

# Restart a service
docker-compose restart backend

# Rebuild after code changes
docker-compose up -d --build

# Stop all services
docker-compose down

# Stop and remove all data
docker-compose down -v

# Access database
docker-compose exec postgres psql -U shubble -d shubble

# Access Redis
docker-compose exec redis redis-cli
```

## Option 2: Native Installation

For developers who prefer not to use Docker or need better performance.

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/shuttletracker-new.git
cd shuttletracker-new
```

### 2. Install PostgreSQL

**macOS (Homebrew):**
```bash
brew install postgresql@16
brew services start postgresql@16

# Create database and user
psql postgres
CREATE DATABASE shubble;
CREATE USER shubble WITH PASSWORD 'shubble';
GRANT ALL PRIVILEGES ON DATABASE shubble TO shubble;
\q
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql

# Create database and user
sudo -u postgres psql
CREATE DATABASE shubble;
CREATE USER shubble WITH PASSWORD 'shubble';
GRANT ALL PRIVILEGES ON DATABASE shubble TO shubble;
\q
```

### 3. Install Redis

**macOS (Homebrew):**
```bash
brew install redis
brew services start redis
```

**Ubuntu/Debian:**
```bash
sudo apt install redis-server
sudo systemctl start redis-server
```

### 4. Set Up Python Backend

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env
```

Edit `.env` for local setup:
```bash
DATABASE_URL=postgresql://shubble:shubble@localhost:5432/shubble
REDIS_URL=redis://localhost:6379/0
FLASK_ENV=development
FLASK_DEBUG=true
LOG_LEVEL=INFO
API_KEY=your_samsara_api_key_here
SAMSARA_SECRET=your_base64_encoded_webhook_secret_here
```

```bash
# Run database migrations
flask --app server:create_app db upgrade

# Start Flask backend
flask --app server:create_app run
```

Backend will be available at http://localhost:5000

### 5. Set Up Frontend

In a new terminal:

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at http://localhost:5173

### 6. Start Worker Process

In a new terminal:

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start worker
python -m server.worker
```

## Development Workflow

### Making Changes

#### Frontend Changes
```bash
# With Docker
docker-compose restart frontend

# Without Docker
# Changes auto-reload with Vite
```

#### Backend Changes
```bash
# With Docker
docker-compose restart backend

# Without Docker
# Flask auto-reloads in debug mode
```

#### Worker Changes
```bash
# With Docker
docker-compose restart worker

# Without Docker
# Restart the worker process (Ctrl+C then restart)
python -m server.worker
```

### Database Migrations

#### Create a new migration
```bash
# With Docker
docker-compose exec backend flask --app server:create_app db migrate -m "description"

# Without Docker
flask --app server:create_app db migrate -m "description"
```

#### Apply migrations
```bash
# With Docker
docker-compose exec backend flask --app server:create_app db upgrade

# Without Docker
flask --app server:create_app db upgrade
```

### Running Tests

#### Frontend Tests (Vitest)
```bash
# Run tests once
npm test

# Run tests in watch mode
npm test -- --watch

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage

# With Docker
docker-compose exec frontend npm test
```

#### Backend Tests (pytest)
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=server --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::test_vehicle_creation

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# With Docker
docker-compose exec backend pytest
```

### Building for Production

```bash
# Build frontend only
npm run build

# Output will be in client/dist/
```

## Development with Mock Samsara API

If you don't have Samsara API credentials, use the mock server for local development:

```bash
# Terminal 1: Start mock Samsara API server
cd test-server
python server.py
# Mock API will run on http://localhost:4000

# Terminal 2 (optional): Start test client UI
cd test-client
npm install
npm run dev
# UI will be available at http://localhost:5173
```

The mock server (`test-server/`) provides:
- Simulated vehicle movement along routes
- Mock geofence events (entry/exit)
- Fake GPS data for testing
- API endpoints to control shuttle states

The test client UI (`test-client/`) provides:
- Visual interface to manage simulated shuttles
- Manual state control (waiting, entering, looping, exiting)
- Automated test scenario execution from JSON
- Real-time event monitoring

Update your `.env` for mock server:
```bash
FLASK_ENV=development
FLASK_DEBUG=true
# Remove or leave API_KEY empty
# API_KEY=
```

**Important**: `test-server/` and `test-client/` are development tools, not automated test suites. For running automated tests, see the "Running Tests" section above.

## Troubleshooting

### Port Already in Use

```bash
# Find process using port
lsof -i :8000  # Backend
lsof -i :3000  # Frontend
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis

# Kill process or change port in docker-compose.yml
```

### Database Connection Errors

```bash
# Check PostgreSQL is running
# Docker:
docker-compose ps postgres

# Native:
# macOS: brew services list
# Linux: sudo systemctl status postgresql

# Test connection
psql -U shubble -d shubble -h localhost
```

### Redis Connection Errors

```bash
# Check Redis is running
# Docker:
docker-compose ps redis

# Native:
# macOS: brew services list
# Linux: sudo systemctl status redis-server

# Test connection
redis-cli ping  # Should return PONG
```

### Worker Not Updating Locations

1. Check worker logs for errors
2. Verify API_KEY is set correctly
3. Ensure vehicles are in geofence (check `geofence_events` table)
4. Check Redis is accessible

```bash
# With Docker
docker-compose logs worker

# Check geofence events
docker-compose exec postgres psql -U shubble -d shubble
SELECT * FROM geofence_events ORDER BY event_time DESC LIMIT 10;
```

### Frontend Can't Reach Backend

1. Check backend is running on correct port
2. Verify CORS settings if needed
3. Check browser console for errors

```bash
# Test backend
curl http://localhost:8000/api/locations
```

### Module Not Found Errors (Python)

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Node Module Errors

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## Running Tests Locally

Before committing code, make sure all tests pass:

```bash
# Frontend tests
npm test

# Backend tests
pytest

# Run all tests with coverage
npm run test:coverage
pytest --cov=server --cov-report=html
```

## Next Steps

- See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- See [ARCHITECTURE.md](ARCHITECTURE.md) for architecture details
- Read the [README.md](README.md) for project overview

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Open a new issue with details about your problem
4. Contact [Joel McCandless](mailto:mail@joelmccandless.com)
