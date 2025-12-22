# Installation Guide

This guide covers setting up Shubble for local development using either native installation or Docker.

## Prerequisites

### For Native Setup (Recommended for Development)
- Node.js 20+ and npm
- Python 3.12+
- PostgreSQL 14+
- Redis 7+
- Git

### For Docker Setup
- Docker Desktop (or Docker Engine + Docker Compose)
- Git

## Option 1: Native Installation (Recommended for Development)

**Native development is recommended for faster iteration** since code changes auto-reload without rebuilding containers. Docker is better suited for production deployments.

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
flask --app backend:create_app db upgrade

# Start Flask backend
flask --app backend:create_app run
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
python -m backend.worker
```

### Development Benefits of Native Setup

- **Instant reload**: Frontend changes reflect immediately with Vite HMR
- **Faster backend reload**: Flask auto-reloads in debug mode without container rebuilds
- **Direct debugging**: Easy to attach debuggers and inspect processes
- **Lower resource usage**: No Docker overhead
- **Faster testing**: Tests run directly without container execution

## Option 2: Docker Setup

Docker provides an isolated environment with all dependencies included. Better for production-like testing or if you don't want to install dependencies locally.

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
docker-compose exec backend flask --app backend:create_app db upgrade
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

## Development Workflow

### Making Changes

#### Frontend Changes
```bash
# Native (auto-reloads with Vite)
# Just save your files - changes appear instantly

# Docker (requires rebuild)
docker-compose up -d --build frontend
```

#### Backend Changes
```bash
# Native (auto-reloads in debug mode)
# Just save your files - Flask reloads automatically

# Docker (requires restart)
docker-compose restart backend
```

#### Worker Changes
```bash
# Native
# Restart the worker process (Ctrl+C then restart)
python -m backend.worker

# Docker
docker-compose restart worker
```

### Database Migrations

#### Create a new migration
```bash
# Native
flask --app backend:create_app db migrate -m "description"

# Docker
docker-compose exec backend flask --app backend:create_app db migrate -m "description"
```

#### Apply migrations
```bash
# Native
flask --app backend:create_app db upgrade

# Docker
docker-compose exec backend flask --app backend:create_app db upgrade
```

### Running Tests

#### Frontend Tests (Vitest)
```bash
# Native
npm test                      # Run tests once
npm test -- --watch          # Run in watch mode
npm run test:ui              # Run with UI
npm run test:coverage        # Run with coverage

# Docker
docker-compose exec frontend npm test
```

#### Backend Tests (pytest)
```bash
# Native
pytest                                    # Run all tests
pytest --cov=backend --cov-report=html   # Run with coverage
pytest testing/tests/test_models.py      # Run specific test file
pytest -m unit                            # Run only unit tests
pytest -m integration                     # Run only integration tests

# Docker
docker-compose exec backend pytest
```

### Building for Production

```bash
# Build frontend only
npm run build

# Output will be in frontend/dist/
```

## Development with Mock Samsara API

If you don't have Samsara API credentials, use the mock server for local development:

```bash
# Terminal 1: Start mock Samsara API server
cd testing/test-server
python server.py
# Mock API will run on http://localhost:4000

# Terminal 2 (optional): Access test client UI
# The test server serves the UI at http://localhost:4000
```

The test server (`testing/test-server/`) provides:
- Simulated vehicle movement along routes
- Mock geofence events (entry/exit)
- Fake GPS data for testing
- API endpoints to control shuttle states
- Web UI for controlling simulated shuttles

The test client UI provides:
- Visual interface to manage simulated shuttles
- Manual state control (waiting, entering, looping, exiting)
- Automated test scenario execution from JSON
- Real-time event monitoring

Update your `.env` for mock server:
```bash
FLASK_ENV=development
FLASK_DEBUG=true
# Remove or leave API_KEY empty to use mock server
# API_KEY=
```

**Important**: `testing/test-server/` and `testing/test-client/` are development tools, not automated test suites. For running automated tests, see the "Running Tests" section above.

## Troubleshooting

### Port Already in Use

```bash
# Find process using port
lsof -i :8000  # Backend
lsof -i :3000  # Frontend (Docker)
lsof -i :5000  # Backend (native)
lsof -i :5173  # Frontend (native)
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis

# Kill process or change port in docker-compose.yml / .env
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
# Native
# Check worker console output

# Docker
docker-compose logs worker

# Check geofence events
# Native:
psql -U shubble -d shubble
# Docker:
docker-compose exec postgres psql -U shubble -d shubble

SELECT * FROM geofence_events ORDER BY event_time DESC LIMIT 10;
```

### Frontend Can't Reach Backend

1. Check backend is running on correct port
2. Verify CORS settings if needed
3. Check browser console for errors

```bash
# Test backend
# Native:
curl http://localhost:5000/api/locations

# Docker:
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

## Running Tests Before Committing

Always run tests before committing code:

```bash
# Frontend tests
npm test

# Backend tests
pytest

# Run all tests with coverage
npm run test:coverage
pytest --cov=backend --cov-report=html
```

## Quick Start Comparison

### Native (3 terminals)
```bash
# Terminal 1: Backend
source venv/bin/activate
flask --app backend:create_app run

# Terminal 2: Frontend
npm run dev

# Terminal 3: Worker
source venv/bin/activate
python -m backend.worker
```

**Pros**: Fast reloads, easy debugging, lower resource usage
**Cons**: Requires installing dependencies

### Docker (1 terminal)
```bash
docker-compose up -d
docker-compose logs -f
```

**Pros**: Isolated environment, no local dependencies
**Cons**: Slower rebuilds, higher resource usage

## Next Steps

- See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- See [ARCHITECTURE.md](ARCHITECTURE.md) for architecture details
- See [TESTING.md](TESTING.md) for testing guide
- Read the [README.md](../README.md) for project overview

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Open a new issue with details about your problem
4. Contact [Joel McCandless](mailto:mail@joelmccandless.com)
