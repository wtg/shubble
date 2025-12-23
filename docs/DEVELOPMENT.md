# Development Guide

This guide covers all development workflows, commands, and tools for working on Shubble. For testing instructions, see [testing/README.md](../testing/README.md).

## Quick Reference

### Start Development Environment

**Native (Recommended for active development):**
```bash
# Terminal 1: Backend
flask --app backend:create_app run

# Terminal 2: Frontend
npm run dev

# Terminal 3: Worker
python -m backend.worker
```

**Docker:**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### Common Development Commands

```bash
# Frontend
npm run dev          # Start dev server
npm run build        # Build for production
npm run lint         # Run ESLint

# Backend
flask --app backend:create_app run     # Start dev server
python -m backend.worker                # Run background worker

# Database
flask --app backend:create_app db migrate -m "description"  # Create migration
flask --app backend:create_app db upgrade                   # Apply migrations
flask --app backend:create_app db downgrade                 # Rollback migration
```

## Native Development

### Prerequisites

Ensure you have completed the [installation steps](INSTALLATION.md) before proceeding.

### Starting Services

#### Backend API

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start Flask development server
flask --app backend:create_app run

# Or with custom port
flask --app backend:create_app run --port 5002

# With debug mode and auto-reload
FLASK_DEBUG=true flask --app backend:create_app run
```

The backend will be available at http://localhost:5001 (or your custom port).

**Flask will automatically:**
- Reload when code changes
- Show detailed error pages
- Enable the interactive debugger

#### Worker Process

The worker polls the Samsara API every 5 seconds for GPS updates:

```bash
# In a new terminal
source venv/bin/activate
python -m backend.worker
```

**Worker options:**
- Polls every 5 seconds (configurable in `backend/worker.py`)
- Only polls vehicles currently in geofence
- Updates driver assignments
- Invalidates relevant caches

#### Frontend

```bash
# Start Vite development server
npm run dev

# With custom port
npm run dev -- --port 5174

# With custom host (for network access)
npm run dev -- --host 0.0.0.0
```

The frontend will be available at http://localhost:5173.

**Vite HMR features:**
- Instant hot module replacement
- Fast refresh for React components
- Source maps for debugging
- Auto-proxy to backend at /api/*

### Frontend Development

#### Building

```bash
# Production build
npm run build

# Build output will be in frontend/dist/

# Preview production build locally
npm run preview
```

#### Linting

```bash
# Run ESLint
npm run lint

# Auto-fix issues
npm run lint -- --fix
```

#### Updating Schedule Data

When you modify `data/schedule.json`:

```bash
# Regenerate aggregated schedule
node data/parseSchedule.js

# Or run dev (automatically runs parseSchedule)
npm run dev
```

### Backend Development

#### Database Migrations

**Creating a migration:**

1. Modify `backend/models.py` to add/change database models
2. Generate migration:
   ```bash
   flask --app backend:create_app db migrate -m "Add new field to Vehicle"
   ```
3. Review the generated migration in `migrations/versions/`
4. Apply migration:
   ```bash
   flask --app backend:create_app db upgrade
   ```

**Common migration commands:**

```bash
# Create new migration
flask --app backend:create_app db migrate -m "description"

# Apply all pending migrations
flask --app backend:create_app db upgrade

# Rollback one migration
flask --app backend:create_app db downgrade

# Show current migration version
flask --app backend:create_app db current

# Show migration history
flask --app backend:create_app db history
```

#### Database Access

**PostgreSQL CLI:**

```bash
# Connect to database
psql postgresql://shubble:shubble@localhost:5432/shubble

# Common psql commands
\dt                # List tables
\d table_name      # Describe table
\q                 # Quit
```

**Useful queries:**

```sql
-- View all vehicles
SELECT * FROM vehicles;

-- Check geofence events today
SELECT * FROM geofence_events
WHERE event_time >= CURRENT_DATE
ORDER BY event_time DESC;

-- View latest locations
SELECT v.name, vl.latitude, vl.longitude, vl.timestamp
FROM vehicle_locations vl
JOIN vehicles v ON v.id = vl.vehicle_id
ORDER BY vl.timestamp DESC
LIMIT 10;
```

#### Redis Access

```bash
# Connect to Redis CLI
redis-cli

# Common Redis commands
KEYS *                    # List all keys
GET key_name              # Get value
DEL key_name              # Delete key
FLUSHALL                  # Clear all data (careful!)
TTL key_name              # Check time-to-live
```

**Cache keys used by Shubble:**
- `vehicle_locations` - Current shuttle positions
- `vehicles_in_geofence` - Set of active vehicle IDs
- `schedule_entries` - Matched schedule results
- `coords:{lat}:{lon}` - Stop detection cache
- `labeled_stops:{date}` - Daily stop visit data

### Environment Configuration

**Backend (.env):**

```bash
# Required
DATABASE_URL=postgresql://shubble:shubble@localhost:5432/shubble
REDIS_URL=redis://localhost:6379/0

# Service URLs
FRONTEND_URL=http://localhost:5173
BACKEND_URL=http://localhost:5001

# Optional (for production)
API_KEY=your_samsara_api_key
SAMSARA_SECRET=your_base64_webhook_secret

# Flask settings
FLASK_ENV=development
FLASK_DEBUG=true
LOG_LEVEL=INFO
```

**Frontend (build-time env vars):**

```bash
# Optional - only needed if changing defaults
VITE_BACKEND_URL=http://localhost:5001
```

## Docker Development

### Starting Services

```bash
# Start all services in background
docker-compose up -d

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f worker

# Stop all services
docker-compose down

# Stop and remove volumes (clears database)
docker-compose down -v
```

### Rebuilding Containers

After changing code or dependencies:

```bash
# Rebuild specific service
docker-compose build backend
docker-compose build frontend

# Rebuild and restart
docker-compose up -d --build

# Force rebuild (ignore cache)
docker-compose build --no-cache
```

### Accessing Services

```bash
# Execute command in running container
docker-compose exec backend bash
docker-compose exec frontend sh

# Access PostgreSQL
docker-compose exec postgres psql -U shubble -d shubble

# Access Redis
docker-compose exec redis redis-cli

# View environment variables
docker-compose exec backend env
```

### Database Operations in Docker

```bash
# Run migrations
docker-compose exec backend flask --app backend:create_app db upgrade

# Create new migration
docker-compose exec backend flask --app backend:create_app db migrate -m "description"

# Access database
docker-compose exec postgres psql -U shubble -d shubble
```

**Note:** Migrations run automatically when the backend container starts.

### Logs and Debugging

```bash
# Follow all logs
docker-compose logs -f

# Follow specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend

# Container status
docker-compose ps

# Resource usage
docker stats
```

### Development with Docker + Native Mix

You can run some services in Docker and others natively for faster iteration:

**Example: Native frontend, Docker backend:**

```bash
# Start backend services only
docker-compose up -d postgres redis backend worker

# Run frontend natively
npm run dev
```

Frontend will automatically proxy API requests to http://localhost:5001 (Docker backend).

**Example: Native backend, Docker infrastructure:**

```bash
# Start infrastructure only
docker-compose up -d postgres redis

# Run backend and frontend natively
flask --app backend:create_app run
npm run dev
python -m backend.worker
```

## Code Organization

### Adding a New API Endpoint

1. Add route in `backend/routes.py`:
   ```python
   @bp.route('/api/my-endpoint')
   def my_endpoint():
       data = {"message": "Hello"}
       return jsonify(data)
   ```

2. Add any database queries if needed
3. Test manually: `curl http://localhost:5001/api/my-endpoint`
4. Add frontend API call in relevant component

### Adding a New Database Table

1. Add model in `backend/models.py`:
   ```python
   class MyModel(db.Model):
       __tablename__ = 'my_table'
       id = db.Column(db.String, primary_key=True)
       name = db.Column(db.String(100))
       created_at = db.Column(db.DateTime, default=datetime.utcnow)
   ```

2. Create migration:
   ```bash
   flask --app backend:create_app db migrate -m "Add my_table"
   ```

3. Review migration in `migrations/versions/`
4. Apply migration:
   ```bash
   flask --app backend:create_app db upgrade
   ```

### Adding a New Route to the Map

1. Update `data/routes.json` with polyline coordinates:
   ```json
   {
     "NEW_ROUTE": {
       "ROUTES": [...],
       "STOPS": ["STOP_1", "STOP_2"],
       "POLYLINE_STOPS": ["STOP_1", "STOP_2"],
       "STOP_1": {
         "NAME": "Stop Name",
         "COORDINATES": [lat, lon]
       }
     }
   }
   ```

2. Update schedule in `data/schedule.json` if needed
3. Regenerate aggregated schedule:
   ```bash
   node data/parseSchedule.js
   ```

4. Restart services to pick up changes

### Modifying the Schedule Algorithm

The schedule matching algorithm is in `data/schedules.py`:

```bash
# After modifying
# Force recompute to test changes
curl "http://localhost:5001/api/matched-schedules?force_recompute=true"
```

Cache is automatically invalidated when:
- New location data is stored
- Manual cache clear: `redis-cli DEL schedule_entries`

## Performance and Debugging

### Frontend Performance

```bash
# Analyze bundle size
npm run build
npx vite-bundle-visualizer

# Profile React components
# Use React DevTools in browser
```

### Backend Performance

```bash
# Profile Python code
python -m cProfile -o output.prof backend/routes.py

# Analyze profile
python -m pstats output.prof

# Check SQL queries
# Add to .env:
SQLALCHEMY_ECHO=true
```

### Cache Debugging

```bash
# Monitor Redis in real-time
redis-cli MONITOR

# Check cache hit/miss
# Add logging in backend code

# Clear specific cache
redis-cli DEL vehicle_locations
```

### Checking Logs

**Native:**
- Flask logs to stdout
- Worker logs to stdout
- Check terminal output

**Docker:**
```bash
docker-compose logs -f backend
docker-compose logs -f worker
```

## Port Reference

Standard ports used in development:

| Service | Native | Docker | Notes |
|---------|--------|--------|-------|
| Frontend | 5173 | 5173 | Vite dev / nginx |
| Backend | 5001 | 5001 | Flask / gunicorn |
| PostgreSQL | 5432 | 5432 | Shared port |
| Redis | 6379 | 6379 | Shared port |
| Test Server | 4000 | 4000 | Mock Samsara API |
| Test Client | 5174 | 5174 | Test UI |

See [PORTS.md](../PORTS.md) for complete port documentation.

## Troubleshooting

### Port Already in Use

```bash
# Find process using port
lsof -i :5001
lsof -i :5173

# Kill process
kill -9 <PID>
```

### Database Connection Issues

```bash
# Check PostgreSQL is running
pg_isready

# macOS (Homebrew)
brew services list
brew services restart postgresql@16

# Linux
sudo systemctl status postgresql
sudo systemctl restart postgresql
```

### Redis Connection Issues

```bash
# Check Redis is running
redis-cli ping

# macOS (Homebrew)
brew services list
brew services restart redis

# Linux
sudo systemctl status redis
sudo systemctl restart redis
```

### Docker Issues

```bash
# Remove all containers and volumes
docker-compose down -v

# Rebuild from scratch
docker-compose build --no-cache

# Check disk space
docker system df

# Clean up unused resources
docker system prune
```

### Migration Issues

```bash
# Check current version
flask --app backend:create_app db current

# Rollback if needed
flask --app backend:create_app db downgrade

# Re-apply
flask --app backend:create_app db upgrade

# If stuck, may need to manually fix in migrations/versions/
```

## Related Documentation

- [INSTALLATION.md](INSTALLATION.md) - Setup instructions
- [testing/README.md](../testing/README.md) - Testing guide (unit tests, mock API)
- [ARCHITECTURE.md](ARCHITECTURE.md) - Project structure
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment
- [../PORTS.md](../PORTS.md) - Complete port reference
