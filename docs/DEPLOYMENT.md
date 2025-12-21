# Production Deployment Guide

This guide covers deploying Shubble to production using Docker containers with Dokploy.

> **Note**: For local development setup, see [INSTALLATION.md](INSTALLATION.md)

## Architecture

The application consists of three services:
- **Frontend**: React app served via nginx
- **Backend**: Flask API server with Gunicorn
- **Worker**: Background process for polling Samsara GPS API

Plus two dependencies:
- **PostgreSQL**: Database for vehicles, locations, events, drivers
- **Redis**: Cache for locations, schedules, and stop detection

## Prerequisites

1. Dokploy instance running
2. GitHub repository connected to Dokploy
3. Samsara API credentials (API_KEY and SAMSARA_SECRET)

## Dokploy Setup

### 1. Create Database Services

In Dokploy, create the following services:

#### PostgreSQL Database
- Type: PostgreSQL
- Name: `shubble-postgres`
- Database: `shubble`
- Username: `shubble`
- Password: (generate secure password)
- Note the connection string: `postgresql://shubble:PASSWORD@shubble-postgres:5432/shubble`

#### Redis Cache
- Type: Redis
- Name: `shubble-redis`
- Note the connection string: `redis://shubble-redis:6379/0`

### 2. Create Application Services

#### Backend Service
- Name: `shubble-backend`
- Source: GitHub repository
- Dockerfile: `Dockerfile.backend`
- Port: `8000`
- Environment Variables:
  ```
  DATABASE_URL=postgresql://shubble:PASSWORD@shubble-postgres:5432/shubble
  REDIS_URL=redis://shubble-redis:6379/0
  FLASK_ENV=production
  FLASK_DEBUG=false
  LOG_LEVEL=INFO
  API_KEY=<your-samsara-api-key>
  SAMSARA_SECRET=<your-base64-encoded-webhook-secret>
  ```
- Health Check: `/api/locations`
- Depends On: `shubble-postgres`, `shubble-redis`

#### Worker Service
- Name: `shubble-worker`
- Source: GitHub repository
- Dockerfile: `Dockerfile.worker`
- Environment Variables:
  ```
  DATABASE_URL=postgresql://shubble:PASSWORD@shubble-postgres:5432/shubble
  REDIS_URL=redis://shubble-redis:6379/0
  FLASK_ENV=production
  LOG_LEVEL=INFO
  API_KEY=<your-samsara-api-key>
  ```
- Depends On: `shubble-postgres`, `shubble-redis`, `shubble-backend`

#### Frontend Service
- Name: `shubble-frontend`
- Source: GitHub repository
- Dockerfile: `Dockerfile.frontend`
- Port: `80`
- Domain: Configure your custom domain
- Depends On: `shubble-backend`

### 3. Configure Networking

Ensure services can communicate:
- Frontend needs to proxy `/api/*` requests to backend
- Backend needs access to PostgreSQL and Redis
- Worker needs access to PostgreSQL and Redis

#### Nginx Configuration for Frontend

If you need to proxy API requests through the frontend, update the nginx config in `Dockerfile.frontend`:

```nginx
# Add proxy configuration
location /api {
    proxy_pass http://shubble-backend:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

### 4. Configure Samsara Webhook

In your Samsara dashboard:
1. Go to Settings > Webhooks
2. Create new webhook for geofence events
3. URL: `https://your-domain.com/api/webhook`
4. Events: `geofenceEntry`, `geofenceExit`
5. Enable webhook signature (this is your SAMSARA_SECRET)

## Environment Variables Reference

### Backend & Worker
| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `API_KEY` | Yes | Samsara API key for GPS data |
| `SAMSARA_SECRET` | No | Base64-encoded webhook signature secret |
| `FLASK_ENV` | Yes | `production` or `development` |
| `FLASK_DEBUG` | No | `true` or `false` (default: false) |
| `LOG_LEVEL` | No | `INFO`, `DEBUG`, `WARNING`, etc. |

## Local Development with Docker

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Run migrations
docker-compose exec backend flask --app server:create_app db migrate -m "description"
docker-compose exec backend flask --app server:create_app db upgrade

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Deployment Checklist

- [ ] PostgreSQL database created and accessible
- [ ] Redis cache created and accessible
- [ ] Backend service deployed with correct environment variables
- [ ] Worker service deployed and running
- [ ] Frontend service deployed
- [ ] Database migrations ran successfully
- [ ] Samsara webhook configured and pointing to backend
- [ ] Custom domain configured (if applicable)
- [ ] SSL/TLS certificate configured
- [ ] Health checks passing for all services
- [ ] Test API endpoints: `/api/locations`, `/api/routes`, `/api/schedule`
- [ ] Test webhook by triggering geofence event
- [ ] Verify worker is polling GPS data (check logs)

## Monitoring

### Backend Logs
```bash
# Check backend API logs
docker logs shubble-backend -f
```

### Worker Logs
```bash
# Check worker polling logs
docker logs shubble-worker -f
```

Look for:
- Successful GPS data fetches
- Schedule matching computations
- Cache hits/misses
- Database query performance

### Database
```bash
# Connect to PostgreSQL
docker exec -it shubble-postgres psql -U shubble -d shubble

# Check recent locations
SELECT vehicle_id, timestamp, latitude, longitude
FROM vehicle_locations
ORDER BY timestamp DESC
LIMIT 10;

# Check geofence events
SELECT vehicle_id, event_type, event_time
FROM geofence_events
ORDER BY event_time DESC
LIMIT 10;
```

### Redis Cache
```bash
# Connect to Redis
docker exec -it shubble-redis redis-cli

# Check cache keys
KEYS *

# Check specific cache
GET schedule_entries
GET vehicle_locations
```

## Troubleshooting

### Frontend can't reach backend
- Check if API proxy is configured in nginx
- Verify backend service is running and healthy
- Check network connectivity between services

### Worker not updating locations
- Verify API_KEY is correct
- Check worker logs for API errors
- Ensure vehicles are in geofence (check geofence_events table)
- Verify Redis is accessible

### Database migration fails
- Ensure DATABASE_URL is correct
- Check PostgreSQL is running and accessible
- Verify database user has necessary permissions

### Webhook not receiving events
- Verify webhook URL is publicly accessible
- Check SAMSARA_SECRET matches Samsara dashboard
- Test webhook with curl:
  ```bash
  curl -X POST https://your-domain.com/api/webhook \
    -H "Content-Type: application/json" \
    -d '{"test": "data"}'
  ```

## Scaling

### Backend
- Increase gunicorn workers: `--workers 4`
- Add more backend replicas in Dokploy
- Use load balancer for multiple instances

### Worker
- Run single worker instance only (to avoid duplicate polling)
- Adjust polling interval in `server/worker.py` if needed
- Increase cache TTL if API rate limits are hit

### Database
- Use PostgreSQL connection pooling
- Add read replicas for analytics queries
- Increase PostgreSQL resources if slow

### Redis
- Increase Redis memory limit
- Use Redis persistence (AOF) for cache durability
- Consider Redis Cluster for high availability
