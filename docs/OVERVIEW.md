# Documentation Overview

Welcome to the Shubble documentation! This guide will help you find the right information quickly.

## Start Here

📖 **New to Shubble?** Start with the [README.md](../README.md) for an introduction to the project.

## Documentation Structure

### For Users & Contributors

- **[README.md](../README.md)** - Project introduction, features, and quick start
  - What Shubble does
  - Key features
  - Tech stack overview
  - Quick start guide
  - How to contribute

### For Developers

- **[INSTALLATION.md](../INSTALLATION.md)** - Setting up your development environment
  - Docker setup (recommended)
  - Native installation (PostgreSQL, Redis, Python, Node.js)
  - Common development commands
  - Troubleshooting guide

- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - Architecture and technical details
  - System architecture and data flow
  - Database models
  - Critical files and their roles
  - Caching strategy
  - Development workflows
  - Common development tasks

- **[TESTING.md](../TESTING.md)** - Testing guide
  - Running frontend tests (Vitest)
  - Running backend tests (pytest)
  - Writing new tests
  - Coverage reports
  - Best practices

### For DevOps

- **[DEPLOYMENT.md](../DEPLOYMENT.md)** - Production deployment guide
  - Dokploy setup instructions
  - Environment variable reference
  - Service configuration
  - Monitoring and troubleshooting
  - Scaling recommendations

## Quick Links

### Get Started
```bash
# Clone and run with Docker
git clone <repo-url>
cd shuttletracker-new
cp .env.example .env
# Edit .env with your credentials
docker-compose up -d
```

### Common Tasks

| Task | Command |
|------|---------|
| Start development | `docker-compose up -d` |
| View logs | `docker-compose logs -f` |
| Run migrations | `docker-compose exec backend flask --app server:create_app db upgrade` |
| Access database | `docker-compose exec postgres psql -U shubble -d shubble` |
| Build frontend | `npm run build` |
| Lint code | `npm run lint` |

### File Structure

```
shuttletracker-new/
├── README.md              # Start here
├── INSTALLATION.md        # Development setup
├── DEPLOYMENT.md          # Production deployment
├── ARCHITECTURE.md        # Architecture details
├── TESTING.md             # Testing guide
├── docker-compose.yml    # Local development setup
├── .env.example          # Environment template
│
├── client/               # React frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   └── ts/          # TypeScript utilities
│   └── package.json
│
├── server/               # Flask backend
│   ├── __init__.py      # App factory
│   ├── routes.py        # API endpoints
│   ├── worker.py        # Background poller
│   ├── models.py        # Database models
│   └── config.py        # Configuration
│
├── data/                 # Static data & processing
│   ├── schedule.json    # Master schedule
│   ├── routes.json      # Route polylines
│   ├── schedules.py     # Matching algorithm
│   └── stops.py         # Stop definitions
│
├── migrations/           # Database migrations
├── test-server/         # Mock Samsara API
│
└── Dockerfile.*         # Production containers
```

## Need Help?

1. **Installation issues?** → [INSTALLATION.md](../INSTALLATION.md) Troubleshooting section
2. **Architecture questions?** → [ARCHITECTURE.md](../ARCHITECTURE.md)
3. **Deployment problems?** → [DEPLOYMENT.md](../DEPLOYMENT.md) Troubleshooting section
4. **Still stuck?** → Contact [Joel McCandless](mailto:mail@joelmccandless.com)
