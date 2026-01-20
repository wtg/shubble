# Shubble Frontend

React + TypeScript frontend for the Shubble shuttle tracking application.

## Tech Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **React Router** - Client-side routing
- **Apple MapKit JS** - Interactive maps

## Project Structure

```
frontend/
├── src/
│   ├── main.tsx              # Entry point (loads config, renders app)
│   ├── App.tsx               # Router setup, main layout
│   ├── index.css             # Global styles
│   │
│   ├── components/           # Shared components
│   │   ├── Navigation.tsx    # Header/footer navigation
│   │   └── ErrorBoundary.tsx # Error handling
│   │
│   ├── locations/            # Live map page
│   │   ├── LiveLocation.tsx  # Main location tracking component
│   │   └── MapKitMap.tsx     # Apple MapKit wrapper
│   │
│   ├── schedule/             # Schedule page
│   ├── dashboard/            # Data analytics page
│   ├── about/                # About page
│   │
│   ├── types/                # TypeScript interfaces
│   └── utils/
│       ├── config.ts         # Runtime configuration loader
│       ├── logger.ts         # Logging utilities
│       └── map.ts            # Map helper functions
│
├── public/                   # Static assets
│   ├── config.json           # Default config for local dev
│   └── *.png                 # Icons and images
│
├── config.template.json      # Template for runtime config
├── package.json
├── vite.config.ts
└── tsconfig.json
```

## Routes

| Path | Component | Description |
|------|-----------|-------------|
| `/` | LiveLocation | Real-time shuttle map (default) |
| `/schedule` | Schedule | Shuttle schedules |
| `/about` | About | About page |
| `/data` | Dashboard | Data analytics |
| `/map` | MapKitMap | Fullscreen map view |

## Configuration

The frontend uses runtime configuration loaded from `/config.json`. This allows environment variables to be set at container startup rather than build time.

### Local Development

Edit `public/config.json`:

```json
{
  "apiBaseUrl": "http://localhost:8000",
  "deployMode": "development"
}
```

### Production (Docker)

The `config.template.json` is processed by `entrypoint.sh` at container startup:

```json
{
  "apiBaseUrl": "${BACKEND_URL}",
  "deployMode": "${DEPLOY_MODE}",
  "mapkitKey": "${MAPKIT_KEY}"
}
```

Environment variables are substituted using `envsubst`.

## Docker

### Development

```bash
# Build and run dev container
docker build -f docker/frontend/Dockerfile.frontend.dev -t shubble-frontend-dev .
docker run -p 3000:3000 shubble-frontend-dev
```

### Production

```bash
# Build production container
docker build -f docker/frontend/Dockerfile.frontend.prod -t shubble-frontend .

# Run with environment variables
docker run -p 80:80 \
  -e BACKEND_URL=https://api.shuttles.rpi.edu \
  -e DEPLOY_MODE=production \
  -e MAPKIT_KEY=your_mapkit_key \
  shubble-frontend
```

## API Integration

The frontend fetches data from the backend API:

```typescript
import config from './utils/config';

// Fetch shuttle locations
const response = await fetch(`${config.apiBaseUrl}/api/locations`);
const locations = await response.json();
```

### Key Endpoints

- `GET /api/locations` - Current shuttle positions
- `GET /api/routes` - Route polylines and stops
- `GET /api/schedule` - Shuttle schedules

## MapKit Integration

The map uses Apple MapKit JS. The MapKit script is loaded in `index.html`:

```html
<script src="https://cdn.apple-mapkit.com/mk/5.x.x/mapkit.core.js"
  crossorigin async
  data-callback="initMapKit"
  data-libraries="map,annotations,overlays,services,user-location">
</script>
```

MapKit is initialized in `MapKitMap.tsx` with a JWT token from the backend.

## Scripts

```bash
npm run dev      # Start dev server with HMR
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Run ESLint
```

## Environment

The build process copies shared data from `../shared/` into `src/shared/`:

```bash
# Executed by npm run dev/build
node ../shared/parseSchedule.js
shx cp -r ../shared/ src/
```

This includes route definitions, schedules, and stop configurations.
