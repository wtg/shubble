# Coding Conventions

**Analysis Date:** 2026-04-05

## Naming Patterns

**Files:**
- Python: `snake_case.py` (e.g., `backend/models.py`, `backend/worker/worker.py`)
- TypeScript/React: `PascalCase.tsx` for components, `camelCase.ts` for utilities (e.g., `frontend/src/components/ErrorBoundary.tsx`, `frontend/src/utils/config.ts`)
- Tests: `test_*.py` prefix for Python, no specific pattern for frontend (tests are minimal)

**Functions:**
- Python: `snake_case` (e.g., `update_locations()`, `compute_per_stop_etas()`, `get_latest_etas()`)
- TypeScript: `camelCase` (e.g., `renderToStaticMarkup()`, `pollLocation()`)
- React Hooks: `use` prefix (e.g., `useStopETAs()`, `useEffect()`)

**Variables:**
- Python: `snake_case` (e.g., `vehicle_ids`, `session_factory`, `mock_etas`)
- TypeScript: `camelCase` (e.g., `selectedRoute`, `setSelectedRoute`, `vehicleAnnotations`)
- React State: Use getter/setter pairs with `const [state, setState]` pattern

**Types:**
- Python: PascalCase for classes (e.g., `Vehicle`, `VehicleLocation`, `Settings`)
- TypeScript: PascalCase for interfaces/types (e.g., `VehicleLocationData`, `StopETA`, `LiveLocationMapKitProps`)
- Discriminated unions in types (e.g., `StopETA` has required `eta`, `vehicle_id`, `route` fields)

**Constants:**
- Python: `UPPER_SNAKE_CASE` for module-level configuration (e.g., `LOG_LEVEL`, `DATABASE_URL`)
- TypeScript: `UPPER_CASE` for true constants (rare; most configuration is imported from `utils/config`)

## Code Style

**Formatting:**
- Python: Inferred to follow PEP 8 conventions based on project structure (no Prettier/Black config found)
- TypeScript/JavaScript: Likely Prettier-formatted via Vite build pipeline (no `.prettierrc` found, but standard React project setup)

**Linting:**
- Python: Ruff available in dev dependencies (`pyproject.toml`), but usage pattern unclear
- TypeScript: ESLint with custom config in `frontend/eslint.config.js` (Flat ESLint format)

**Linting Rules (Frontend):**
- `@typescript-eslint/no-unsafe-member-access`: **error** (strict type safety)
- `@typescript-eslint/no-unsafe-assignment`: **warn** (allows unknown types with warning)
- `@typescript-eslint/no-unsafe-call`: **warn** (allows calling unknown types with warning)
- `@typescript-eslint/no-unused-vars`: **warn** with `argsIgnorePattern: "^_"` (allows intentionally unused params prefixed with `_`)
- `react-hooks/rules-of-hooks`: **error** (enforce Hook Rules)
- `react-hooks/exhaustive-deps`: **warn** (warn on missing dependencies)
- `react/react-in-jsx-scope`: **off** (React 17+ doesn't require explicit import)

## Import Organization

**Python Order:**
1. Standard library (`import asyncio`, `import logging`)
2. Third-party packages (`import httpx`, `from sqlalchemy import ...`)
3. Local backend imports (`from backend.config import settings`, `from backend.models import ...`)
4. Local ml/shared imports (`from ml.deploy.lstm import ...`, `from shared.stops import Stops`)

**TypeScript Order:**
1. React imports (`import { useState, useEffect } from 'react'`)
2. React Router (`import { BrowserRouter, Routes } from 'react-router'`)
3. Third-party UI/library imports (`import { renderToStaticMarkup } from 'react-dom/server'`)
4. Local type imports (`import type { ShuttleRouteData } from './types/route'`)
5. Local component/utility imports (`import Navigation from './components/Navigation'`)
6. Data imports (JSON, constants) (`import routeData from './shared/routes.json'`)
7. Styles (`import './App.css'`)

**Path Aliases:**
- TypeScript: No path aliases configured in `tsconfig.json` (uses relative imports)
- Shared build process copies `/shared/` into `src/shared/` at build time (see `frontend/package.json` scripts)

## Error Handling

**Python Patterns:**
- Async functions use try/except blocks (e.g., in `backend/worker/worker.py` lines 62-90)
- API errors logged with `logger.error()` before returning error response
- HTTP non-200 responses handled explicitly: `if response.status_code != 200: logger.error(...); return []`
- Optional returns: Functions return empty dict `{}` or empty list `[]` on error, not `None` (e.g., `compute_per_stop_etas` returns `{}`)
- Async context managers use `async with` for resource cleanup (e.g., `async with httpx.AsyncClient() as client:`)

**TypeScript/React Patterns:**
- Error Boundary component (`frontend/src/components/ErrorBoundary.tsx`) catches React component errors
- ErrorBoundary uses `getDerivedStateFromError()` and `componentDidCatch()` lifecycle methods
- Error messages logged to console: `console.error('ErrorBoundary caught an error:', error, errorInfo)`
- Graceful UI fallback: Shows error banner with reload button instead of crashing
- Fetch errors in React effects: Use AbortController for cancellation (`new AbortController()`)

**Validation:**
- Pydantic models validate all input in Python backend (e.g., `Settings` class in `backend/config.py`)
- Type annotations throughout TypeScript prevent runtime type errors
- No defensive null-checks; rely on TypeScript strict mode

## Logging

**Framework:** Python `logging` stdlib; frontend uses `console.log/console.error`

**Python Patterns:**
- Logger created per module: `logger = logging.getLogger(__name__)`
- Log level configured from environment: `settings.get_log_level(component)` (supports per-component levels)
- Logging setup in module initialization (see `backend/worker/worker.py` lines 20-28, `backend/fastapi/__init__.py` lines 13-22)
- Log messages include context: `logger.error(f"API error: {response.status_code} {response.text}")`
- Lifecycle events logged at startup/shutdown (e.g., "Starting up FastAPI application...", "Database engine initialized")

**Frontend Patterns:**
- `console.error()` for exceptions and boundaries
- `console.log()` for development (no structured logging observed)

## Comments

**When to Comment:**
- **Docstrings required**: All module, function, and class definitions include docstrings
- Module docstrings: Single-line summary (e.g., `"""Async background worker for fetching vehicle data from Samsara API."""`)
- Function docstrings: Args, Returns, purpose (e.g., in `backend/config.py` lines 59-68)
- Inline comments: Used for non-obvious logic (e.g., in `backend/fastapi/routes.py` line 48: `# lazy="raise" prevents accidental N+1 queries`)
- TODO comments: Rare but present (e.g., `frontend/src/locations/LiveLocation.tsx` line 15: `// TODO: figure out how to make this type correct...`)

**JSDoc/TSDoc:**
- React components use TypeScript interfaces for prop documentation (e.g., `interface ErrorBoundaryProps`, `type LiveLocationMapKitProps`)
- No explicit JSDoc comments observed; types serve as documentation

## Function Design

**Size:** Small, single-responsibility functions
- Python async workers: 15-60 lines for core logic (e.g., `update_locations()` has ~100 lines with pagination loop)
- TypeScript components: 40-120 lines (e.g., `ErrorBoundary` is 47 lines, `LiveLocation` is 49 lines)
- React hooks: Extract complex logic into custom hooks (e.g., `useStopETAs()` for shared ETA fetching)

**Parameters:**
- Python: Use explicit positional args for required params, `*args`/`**kwargs` avoided
- Python async: Inject dependencies (e.g., `session_factory`, `cache` decorator) rather than global state
- TypeScript: Destructure props in function signature (e.g., `{ routeData, selectedRoute, ...}: LiveLocationMapKitProps`)
- Optional params documented in type interfaces with `?` (e.g., `displayVehicles?: boolean`)

**Return Values:**
- Python: Functions document return type in type hints (e.g., `async def compute_per_stop_etas(...) -> dict`)
- TypeScript: Return types explicit (e.g., `() => JSX.Element`, `async () => Promise<VehicleLocationMap>`)
- Empty collections preferred over `None`: Return `{}` or `[]` on empty/error (not `null`)

## Module Design

**Exports:**
- Python: Explicit imports (no `from backend import *`)
- Backend `__init__.py` files provide package-level exports (e.g., `backend/__init__.py` exports `app`, `models`, `utils`)
- TypeScript: Default exports for components, named exports for utilities/types

**Barrel Files:**
- Python: Each package has `__init__.py` with selective re-exports (e.g., `backend/worker/__init__.py` exports `run_worker`)
- TypeScript: No barrel files observed; direct imports used (no `index.ts` re-export pattern)
- Shared data: JSON files imported directly (`import routeData from './shared/routes.json'`)

**Dependency Injection:**
- FastAPI routes receive `request: Request` and access `request.app.state.session_factory`
- Cache decorator handles Redis connection via `@cache(...)` (see `backend/fastapi/routes.py` line 44)
- Database session passed explicitly: `async def get_locations(...) -> AsyncGenerator[AsyncSession, None]` pattern in `backend/database.py`

---

*Convention analysis: 2026-04-05*
