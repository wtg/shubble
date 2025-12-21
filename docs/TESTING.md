# Testing Guide

This document describes how to run tests for the Shubble project.

## Overview

Shubble uses different testing frameworks for frontend and backend:
- **Frontend**: Vitest (modern, fast Vite-native test runner)
- **Backend**: pytest (Python's standard testing framework)

## Frontend Tests (Vitest)

### Running Tests

```bash
# Run all tests once
npm test

# Run tests in watch mode (auto-rerun on file changes)
npm test -- --watch

# Run tests with interactive UI
npm run test:ui

# Run tests with coverage report
npm run test:coverage
```

### Writing Tests

Frontend tests are located in `client/src/test/` and follow the pattern `*.test.tsx` or `*.test.ts`.

**Example test:**
```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import MyComponent from '../MyComponent';

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });
});
```

### Test Setup

- Test configuration: `vite.config.ts`
- Test setup file: `client/src/test/setup.ts`
- Global test utilities from `@testing-library/react`

## Backend Tests (pytest)

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=server --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::test_vehicle_creation

# Run tests by marker
pytest -m unit              # Only unit tests
pytest -m integration       # Only integration tests
pytest -m "not slow"        # Exclude slow tests
```

### Writing Tests

Backend tests are located in `tests/` and follow the pattern `test_*.py`.

**Example test:**
```python
import pytest
from server.models import Vehicle

@pytest.mark.unit
def test_vehicle_creation():
    """Test Vehicle model instantiation"""
    vehicle = Vehicle(
        id="test_1",
        name="Test Shuttle",
        license_plate="ABC123"
    )
    assert vehicle.id == "test_1"
    assert vehicle.name == "Test Shuttle"
```

### Test Markers

Use markers to categorize tests:
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests requiring DB/Redis
- `@pytest.mark.slow` - Slow running tests

### Test Fixtures

Common fixtures are defined in `tests/conftest.py`:
- `app` - Flask application instance
- `client` - Test client for making requests
- `db_session` - Database session with automatic rollback

## Running Tests in Docker

### Frontend
```bash
docker-compose exec frontend npm test
docker-compose exec frontend npm run test:coverage
```

### Backend
```bash
docker-compose exec backend pytest
docker-compose exec backend pytest --cov=server
```

## Continuous Integration

Tests should be run before every commit:

```bash
# Quick check
npm test && pytest

# Full check with coverage
npm run test:coverage && pytest --cov=server --cov-report=html
```

## Coverage Reports

### Frontend Coverage
After running `npm run test:coverage`, open `client/coverage/index.html` in your browser.

### Backend Coverage
After running `pytest --cov=server --cov-report=html`, open `htmlcov/index.html` in your browser.

## Test Data

### Using Mock Samsara API (Development Tool)

For local development without real GPS credentials, use the mock server:

```bash
# Terminal 1: Start mock Samsara API server
cd test-server
python server.py

# Terminal 2: Start test client UI (optional)
cd test-client
npm install
npm run dev
# Access UI at http://localhost:5173
```

In your `.env`, set:
```bash
FLASK_ENV=development
# Leave API_KEY empty or remove it
```

The mock server (`test-server/`) will:
- Simulate vehicle movement along routes
- Provide fake geofence events
- Return mock GPS data
- Allow manual control of shuttle states

The test client UI (`test-client/`) provides:
- Visual shuttle management interface
- Ability to trigger state changes (entering, looping, exiting)
- Automated test scenario execution from JSON files
- Event monitoring and debugging

**Note**: The `test-server/` and `test-client/` directories are development tools, not automated test suites. For automated tests, use the `tests/` directory (pytest) and `client/src/test/` (vitest).

## Best Practices

### Frontend
1. Test user interactions, not implementation details
2. Use `screen.getByRole()` for accessibility-friendly queries
3. Mock external dependencies (API calls, MapKit)
4. Keep tests fast and independent

### Backend
1. Use fixtures for common test setup
2. Mark tests appropriately (`@pytest.mark.unit`, etc.)
3. Test edge cases and error conditions
4. Use in-memory SQLite for fast unit tests
5. Roll back database changes after each test

## Troubleshooting

### Frontend Tests Fail

**Issue**: Module not found errors
```bash
npm install
```

**Issue**: Tests timeout
- Increase timeout in `vite.config.ts`
- Check for infinite loops or slow operations

### Backend Tests Fail

**Issue**: Database connection errors
```bash
# Check DATABASE_URL in test config
# Use SQLite in-memory for tests: sqlite:///:memory:
```

**Issue**: Redis connection errors
```bash
# Ensure Redis is running
redis-cli ping

# Or use fakeredis for tests
pip install fakeredis
```

**Issue**: Import errors
```bash
# Ensure you're in virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

## Adding New Tests

### For Frontend Features
1. Create `ComponentName.test.tsx` next to component
2. Import testing utilities
3. Write descriptive test cases
4. Run `npm test` to verify

### For Backend Features
1. Create `test_feature.py` in `tests/`
2. Add appropriate markers
3. Use fixtures from `conftest.py`
4. Run `pytest` to verify

## Related Documentation

- [INSTALLATION.md](INSTALLATION.md) - Development setup
- [ARCHITECTURE.md](ARCHITECTURE.md) - Code architecture
- [README.md](README.md) - Project overview
