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

Frontend tests are located in `frontend/src/test/` and follow the pattern `*.test.tsx` or `*.test.ts`.

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

### Test Structure

- **Test configuration**: `vite.config.ts`
- **Test setup file**: `frontend/src/test/setup.ts`
- **Component tests**: `frontend/src/test/components/`
- **Integration tests**: `frontend/src/test/integration/`
- **API tests**: `frontend/src/test/api/`
- **Utility tests**: `frontend/src/test/utils/`

### Test Setup

Global test utilities from `@testing-library/react` are available, along with mocks for browser APIs (like `scrollIntoView`, `DOMPoint`) in the setup file.

## Backend Tests (pytest)

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest testing/tests/test_models.py

# Run specific test
pytest testing/tests/test_models.py::test_vehicle_creation

# Run tests by marker
pytest -m unit              # Only unit tests
pytest -m integration       # Only integration tests
pytest -m "not slow"        # Exclude slow tests
```

### Writing Tests

Backend tests are located in `testing/tests/` and follow the pattern `test_*.py`.

**Example test:**
```python
import pytest
from backend.models import Vehicle

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

### Test Structure

- **Test configuration**: `pytest.ini` (testpaths set to `testing/tests`)
- **Test fixtures**: `testing/tests/conftest.py`
- **Unit tests**: `testing/tests/test_*.py`
- **Integration tests**: `testing/tests/integration/`

### Test Markers

Use markers to categorize tests:
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests requiring DB/Redis
- `@pytest.mark.slow` - Slow running tests

### Test Fixtures

Common fixtures are defined in `testing/tests/conftest.py`:
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
docker-compose exec backend pytest --cov=backend
```

## Continuous Integration

Tests should be run before every commit:

```bash
# Quick check
npm test && pytest

# Full check with coverage
npm run test:coverage && pytest --cov=backend --cov-report=html
```

## Coverage Reports

### Frontend Coverage
After running `npm run test:coverage`, open `frontend/coverage/index.html` in your browser.

### Backend Coverage
After running `pytest --cov=backend --cov-report=html`, open `htmlcov/index.html` in your browser.

## Development Tools (Not Automated Tests)

### Using Mock Samsara API

For local development without real GPS credentials, use the mock server:

```bash
# Terminal 1: Start mock Samsara API server
cd testing/test-server
python server.py
# Mock API runs on http://localhost:4000
# Test client UI served at http://localhost:4000
```

In your `.env`, set:
```bash
FLASK_ENV=development
# Leave API_KEY empty or remove it to use mock server
```

The mock server (`testing/test-server/`) will:
- Simulate vehicle movement along routes
- Provide fake geofence events
- Return mock GPS data
- Allow manual control of shuttle states
- Serve the test client UI

The test client UI (served by test-server) provides:
- Visual shuttle management interface
- Ability to trigger state changes (entering, looping, exiting)
- Automated test scenario execution from JSON files
- Event monitoring and debugging

**Important Distinction**:
- `testing/test-server/` and `testing/test-client/` = Development tools for manual testing
- `testing/tests/` = Automated test suite (pytest)
- `frontend/src/test/` = Automated frontend tests (vitest)

## Best Practices

### Frontend
1. Test user interactions, not implementation details
2. Use `screen.getByRole()` for accessibility-friendly queries
3. Mock external dependencies (API calls, MapKit)
4. Keep tests fast and independent
5. Use descriptive test names

### Backend
1. Use fixtures for common test setup
2. Mark tests appropriately (`@pytest.mark.unit`, etc.)
3. Test edge cases and error conditions
4. Use in-memory SQLite for fast unit tests
5. Roll back database changes after each test
6. Clear cache before and after each test

## Troubleshooting

### Frontend Tests Fail

**Issue**: Module not found errors
```bash
npm install
```

**Issue**: Tests timeout
- Increase timeout in `vite.config.ts`
- Check for infinite loops or slow operations
- Ensure mocks are properly set up

**Issue**: MapKit errors in tests
- Check that `frontend/src/test/setup.ts` has proper MapKit mocks
- Verify `global.mapkit` is properly mocked in test files

### Backend Tests Fail

**Issue**: Database connection errors
```bash
# Check DATABASE_URL in test config
# Tests use SQLite in-memory: sqlite:///:memory:
# Check testing/tests/conftest.py
```

**Issue**: Redis connection errors
```bash
# Tests use SimpleCache, not Redis
# Check testing/tests/conftest.py for cache configuration
```

**Issue**: Import errors (`ModuleNotFoundError: No module named 'backend'`)
```bash
# Ensure you're in virtual environment
source venv/bin/activate
pip install -r requirements.txt

# Check pytest.ini testpaths setting
```

**Issue**: Foreign key constraint errors
```bash
# Ensure proper test data setup
# Create parent records (Vehicle) before child records (GeofenceEvent)
```

## Adding New Tests

### For Frontend Features
1. Create test file in appropriate `frontend/src/test/` subdirectory
2. Import testing utilities from `vitest` and `@testing-library/react`
3. Write descriptive test cases
4. Run `npm test` to verify
5. Check coverage with `npm run test:coverage`

Example:
```typescript
// frontend/src/test/components/MyComponent.test.tsx
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import MyComponent from '../../components/MyComponent';

describe('MyComponent', () => {
  it('should render without crashing', () => {
    const { container } = render(<MyComponent />);
    expect(container).toBeTruthy();
  });
});
```

### For Backend Features
1. Create `test_feature.py` in `testing/tests/`
2. Add appropriate markers (`@pytest.mark.unit` or `@pytest.mark.integration`)
3. Use fixtures from `conftest.py`
4. Run `pytest` to verify
5. Check coverage with `pytest --cov=backend`

Example:
```python
# testing/tests/test_feature.py
import pytest
from backend.models import MyModel

@pytest.mark.unit
def test_my_feature():
    """Test description"""
    result = MyModel.do_something()
    assert result == expected_value
```

## Test Organization

```
testing/
├── tests/                  # Automated pytest tests
│   ├── conftest.py        # Test fixtures and configuration
│   ├── test_api_endpoints.py
│   ├── test_models.py
│   ├── test_worker.py
│   └── integration/       # Integration tests
│       └── test_shuttle_workflow.py
│
├── test-server/           # Mock Samsara API (dev tool)
│   ├── server.py
│   └── shuttle.py
│
└── test-client/           # UI for controlling mock shuttles (dev tool)

frontend/src/test/         # Automated vitest tests
├── setup.ts              # Test setup and global mocks
├── components/           # Component tests
├── integration/          # Integration tests
├── api/                  # API tests
└── utils/                # Utility tests
```

## Related Documentation

- [INSTALLATION.md](INSTALLATION.md) - Development setup
- [ARCHITECTURE.md](ARCHITECTURE.md) - Code architecture
- [README.md](../README.md) - Project overview
