"""FastAPI test server - Mock Samsara API for development/testing."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.config import settings
from backend.database import create_async_db_engine, create_session_factory
from .shuttles import (
    router as shuttles_router,
    shuttles,
    stop_all_shuttles,
    setup_shuttles,
)
from .events import router as events_router
from .mock_samsara import router as mock_samsara_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting test server...")

    # Initialize database
    app.state.db_engine = create_async_db_engine(
        settings.DATABASE_URL, echo=settings.DEBUG
    )
    app.state.session_factory = create_session_factory(app.state.db_engine)
    logger.info("Database initialized")

    # Setup shuttles from database
    await setup_shuttles(app.state.session_factory)
    logger.info(f"Initialized {len(shuttles)} shuttles from database")

    yield

    # Shutdown
    logger.info("Shutting down test server...")
    stop_all_shuttles()
    await app.state.db_engine.dispose()
    logger.info("Test server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Mock Samsara API",
    description="Test server for Shubble development",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for test-client
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.TEST_FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(shuttles_router)
app.include_router(events_router)
app.include_router(mock_samsara_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=4000)
