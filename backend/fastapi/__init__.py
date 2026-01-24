"""FastAPI application factory."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database import create_async_db_engine, create_session_factory
from backend.cache import init_cache, close_cache


# Configure logging for FastAPI
fastapi_log_level = settings.get_log_level("fastapi")
numeric_level = logging._nameToLevel.get(fastapi_log_level.upper(), logging.INFO)
logging.basicConfig(
    level=numeric_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
logger.info(f"FastAPI logging level: {fastapi_log_level}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting up FastAPI application...")

    # Initialize database engine and session factory
    app.state.db_engine = create_async_db_engine(
        settings.DATABASE_URL, echo=settings.DEBUG
    )
    app.state.session_factory = create_session_factory(app.state.db_engine)
    logger.info("Database engine and session factory initialized")

    # Initialize Redis cache
    app.state.redis = await init_cache(settings.REDIS_URL)
    logger.info("Redis cache initialized")

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application...")
    await close_cache()
    await app.state.db_engine.dispose()
    logger.info("Database connections closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Shubble API",
        description="Shuttle tracking API for Shubble",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.FRONTEND_URLS.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from .routes import router
    app.include_router(router)

    return app


# Create app instance
app = create_app()
