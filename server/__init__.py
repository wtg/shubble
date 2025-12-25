"""FastAPI application factory."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

from .config import settings
from .database import create_async_db_engine, create_session_factory


# Configure logging
numeric_level = logging._nameToLevel.get(settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=numeric_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


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
    app.state.redis = await aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    FastAPICache.init(RedisBackend(app.state.redis), prefix="fastapi-cache")
    logger.info("Redis cache initialized")

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application...")
    await app.state.redis.close()
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
        allow_origins=[settings.FRONTEND_URL],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from .routes import router
    app.include_router(router)

    # Mount static files for frontend (this should be last)
    try:
        app.mount("/", StaticFiles(directory="../client/dist", html=True), name="static")
    except RuntimeError:
        # Static directory doesn't exist yet (development mode)
        logger.warning("Static files directory not found. Skipping static file mounting.")

    return app


# Create app instance
app = create_app()
