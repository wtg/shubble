"""Async database configuration for FastAPI."""
from fastapi import Request
from typing import AsyncGenerator, TYPE_CHECKING
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine,
)
from sqlalchemy.orm import declarative_base

if TYPE_CHECKING:
    from fastapi import Request

Base = declarative_base()


def create_async_db_engine(database_url: str, echo: bool = False) -> AsyncEngine:
    """
    Create an async database engine.

    Args:
        database_url: Database connection URL
        echo: Whether to log SQL statements

    Returns:
        AsyncEngine instance
    """
    # Convert postgresql:// to postgresql+asyncpg://
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
    elif database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+asyncpg://")

    return create_async_engine(database_url, echo=echo, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine):
    """
    Create an async session factory.

    Args:
        engine: AsyncEngine instance

    Returns:
        async_sessionmaker instance
    """
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting async database sessions.

    Accesses the session_factory from app.state which is initialized
    during application startup in the lifespan context manager.

    Args:
        request: FastAPI Request object (injected automatically)

    Yields:
        AsyncSession instance
    """
    async with request.app.state.session_factory() as session:
        yield session
