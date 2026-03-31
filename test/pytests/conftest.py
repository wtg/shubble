import asyncio
import pytest
import pytest_asyncio
from fastapi import FastAPI, Depends
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# Database set up in memory sqlite db for testing
DATABASE_URL = "sqlite+aiosqlite:///:memory:"
Base = declarative_base()

# pytest-asyncio needs event loop for async tests
# scope = "session" means one loop per test
@pytest_asyncio.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

# Test building connect to the sqlite in memory db and creating tables
# yield the engine for use in tests, and dispose after tests are done
@pytest_asyncio.fixture(scope = "session")
async def engine():
    engine = create_async_engine(DATABASE_URL, echo = False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()

# Creates an async db session for each test
# ensure db is rolled back after each test to ensure tests don't mess with each other
@pytest_asyncio.fixture
async def db_session(engine):
    async_session = sessionmaker(
        engine, class_= AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
        await session.rollback()


# Create FastAPI app
# add simple route for testing
# returns the app to be used by tests
@pytest_asyncio.fixture
async def app(db_session):
    app = FastAPI()

    # Dependency override
    async def get_db():
        yield db_session

    @app.get("/test")
    async def test_route(db: AsyncSession = Depends(get_db)):
        return {"message": "ok"}

    return app


# HTTP Client fixture to test FastAPI routes
@pytest_asyncio.fixture
async def client(app: FastAPI):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac