"""
Pytest configuration and fixtures
"""
import pytest
import os
from backend import create_app, db, cache


@pytest.fixture(scope='session')
def app():
    """Create application for testing"""
    # Set test configuration
    os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
    os.environ['REDIS_URL'] = 'redis://localhost:6379/1'  # Use DB 1 for tests
    os.environ['FLASK_ENV'] = 'testing'

    app = create_app()
    app.config['TESTING'] = True
    app.config['CACHE_TYPE'] = 'SimpleCache'  # Use simple cache for testing

    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture(scope='function')
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture(scope='function', autouse=True)
def db_session(app):
    """Create a new database session for a test"""
    with app.app_context():
        # Clear cache before each test
        cache.clear()

        yield db.session

        # Clear all data after each test
        db.session.rollback()

        # Delete all data from tables
        for table in reversed(db.metadata.sorted_tables):
            db.session.execute(table.delete())
        db.session.commit()

        # Clear cache after each test
        cache.clear()
