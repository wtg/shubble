"""
Pytest configuration and fixtures
"""
import pytest
import os
from server import create_app, db


@pytest.fixture(scope='session')
def app():
    """Create application for testing"""
    # Set test configuration
    os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
    os.environ['REDIS_URL'] = 'redis://localhost:6379/1'  # Use DB 1 for tests
    os.environ['FLASK_ENV'] = 'testing'

    app = create_app()
    app.config['TESTING'] = True

    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture(scope='function')
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture(scope='function')
def db_session(app):
    """Create a new database session for a test"""
    with app.app_context():
        db.session.begin_nested()
        yield db.session
        db.session.rollback()
