import base64
from dotenv import load_dotenv
import os
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

load_dotenv()


class Config:
    # hosting settings
    DEBUG = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    ENV = os.environ.get('FLASK_ENV', 'development').lower()
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

    # service URLs
    FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:5173')
    BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:5001')
    TEST_FRONTEND_URL = os.environ.get('TEST_FRONTEND_URL', 'http://localhost:5174')
    TEST_BACKEND_URL = os.environ.get('TEST_BACKEND_URL', 'http://localhost:4000')

    # CORS allowed origins - allow requests from frontend and test frontend
    CORS_ORIGINS = [FRONTEND_URL, TEST_FRONTEND_URL]

    # Extract port from BACKEND_URL for Flask to bind to
    _parsed_backend = urlparse(BACKEND_URL)
    PORT = _parsed_backend.port or 5001

    # database settings
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")
    if SQLALCHEMY_DATABASE_URI.startswith('postgres://'):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('postgres://', 'postgresql://', 1)
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    REDIS_URL = os.environ.get('REDIS_URL')
    if secret := os.environ.get('SAMSARA_SECRET', None):
        SAMSARA_SECRET = base64.b64decode(secret.encode('utf-8'))
    else:
        SAMSARA_SECRET = None


    # shubble settings
    CAMPUS_TZ = ZoneInfo('America/New_York')