import base64
import os

class Config:
    # hosting settings
    DEBUG = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    ENV = os.environ.get('FLASK_ENV', 'development').lower()
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

    # database settings
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")
    if SQLALCHEMY_DATABASE_URI.startswith('postgres://'):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('postgres://', 'postgresql://', 1)
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    if secret := os.environ.get('SAMSARA_SECRET', None):
        SAMSARA_SECRET = base64.b64decode(secret.encode('utf-8'), 'utf-8')
    else:
        SAMSARA_SECRET = None