import os

class Config:
    # hosting settings
    DEBUG = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    ENV = os.environ.get('FLASK_ENV', 'development').lower() == 'production'
    PORT = int(os.environ.get('PORT', 3000))
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')

    # database settings
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///dev.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
