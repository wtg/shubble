from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_caching import Cache
from flask_cors import CORS
import logging
from .config import Config

numeric_level = logging._nameToLevel.get(Config.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=numeric_level,
)

db = SQLAlchemy()
migrate = Migrate()
cache = Cache()

def create_app():
    # create and configure the app
    app = Flask(__name__, static_folder='../client/dist', static_url_path='/')
    app.config.from_object(Config)

    # Enable CORS for cross-origin requests from frontend
    CORS(app,
         resources={r"/api/*": {"origins": app.config['CORS_ORIGINS']}},
         supports_credentials=True,
         allow_headers=["Content-Type"],
         methods=["GET", "POST", "DELETE", "OPTIONS", "PUT", "PATCH"])

    # initialize database
    db.init_app(app)
    # make any necessary migrations
    migrate.init_app(app, db)

    # initialize cache
    cache.init_app(app, config={'CACHE_TYPE': 'RedisCache', 'CACHE_REDIS_URL': app.config["REDIS_URL"]})

    # register routes
    from . import routes
    app.register_blueprint(routes.bp)

    return app