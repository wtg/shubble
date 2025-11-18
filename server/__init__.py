from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_caching import Cache
import logging
from .config import Config
from .services.eta_predictor import ETAPredictor
from pathlib import Path


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

    # initialize database
    db.init_app(app)
    # make any necessary migrations
    migrate.init_app(app, db)
    
    # initialize cache
    cache.init_app(app, config={'CACHE_TYPE': 'RedisCache', 'CACHE_REDIS_URL': app.config["REDIS_URL"]})

    # register routes
    from . import routes
    app.register_blueprint(routes.bp)

    model_path = Path(__file__).parent.parent / 'models'
    ETAPredictor().initialize(model_path)
    
    return app
