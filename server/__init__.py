from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import logging
from .config import Config

logging.basicConfig(
    level=logging.INFO,
)

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    # create and configure the app
    app = Flask(__name__, static_folder='../client/dist', static_url_path='/')
    app.config.from_object(Config)

    # initialize database
    db.init_app(app)
    # make any necessary migrations
    migrate.init_app(app, db)

    # register routes
    from . import routes
    app.register_blueprint(routes.bp)

    return app
