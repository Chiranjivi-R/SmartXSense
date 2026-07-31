"""Flask application factory for SmartXSense."""

from flask import Flask

from .config import Config
from .extensions import cors, db, jwt, migrate


def create_app(config_class=Config):
    """Create and configure a SmartXSense Flask application instance."""
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    cors.init_app(app, resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}})

    from .api.auth import auth_bp
    from .api.dashboard import dashboard_bp
    from .api.datasets import datasets_bp
    from .api.explainability import explainability_bp
    from .api.forecasting import forecasting_bp
    from .api.health import health_bp
    from .api.reports import reports_bp
    from .api.settings import settings_bp

    for blueprint in (auth_bp, datasets_bp, forecasting_bp, explainability_bp, dashboard_bp, reports_bp, settings_bp, health_bp):
        app.register_blueprint(blueprint)

    return app
