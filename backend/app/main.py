from __future__ import annotations

from fastapi import FastAPI

from backend.app.api.routes import backtests, chart, data_status, explanation, features, forecast, health, market_context, models
from backend.app.web.static_server import register_static_frontend
from market_ai.config import get_settings
from market_ai.constants import (
    CONFIDENCE_Z,
    FALLBACK_INTERVAL,
    INTERVAL_TO_DELTA,
    INTERVAL_TO_HORIZON,
    INTERVAL_TO_MAX_LOG_BAND,
    INTERVAL_TO_PERIOD,
    INTERVAL_TO_PERIOD_CANDIDATES,
    INTERVAL_TO_RETURN_CLIP,
)


settings = get_settings()
app = FastAPI(title="Universal Market Forecast Platform", version=settings.app_version)

for router in [
    forecast.router,
    explanation.router,
    chart.router,
    data_status.router,
    features.router,
    backtests.router,
    market_context.router,
    models.router,
    health.router,
]:
    app.include_router(router)

register_static_frontend(app)


__all__ = [
    "CONFIDENCE_Z",
    "FALLBACK_INTERVAL",
    "INTERVAL_TO_DELTA",
    "INTERVAL_TO_HORIZON",
    "INTERVAL_TO_MAX_LOG_BAND",
    "INTERVAL_TO_PERIOD",
    "INTERVAL_TO_PERIOD_CANDIDATES",
    "INTERVAL_TO_RETURN_CLIP",
    "app",
]
