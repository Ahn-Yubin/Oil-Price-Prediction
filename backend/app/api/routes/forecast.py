from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.api.dependencies import service_error
from market_ai.config import get_settings
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable
from market_ai.forecasting.service import ForecastUnavailable, build_forecast
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError
from market_ai.schemas.market import ForecastResponse


router = APIRouter()


@router.get("/api/forecast", response_model=ForecastResponse)
def forecast_data(
    symbol: str = Query(default=None),
    interval: str = Query(default=None),
    horizon: int | None = Query(default=None, ge=1),
    models: str | None = Query(default=None),
    include_explanation: bool = Query(default=False),
    include_scenarios: bool = Query(default=True),
):
    current_settings = get_settings()
    try:
        bundle = build_forecast(
            symbol=symbol or current_settings.default_symbol,
            interval=interval or current_settings.default_interval,
            horizon=horizon,
            models=models,
            include_explanation=include_explanation,
            include_scenarios=include_scenarios,
            settings=current_settings,
        )
        return bundle.response
    except (MarketDataUnavailable, PretrainedModelNotFoundError, ForecastUnavailable) as exc:
        raise service_error(exc) from exc
