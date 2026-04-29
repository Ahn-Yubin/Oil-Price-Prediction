from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.api.dependencies import service_error
from market_ai.config import get_settings
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable
from market_ai.forecasting.service import ForecastUnavailable, build_forecast, chart_payload_from_forecast
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError


router = APIRouter()


@router.get("/api/chart")
def chart_data(symbol: str = Query(default=None), interval: str = Query(default=None)):
    current_settings = get_settings()
    try:
        bundle = build_forecast(
            symbol=symbol or current_settings.default_symbol,
            interval=interval or current_settings.default_interval,
            settings=current_settings,
        )
        return chart_payload_from_forecast(bundle)
    except (MarketDataUnavailable, PretrainedModelNotFoundError, ForecastUnavailable) as exc:
        raise service_error(exc) from exc
