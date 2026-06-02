from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.api.dependencies import service_error
from market_ai.config import get_settings
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable
from market_ai.forecasting.service import ForecastUnavailable, build_forecast, chart_payload_from_forecast
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError
from market_ai.modeling.model_catalog import InvalidModelRequest


router = APIRouter()


@router.get("/api/chart")
def chart_data(
    symbol: str = Query(default=None),
    interval: str = Query(default=None),
    horizon: int | None = Query(default=None, ge=1),
    models: str | None = Query(default=None),
):
    current_settings = get_settings()
    try:
        bundle = build_forecast(
            symbol=symbol or current_settings.default_symbol,
            interval=interval or current_settings.default_interval,
            horizon=horizon,
            models=models,
            allow_removed_models_as_warning=True,
            settings=current_settings,
        )
        return chart_payload_from_forecast(bundle)
    except (MarketDataUnavailable, PretrainedModelNotFoundError, ForecastUnavailable) as exc:
        raise service_error(exc) from exc
    except InvalidModelRequest as exc:
        raise HTTPException(status_code=400, detail=exc.as_detail()) from exc
