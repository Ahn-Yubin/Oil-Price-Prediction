from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.api.dependencies import service_error
from market_ai.config import get_settings
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable
from market_ai.forecasting.service import ForecastUnavailable, build_forecast
from market_ai.llm.event_encoder import deterministic_explanation, encoder_from_settings
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError
from market_ai.schemas.llm_context import ExplanationOutput, MarketContextInput


router = APIRouter()


@router.get("/api/explanation", response_model=ExplanationOutput)
def explanation(symbol: str = Query(default=None), interval: str = Query(default=None)):
    current_settings = get_settings()
    try:
        bundle = build_forecast(
            symbol=symbol or current_settings.default_symbol,
            interval=interval or current_settings.default_interval,
            settings=current_settings,
        )
    except (MarketDataUnavailable, PretrainedModelNotFoundError, ForecastUnavailable) as exc:
        raise service_error(exc) from exc

    response = bundle.response
    first = response.forecast[0] if response.forecast else None
    regime_values = {
        "trend_up": response.regime.trend_up,
        "trend_down": response.regime.trend_down,
        "range": response.regime.range,
        "high_volatility": response.regime.high_volatility,
        "event_driven": response.regime.event_driven,
    }
    forecast_summary = {
        "current_price": response.current_price,
        "p50_first": first.p50 if first else None,
        "confidence": first.confidence if first else None,
        "regime": max(regime_values, key=regime_values.get),
    }
    context_input = MarketContextInput(
        symbol=response.symbol,
        interval=response.interval,
        asset_class=str(response.asset_metadata.asset_class),
        generated_at=response.generated_at,
        data_status=response.data_status.model_dump(),
        forecast_summary=forecast_summary,
    )
    llm_context = encoder_from_settings(current_settings).encode_events(context_input)
    return deterministic_explanation(
        symbol=response.symbol,
        interval=response.interval,
        forecast_summary=forecast_summary,
        data_status=response.data_status.model_dump(),
        llm_context=llm_context,
    )
