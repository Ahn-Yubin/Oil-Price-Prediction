from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.api.dependencies import service_error
from market_ai.config import get_settings
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable
from market_ai.forecasting.scenarios import ScenarioForecastUnavailable, build_scenario_forecast
from market_ai.forecasting.service import ForecastUnavailable
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError
from market_ai.modeling.model_catalog import InvalidModelRequest
from market_ai.schemas.market import ScenarioForecastRequest, ScenarioForecastResponse


router = APIRouter()


@router.post("/api/scenarios/forecast", response_model=ScenarioForecastResponse)
def scenario_forecast(request: ScenarioForecastRequest) -> ScenarioForecastResponse:
    current_settings = get_settings()
    try:
        return build_scenario_forecast(
            title=request.title,
            content=request.content,
            event_time=request.event_time,
            events=request.events,
            symbol=request.symbol or current_settings.default_symbol,
            interval=request.interval or current_settings.default_interval,
            horizon=request.horizon,
            models=request.models,
            settings=current_settings,
        )
    except ScenarioForecastUnavailable as exc:
        raise HTTPException(status_code=503, detail={"message": str(exc)}) from exc
    except (MarketDataUnavailable, PretrainedModelNotFoundError, ForecastUnavailable) as exc:
        raise service_error(exc) from exc
    except InvalidModelRequest as exc:
        raise HTTPException(status_code=400, detail=exc.as_detail()) from exc
