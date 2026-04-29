from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from market_ai.data.providers.yfinance_provider import MarketDataUnavailable
from market_ai.forecasting.service import ForecastUnavailable
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError


def service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, MarketDataUnavailable):
        detail: Any = str(exc)
        if exc.data_status is not None:
            detail = {"message": str(exc), "data_status": exc.data_status.model_dump()}
        return HTTPException(status_code=503, detail=detail)
    if isinstance(exc, PretrainedModelNotFoundError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, ForecastUnavailable):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))
