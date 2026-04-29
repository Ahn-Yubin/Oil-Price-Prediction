from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.api.dependencies import service_error
from market_ai.config import get_settings
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable, load_market_data_window
from market_ai.schemas.market import DataStatus


router = APIRouter()


@router.get("/api/data-status", response_model=DataStatus)
def data_status(symbol: str = Query(default=None), interval: str = Query(default=None)):
    current_settings = get_settings()
    try:
        window = load_market_data_window(
            symbol or current_settings.default_symbol,
            interval or current_settings.default_interval,
            settings=current_settings,
        )
        return window.data_status
    except MarketDataUnavailable as exc:
        raise service_error(exc) from exc
