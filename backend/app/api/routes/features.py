from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Query

from backend.app.api.dependencies import service_error
from market_ai.config import get_settings
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable, load_market_data_window
from market_ai.features.calendar_features import build_calendar_features
from market_ai.features.price_features import FEATURE_SET_VERSION, build_price_features


router = APIRouter()


@router.get("/api/features")
def features(
    symbol: str = Query(default=None),
    interval: str = Query(default=None),
    rows: int = Query(default=5, ge=1, le=100),
    debug: bool = Query(default=False),
) -> dict[str, Any]:
    current_settings = get_settings()
    try:
        window = load_market_data_window(
            symbol or current_settings.default_symbol,
            interval or current_settings.default_interval,
            settings=current_settings,
        )
    except MarketDataUnavailable as exc:
        raise service_error(exc) from exc

    candle_rows = []
    for candle in window.candles:
        row = candle.model_dump()
        row["date"] = pd.to_datetime(candle.time, unit="s", utc=True)
        candle_rows.append(row)
    candle_frame = pd.DataFrame(candle_rows)
    price_features = build_price_features(candle_frame)
    calendar_features = build_calendar_features(candle_frame["date"])
    combined = pd.concat([price_features, calendar_features.drop(columns=["date"], errors="ignore")], axis=1)
    tail = combined.tail(rows)
    summary_cols = [col for col in tail.columns if col != "date"]
    summary = {
        col: {
            "latest": float(tail[col].iloc[-1]) if pd.api.types.is_numeric_dtype(tail[col]) else str(tail[col].iloc[-1]),
            "mean": float(tail[col].mean()) if pd.api.types.is_numeric_dtype(tail[col]) else None,
        }
        for col in summary_cols
    }
    payload: dict[str, Any] = {
        "symbol": window.symbol.provider_symbol,
        "interval": window.timeframe.normalized,
        "feature_set_version": FEATURE_SET_VERSION,
        "data_status": window.data_status.model_dump(),
        "rows": len(combined),
        "summary": summary,
    }
    if debug:
        payload["tail"] = tail.to_dict(orient="records")
    return payload
