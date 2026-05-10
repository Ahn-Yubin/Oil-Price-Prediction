from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.app.api.dependencies import service_error
from market_ai.config import get_settings
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable
from market_ai.data.storage import DATA_ROOT, read_table
from market_ai.forecasting.service import ForecastUnavailable, build_forecast
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError
from market_ai.modeling.model_catalog import InvalidModelRequest


router = APIRouter()


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return read_table(path)
    except Exception:
        return pd.DataFrame()


def _iso(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.isoformat()


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def _filter_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns:
        return frame
    symbol_upper = symbol.upper()
    return frame[frame["symbol"].astype(str).str.upper().isin([symbol_upper, "ALL", "*"])].copy()


def _news_items(*, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, limit: int) -> list[dict[str, Any]]:
    frame = _read_optional(DATA_ROOT / "raw" / "news" / "public_market_news.csv")
    if frame.empty:
        return []
    frame = _filter_symbol(frame, symbol)
    frame["published_at"] = pd.to_datetime(frame.get("published_at"), errors="coerce", utc=True)
    frame = frame.dropna(subset=["published_at"])
    frame = frame[(frame["published_at"] >= start_ts) & (frame["published_at"] <= end_ts)].sort_values("published_at")
    rows = []
    for row in frame.tail(limit).to_dict(orient="records"):
        rows.append(
            {
                "time": int(pd.Timestamp(row["published_at"]).timestamp()),
                "published_at": _iso(row.get("published_at")),
                "symbol": row.get("symbol"),
                "headline": _text(row.get("headline")),
                "source": _text(row.get("source")),
                "url": _text(row.get("url")),
            }
        )
    return rows


def _context_points(*, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, limit: int) -> list[dict[str, Any]]:
    frame = _read_optional(DATA_ROOT / "processed" / "event_context" / "event_context_daily.csv")
    if frame.empty:
        return []
    frame = _filter_symbol(frame, symbol)
    frame["timestamp"] = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"])
    frame = frame[(frame["timestamp"] >= start_ts.floor("D")) & (frame["timestamp"] <= end_ts.ceil("D"))].sort_values("timestamp")
    if "event_count" in frame.columns:
        event_count = pd.to_numeric(frame["event_count"], errors="coerce").fillna(0.0)
        impact = pd.to_numeric(frame.get("impact_score", 0.0), errors="coerce").fillna(0.0)
        frame = frame[(event_count > 0) | (impact > 0)]
    rows = []
    for row in frame.tail(limit).to_dict(orient="records"):
        ts = pd.Timestamp(row["timestamp"])
        rows.append(
            {
                "time": int(ts.timestamp()),
                "timestamp": ts.isoformat(),
                "symbol": row.get("symbol"),
                "llm_mode": row.get("llm_mode"),
                "overall_bias": row.get("overall_bias"),
                "impact_score": float(pd.to_numeric(pd.Series([row.get("impact_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                "uncertainty": float(pd.to_numeric(pd.Series([row.get("uncertainty", 1.0)]), errors="coerce").fillna(1.0).iloc[0]),
                "event_count": int(pd.to_numeric(pd.Series([row.get("event_count", 0)]), errors="coerce").fillna(0).iloc[0]),
                "explanation": _text(row.get("explanation")),
                "warnings": _text(row.get("warnings")),
            }
        )
    return rows


def _scenario_commentary(bundle) -> dict[str, Any]:
    response = bundle.response
    first = response.forecast[0] if response.forecast else None
    last = response.forecast[-1] if response.forecast else None
    direction = "flat"
    if last is not None and last.p50 > response.current_price:
        direction = "upside"
    elif last is not None and last.p50 < response.current_price:
        direction = "downside"
    regime_values = response.regime.model_dump()
    regime_label = max((key for key in regime_values if key != "confidence"), key=lambda key: regime_values[key])
    context_role = response.llm_context_summary.get("role") or "context/event encoder only"
    return {
        "mode": "deterministic_context_narrative",
        "summary": (
            f"{response.primary_model or 'selected model'} median path currently leans {direction}; "
            f"dominant regime is {regime_label}. LLM context role: {context_role}."
        ),
        "bull": "Bull case follows the upper quantile path when trend, event impulse, and volatility remain favorable.",
        "base": "Base case follows the median cumulative log-return path restored back into price space.",
        "bear": "Bear case follows the lower quantile path when volatility expands or recent event/context pressure turns adverse.",
        "confidence_warning": None
        if first and first.confidence >= 0.45
        else "Scenario text is explanatory only; probability bands are not validated confidence intervals unless calibration says so.",
    }


@router.get("/api/market-context")
def market_context(
    symbol: str = Query(default=None),
    interval: str = Query(default=None),
    horizon: int | None = Query(default=None, ge=1),
    models: str | None = Query(default=None),
    limit: int = Query(default=60, ge=1, le=200),
):
    current_settings = get_settings()
    try:
        bundle = build_forecast(
            symbol=symbol or current_settings.default_symbol,
            interval=interval or current_settings.default_interval,
            horizon=horizon,
            models=models,
            settings=current_settings,
        )
    except (MarketDataUnavailable, PretrainedModelNotFoundError, ForecastUnavailable) as exc:
        raise service_error(exc) from exc
    except InvalidModelRequest as exc:
        raise HTTPException(status_code=400, detail=exc.as_detail()) from exc

    candles = bundle.market_data.candles
    if candles:
        start_ts = pd.to_datetime(candles[0].time, unit="s", utc=True)
        end_ts = max(pd.to_datetime(candles[-1].time, unit="s", utc=True), pd.Timestamp(datetime.now(timezone.utc)))
    else:
        end_ts = pd.Timestamp(datetime.now(timezone.utc))
        start_ts = end_ts - pd.Timedelta(days=120)
    resolved_symbol = bundle.response.symbol
    return {
        "symbol": resolved_symbol,
        "interval": bundle.response.interval,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_context_summary": bundle.response.llm_context_summary,
        "news": _news_items(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=limit),
        "context_points": _context_points(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=limit),
        "scenario_commentary": _scenario_commentary(bundle),
        "primary_model": bundle.response.primary_model,
        "calibration_status": bundle.response.calibration_status,
    }
