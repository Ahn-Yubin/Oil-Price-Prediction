from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import json

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from market_ai.config import PROJECT_DIR, get_settings
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable, load_market_data_window
from market_ai.forecasting.service import ForecastUnavailable, build_forecast, chart_payload_from_forecast
from market_ai.schemas.market import Candle, MarketDataWindow


router = APIRouter()


def _parse_origin_time(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="origin_time is required.")
    if text.isdigit():
        return int(text)
    try:
        timestamp = pd.Timestamp(text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="origin_time must be a unix timestamp or ISO datetime.") from exc
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.timestamp())


def _origin_index(candles: list[Candle], origin_time: int) -> int:
    origin_idx = -1
    for idx, candle in enumerate(candles):
        if int(candle.time) <= origin_time:
            origin_idx = idx
        else:
            break
    if origin_idx < 0:
        raise HTTPException(status_code=400, detail="origin_time is before the first available candle.")
    if origin_idx >= len(candles) - 1:
        raise HTTPException(status_code=400, detail="origin_time must be before the latest available candle.")
    return origin_idx


def _point_in_time_window(market: MarketDataWindow, candles: list[Candle]) -> MarketDataWindow:
    last_bar_time = datetime.fromtimestamp(candles[-1].time, tz=timezone.utc).isoformat()
    data_status = market.data_status.model_copy(update={"last_bar_time": last_bar_time})
    return market.model_copy(update={"candles": candles, "data_status": data_status})


def _actual_future_candles(candles: list[Candle], origin_idx: int, horizon: int) -> list[dict[str, Any]]:
    end_idx = min(len(candles), origin_idx + horizon + 1)
    return [candle.model_dump() for candle in candles[origin_idx + 1 : end_idx]]


def _backtest_metric_summary(payload: dict[str, Any], actual_future: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = payload.get("predicted") or []
    pred_values = [float(point["value"]) for point in predicted[1 : len(actual_future) + 1] if point.get("value") is not None]
    actual_values = [float(candle["close"]) for candle in actual_future[: len(pred_values)] if candle.get("close") is not None]
    if not pred_values or len(pred_values) != len(actual_values):
        return {"mae": None, "rmse": None, "mape": None, "metric_mode": "backtest", "metric_label": "Backtest metrics unavailable"}
    errors = [pred - actual for pred, actual in zip(pred_values, actual_values)]
    abs_errors = [abs(value) for value in errors]
    mae = sum(abs_errors) / len(abs_errors)
    rmse = (sum(value * value for value in errors) / len(errors)) ** 0.5
    mape = sum(abs(pred - actual) / max(abs(actual), 1e-8) for pred, actual in zip(pred_values, actual_values)) / len(actual_values) * 100.0
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "metric_mode": "backtest",
        "metric_label": "Backtest origin metrics",
        "metric_samples": len(actual_values),
    }


@router.get("/api/backtests")
def backtests(symbol: str = Query(default=None), interval: str = Query(default=None)) -> dict[str, Any]:
    current_settings = get_settings()
    del symbol
    requested_symbol = current_settings.default_symbol.replace("=", "_")
    requested_interval = interval or current_settings.default_interval
    output_dir = PROJECT_DIR / "outputs" / "backtests"
    latest_leaderboard = output_dir / "leaderboards" / "latest.json"
    if latest_leaderboard.exists():
        meta = json.loads(latest_leaderboard.read_text(encoding="utf-8"))
        latest_dir = PROJECT_DIR / meta.get("output_dir", "")
        leaderboard_path = latest_dir / "leaderboard.csv"
        availability_path = latest_dir / "model_availability.csv"
        if leaderboard_path.exists():
            frame = pd.read_csv(leaderboard_path)
            if "symbol" in frame.columns:
                frame = frame[frame["symbol"].astype(str).str.replace("=", "_") == requested_symbol]
            if "interval" in frame.columns:
                frame = frame[frame["interval"].astype(str) == requested_interval]
            availability = pd.read_csv(availability_path).to_dict(orient="records") if availability_path.exists() else []
            return {
                "status": "available",
                "path": str(leaderboard_path.relative_to(PROJECT_DIR)),
                "rows": len(frame),
                "leaderboard": frame.head(25).to_dict(orient="records"),
                "model_availability": availability,
                "latest_run": meta,
            }
    candidates = [
        output_dir / f"{requested_symbol}_{requested_interval}_leaderboard.csv",
        output_dir / f"{requested_symbol}_leaderboard.csv",
        output_dir / f"{requested_symbol}_{requested_interval}_summary.csv",
        output_dir / f"{requested_symbol}_summary.csv",
    ]
    availability_candidates = [
        output_dir / f"{requested_symbol}_{requested_interval}_model_availability.csv",
        output_dir / "latest_model_availability.csv",
    ]

    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            availability_path = next((candidate for candidate in availability_candidates if candidate.exists()), None)
            availability = pd.read_csv(availability_path).to_dict(orient="records") if availability_path else []
            return {
                "status": "available",
                "path": str(path.relative_to(PROJECT_DIR)),
                "rows": len(frame),
                "leaderboard": frame.head(25).to_dict(orient="records"),
                "model_availability": availability,
            }
    availability_path = next((candidate for candidate in availability_candidates if candidate.exists()), None)
    availability = pd.read_csv(availability_path).to_dict(orient="records") if availability_path else []
    return {"status": "missing", "path": None, "rows": 0, "leaderboard": [], "model_availability": availability}


@router.get("/api/backtests/visualization")
def backtest_visualization(
    symbol: str = Query(default=None),
    interval: str = Query(default=None),
    origin_time: str = Query(...),
    horizon: int | None = Query(default=None),
    models: str | None = Query(default=None),
) -> dict[str, Any]:
    current_settings = get_settings()
    del symbol
    requested_symbol = current_settings.default_symbol
    requested_interval = interval or current_settings.default_interval
    origin_ts = _parse_origin_time(origin_time)
    try:
        full_market = load_market_data_window(requested_symbol, requested_interval, settings=current_settings)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    candles = sorted(full_market.candles, key=lambda candle: candle.time)
    if len(candles) < 2:
        raise HTTPException(status_code=400, detail="Backtest visualization requires at least two candles.")

    origin_idx = _origin_index(candles, origin_ts)
    point_in_time_market = _point_in_time_window(
        full_market.model_copy(update={"candles": candles}),
        candles[: origin_idx + 1],
    )
    try:
        bundle = build_forecast(
            symbol=requested_symbol,
            interval=requested_interval,
            horizon=horizon,
            models=models,
            include_scenarios=True,
            allow_removed_models_as_warning=True,
            settings=current_settings,
            market_override=point_in_time_market,
        )
    except ForecastUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    origin_candle = candles[origin_idx]
    payload = chart_payload_from_forecast(bundle)
    actual_future = _actual_future_candles(candles, origin_idx, bundle.horizon)
    payload["metrics"] = {
        **(payload.get("metrics") or {}),
        **_backtest_metric_summary(payload, actual_future),
    }
    payload.update(
        {
            "mode": "backtest_visualization",
            "origin_time": origin_candle.time,
            "requested_origin_time": origin_ts,
            "origin_index": origin_idx,
            "actual_future_candles": actual_future,
            "backtest": {
                "origin_time": origin_candle.time,
                "origin_index": origin_idx,
                "history_rows": origin_idx + 1,
                "actual_future_rows": len(actual_future),
                "horizon": bundle.horizon,
                "symbol": payload.get("symbol_resolved"),
                "interval": payload.get("interval_resolved"),
            },
        }
    )
    return payload
