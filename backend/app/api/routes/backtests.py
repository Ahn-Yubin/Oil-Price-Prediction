from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any
import json

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from market_ai.config import PROJECT_DIR, get_settings
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable, load_market_data_window
from market_ai.forecasting.service import ForecastUnavailable, build_forecast, chart_payload_from_forecast
from market_ai.schemas.market import Candle, MarketDataWindow


router = APIRouter()

ONLINE_RESIDUAL_WINDOW = 8
ONLINE_RESIDUAL_GAIN = 2.0
ONLINE_RESIDUAL_MAX_ABS_LOG = 0.18


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


def _log_residual_vector(payload: dict[str, Any], actual_future: list[dict[str, Any]], horizon: int) -> list[float] | None:
    predicted = payload.get("predicted") or []
    pred_values = [point.get("value") for point in predicted[1 : horizon + 1]]
    actual_values = [candle.get("close") for candle in actual_future[:horizon]]
    if len(pred_values) != horizon or len(actual_values) != horizon:
        return None
    residuals: list[float] = []
    for pred, actual in zip(pred_values, actual_values):
        try:
            pred_value = float(pred)
            actual_value = float(actual)
        except (TypeError, ValueError):
            return None
        if pred_value <= 0.0 or actual_value <= 0.0:
            return None
        residuals.append(math.log(actual_value / pred_value))
    return residuals


def _apply_log_correction_to_series(series: list[dict[str, Any]], correction: list[float]) -> None:
    for idx, delta in enumerate(correction, start=1):
        if idx >= len(series):
            break
        value = series[idx].get("value")
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if value_float > 0.0:
            series[idx]["value"] = value_float * math.exp(delta)


def _apply_online_residual_calibration(
    *,
    payload: dict[str, Any],
    full_market: MarketDataWindow,
    candles: list[Candle],
    origin_idx: int,
    requested_symbol: str,
    requested_interval: str,
    horizon: int,
    models: str | None,
    settings: Any,
) -> dict[str, Any]:
    model_trained_at = payload.get("model_trained_at")
    if model_trained_at:
        cutoff = pd.to_datetime(model_trained_at, errors="coerce", utc=True)
        origin_time = datetime.fromtimestamp(candles[origin_idx].time, tz=timezone.utc)
        if pd.notna(cutoff) and origin_time <= cutoff.to_pydatetime():
            return {"applied": False, "reason": "origin_overlaps_artifact_sample_window", "samples": 0}

    last_calibration_origin = origin_idx - horizon
    if horizon <= 0 or last_calibration_origin < 1:
        return {"applied": False, "reason": "not_enough_prior_actuals", "samples": 0}

    first_calibration_origin = max(1, last_calibration_origin - ONLINE_RESIDUAL_WINDOW + 1)
    residuals: list[list[float]] = []
    for calibration_origin_idx in range(first_calibration_origin, last_calibration_origin + 1):
        calibration_market = _point_in_time_window(
            full_market.model_copy(update={"candles": candles}),
            candles[: calibration_origin_idx + 1],
        )
        try:
            calibration_bundle = build_forecast(
                symbol=requested_symbol,
                interval=requested_interval,
                horizon=horizon,
                models=models,
                include_scenarios=False,
                allow_removed_models_as_warning=True,
                settings=settings,
                market_override=calibration_market,
            )
        except (ForecastUnavailable, ValueError):
            continue
        calibration_payload = chart_payload_from_forecast(calibration_bundle)
        calibration_actual = _actual_future_candles(candles, calibration_origin_idx, calibration_bundle.horizon)
        vector = _log_residual_vector(calibration_payload, calibration_actual, horizon)
        if vector is not None:
            residuals.append(vector)

    if len(residuals) < ONLINE_RESIDUAL_WINDOW:
        return {"applied": False, "reason": "not_enough_prior_residuals", "samples": len(residuals)}

    correction = []
    for step_values in zip(*residuals):
        raw_delta = ONLINE_RESIDUAL_GAIN * (sum(step_values) / len(step_values))
        correction.append(max(-ONLINE_RESIDUAL_MAX_ABS_LOG, min(ONLINE_RESIDUAL_MAX_ABS_LOG, raw_delta)))

    for key in ["predicted", "predicted_lower", "predicted_upper", "predicted_tail_lower", "predicted_tail_upper"]:
        series = payload.get(key)
        if isinstance(series, list):
            _apply_log_correction_to_series(series, correction)

    primary_model = payload.get("primary_model")
    for model in payload.get("forecast_models") or []:
        if primary_model and model.get("id") != primary_model:
            continue
        points = model.get("points")
        if isinstance(points, list):
            _apply_log_correction_to_series(points, correction)

    return {
        "applied": True,
        "samples": len(residuals),
        "window": ONLINE_RESIDUAL_WINDOW,
        "gain": ONLINE_RESIDUAL_GAIN,
        "max_abs_log_correction": ONLINE_RESIDUAL_MAX_ABS_LOG,
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
    online_calibration = _apply_online_residual_calibration(
        payload=payload,
        full_market=full_market,
        candles=candles,
        origin_idx=origin_idx,
        requested_symbol=requested_symbol,
        requested_interval=requested_interval,
        horizon=bundle.horizon,
        models=models,
        settings=current_settings,
    )
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
                "online_residual_calibration": online_calibration,
            },
        }
    )
    return payload
