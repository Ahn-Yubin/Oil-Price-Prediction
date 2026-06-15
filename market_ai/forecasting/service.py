from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_ai.config import Settings, get_settings
from market_ai.constants import (
    CONFIDENCE_Z,
    FALLBACK_INTERVAL,
    INTERVAL_TO_DELTA,
    INTERVAL_TO_HORIZON,
    INTERVAL_TO_MAX_LOG_BAND,
    INTERVAL_TO_RETURN_CLIP,
    select_model_horizon,
)
from market_ai.modeling.forecasters.baselines import BASELINE_FORECASTERS, ForecastContext
from market_ai.modeling.deep.availability import DeepArtifactAvailability, deep_artifact_availability
from market_ai.modeling.forecasters.deep_fusion import DeepModelUnavailable, forecast_with_deep_model
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError
from market_ai.modeling.forecasters.motif import forecast_model_comparison
from market_ai.modeling.calibration.conformal import apply_calibration_to_points, load_calibration_artifact
from market_ai.modeling.model_catalog import DEEP_MODELS, USER_FACING_MODELS, InvalidModelRequest, resolve_model_selection, split_model_query
from market_ai.llm.live_context import build_live_event_context
from market_ai.schemas.market import (
    DataStatusKind,
    ForecastPoint,
    ForecastResponse,
    ForecastWarning,
    MarketDataWindow,
    ModelInfo,
    RegimeProbabilities,
    ScenarioPoint,
    ScenarioResponse,
)
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable, load_market_data_window
from market_ai.data.storage import read_table
from market_ai.modeling.registry import ModelRegistry
from market_ai.data.related_assets import cross_asset_context_summary
from market_ai.data.symbols import asset_metadata
from market_ai.modeling.regimes.detector import detect_regime


class ForecastUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ForecastBundle:
    response: ForecastResponse
    market_data: MarketDataWindow
    forecast_models: list[dict[str, Any]]
    metrics: dict[str, Any]
    model_info: dict[str, Any]
    horizon: int


def _resolve_horizons(interval: str, requested_horizon: int | None) -> tuple[int, int]:
    return select_model_horizon(interval, requested_horizon)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _future_datetimes(last_time: int, interval: str, horizon: int) -> list[datetime]:
    start = pd.to_datetime(last_time, unit="s", utc=True).to_pydatetime()
    step = INTERVAL_TO_DELTA.get(interval, INTERVAL_TO_DELTA[FALLBACK_INTERVAL])
    return [start + step * (i + 1) for i in range(horizon)]


def _model_metrics(
    model_info: dict[str, Any],
    symbol: str,
    *,
    primary_model: dict[str, Any] | None = None,
    deep_model_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_id = str(primary_model.get("id")) if primary_model and primary_model.get("id") else None
    model_label = primary_model.get("label") if primary_model else None
    metrics = {
        "mae": model_info.get("val_mae_ret"),
        "rmse": model_info.get("val_rmse_ret"),
        "mape": model_info.get("val_mape_pct"),
        "symbol": symbol,
        "model": model_label or model_info.get("model_name", "Global DL model"),
        "model_id": model_id,
        "band_calibration": model_info.get("band_calibration"),
        "band_scale": model_info.get("band_scale"),
        "feature_version": model_info.get("feature_version"),
        "target_mode": model_info.get("target_mode"),
        "path_gain": model_info.get("path_gain"),
        "pattern_engine": model_info.get("pattern_engine"),
        "motif_matches": model_info.get("motif_matches"),
    }
    primary_deep_info = (deep_model_info or {}).get(model_id or "", {}) if model_id else {}
    primary_metadata = primary_deep_info.get("metadata", {}) if isinstance(primary_deep_info, dict) else {}
    primary_metrics = primary_metadata.get("metrics", {}) if isinstance(primary_metadata, dict) else {}
    if isinstance(primary_metrics, dict) and primary_metrics:
        metrics.update(
            {
                "mae": primary_metrics.get("validation_mae", metrics.get("mae")),
                "rmse": primary_metrics.get("validation_rmse", metrics.get("rmse")),
                "mape": primary_metrics.get("validation_mape", metrics.get("mape")),
                "model": model_label or primary_metadata.get("model_name") or metrics.get("model"),
                "feature_version": primary_metadata.get("feature_set") or metrics.get("feature_version"),
                "target_mode": primary_metadata.get("target") or metrics.get("target_mode"),
                "training_cutoff": primary_metadata.get("training_cutoff") or primary_metadata.get("train_end"),
            }
        )
    return metrics


def _quantile_points(
    *,
    current_price: float,
    future_times: list[datetime],
    p50_prices: np.ndarray,
    log_band: np.ndarray,
    data_status: DataStatusKind | str,
) -> list[ForecastPoint]:
    points: list[ForecastPoint] = []
    log_band = np.asarray(log_band, dtype=np.float64)
    if len(log_band) < len(p50_prices):
        pad = float(log_band[-1]) if len(log_band) else 0.02
        log_band = np.pad(log_band, (0, len(p50_prices) - len(log_band)), constant_values=pad)

    avg_band = float(np.mean(np.abs(log_band[: len(p50_prices)]))) if len(p50_prices) else 0.0
    base_confidence = 0.76 - min(avg_band * 1.8, 0.28)
    if str(data_status) != DataStatusKind.real.value:
        base_confidence -= 0.18

    for dt_value, p50, band in zip(future_times, p50_prices, log_band):
        p50 = max(_finite_float(p50, current_price), 1e-8)
        band = max(_finite_float(band, 0.02), 1e-6)
        mid_log = float(np.log(p50 / current_price))
        raw = {
            "p05": current_price * np.exp(mid_log - 1.25 * band),
            "p10": current_price * np.exp(mid_log - band),
            "p25": current_price * np.exp(mid_log - 0.45 * band),
            "p50": p50,
            "p75": current_price * np.exp(mid_log + 0.45 * band),
            "p90": current_price * np.exp(mid_log + band),
            "p95": current_price * np.exp(mid_log + 1.25 * band),
        }
        ordered = sorted(raw.values())
        expected_vol = max(float(abs(band)), 1e-6)
        score = mid_log / expected_vol
        prob_up = 0.5 + 0.25 * float(np.tanh(score))
        points.append(
            ForecastPoint(
                time=int(pd.Timestamp(dt_value).timestamp()),
                p05=float(ordered[0]),
                p10=float(ordered[1]),
                p25=float(ordered[2]),
                p50=float(ordered[3]),
                p75=float(ordered[4]),
                p90=float(ordered[5]),
                p95=float(ordered[6]),
                expected_return=mid_log,
                expected_volatility=expected_vol,
                prob_up=_clip01(prob_up),
                confidence=_clip01(base_confidence),
            )
        )
    return points


def _regime_from_close(close: np.ndarray) -> RegimeProbabilities:
    close = np.asarray(close, dtype=np.float64)
    returns = np.diff(np.log(close[close > 0.0]))
    if len(returns) < 8:
        return RegimeProbabilities().normalized()

    trend = float(np.sum(returns[-min(20, len(returns)) :]))
    short_vol = float(np.std(returns[-min(20, len(returns)) :]))
    long_vol = float(np.std(returns[-min(80, len(returns)) :])) if len(returns) >= 20 else short_vol
    high_vol_score = 0.35 if long_vol > 0 and short_vol > long_vol * 1.25 else 0.15
    trend_up = 0.35 if trend > short_vol else 0.15
    trend_down = 0.35 if trend < -short_vol else 0.15
    range_prob = 0.35 if abs(trend) <= max(short_vol, 1e-8) else 0.2
    regime = RegimeProbabilities(
        trend_up=trend_up,
        trend_down=trend_down,
        range=range_prob,
        high_volatility=high_vol_score,
        event_driven=0.05,
        confidence=0.55,
    )
    return regime.normalized()


def _scenario_response(points: list[ForecastPoint]) -> ScenarioResponse:
    return ScenarioResponse(
        bull=[ScenarioPoint(time=p.time, value=p.p90) for p in points],
        base=[ScenarioPoint(time=p.time, value=p.p50) for p in points],
        bear=[ScenarioPoint(time=p.time, value=p.p10) for p in points],
    )


def _logical_model_infos(comparison_models: list[dict[str, Any]], model_info: dict[str, Any], interval: str) -> list[ModelInfo]:
    infos: list[ModelInfo] = []
    for model in comparison_models:
        infos.append(
            ModelInfo(
                name=str(model.get("id")),
                label=model.get("label"),
                model_type="legacy_forecaster",
                version="phase3_adapter",
                status="available",
                supported_intervals=[interval],
                supported_asset_classes=["unknown"],
                training_cutoff=model_info.get("trained_at"),
                feature_version=model_info.get("feature_version"),
                metrics={
                    "val_mae_ret": model_info.get("val_mae_ret"),
                    "val_rmse_ret": model_info.get("val_rmse_ret"),
                    "val_mape_pct": model_info.get("val_mape_pct"),
                },
                notes=model.get("description"),
            )
        )
    return infos


def _baseline_comparison_models(close: np.ndarray, interval: str, horizon: int) -> list[dict[str, Any]]:
    context = ForecastContext(close=close, interval=interval, horizon=horizon, current_price=float(close[-1]))
    colors = {
        "random_walk": "#8b949e",
        "drift": "#ff7b72",
        "seasonal_naive": "#f2cc60",
        "volatility_scaled_naive": "#db6d28",
    }
    labels = {
        "random_walk": "Random Walk",
        "drift": "Drift",
        "seasonal_naive": "Seasonal Naive",
        "volatility_scaled_naive": "Vol-Scaled Naive",
    }
    out: list[dict[str, Any]] = []
    for name in ["random_walk", "drift", "seasonal_naive", "volatility_scaled_naive"]:
        result = BASELINE_FORECASTERS[name](context)
        prices = context.current_price * np.exp(result.cum_log_path)
        out.append(
            {
                "id": name,
                "label": labels[name],
                "description": result.metadata.get("description") or "Baseline forecast",
                "color": colors[name],
                "values": np.asarray(prices, dtype=np.float64),
                "quantile_prices": result.price_quantiles(context.current_price),
                "prob_up": result.prob_up,
                "expected_volatility": result.expected_volatility,
                "confidence": result.confidence,
            }
        )
    return out


def _add_warning(
    *,
    warnings: list[str],
    warning_objects: list[ForecastWarning],
    code: str,
    severity: str,
    message: str,
    action: str | None = None,
) -> None:
    warnings.append(message)
    warning_objects.append(ForecastWarning(code=code, severity=severity, message=message, action=action))


def _deep_availability_by_model(settings: Settings, interval: str, horizon: int) -> dict[str, DeepArtifactAvailability]:
    return {
        model_name: deep_artifact_availability(settings=settings, model_name=model_name, interval=interval, horizon=horizon)
        for model_name in DEEP_MODELS
    }


def _default_models_for_artifacts(availabilities: dict[str, DeepArtifactAvailability]) -> tuple[str, ...]:
    del availabilities
    return USER_FACING_MODELS


def _candle_frame_from_market(market: MarketDataWindow) -> pd.DataFrame:
    return pd.DataFrame([candle.model_dump() for candle in market.candles]).assign(
        date=lambda frame: pd.to_datetime(frame["time"], unit="s", utc=True)
    )


@lru_cache(maxsize=4)
def _load_processed_event_context_for_adapter(data_dir: str) -> pd.DataFrame | None:
    path = Path(data_dir) / "processed" / "event_context" / "event_context_daily.csv"
    if not path.exists():
        return None
    try:
        return read_table(path)
    except Exception:
        return None


def _normalize_event_context_for_adapter(frame: pd.DataFrame | None, *, symbol: str, as_of_time: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    time_col = next((col for col in ("feature_available_at", "timestamp", "date", "as_of_time") if col in out.columns), "")
    if not time_col:
        return pd.DataFrame()
    out["_feature_time"] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    out = out.dropna(subset=["_feature_time"]).sort_values("_feature_time")
    if "symbol" in out.columns:
        symbol_upper = symbol.upper()
        out = out[out["symbol"].astype(str).str.upper().isin([symbol_upper, "ALL", "*"])]
    cutoff = pd.to_datetime(as_of_time, unit="s", utc=True)
    return out[out["_feature_time"] <= cutoff].tail(1)


def _technical_rsi(close: np.ndarray, window: int = 14) -> float:
    if len(close) < window + 1:
        return 50.0
    diff = np.diff(close[-(window + 1) :])
    gain = float(np.sum(diff[diff > 0.0]) / window)
    loss = float(np.sum(-diff[diff < 0.0]) / window)
    if loss <= 1e-12:
        return 100.0 if gain > 0.0 else 50.0
    return float(100.0 - 100.0 / (1.0 + gain / loss))


def _detemplated_pattern_residual_path(current_price: float, deep_values: np.ndarray, pattern_values: np.ndarray) -> np.ndarray:
    horizon = min(len(deep_values), len(pattern_values))
    if horizon < 4:
        return deep_values
    deep_log = np.log(np.maximum(deep_values[:horizon], 1e-8) / current_price)
    pattern_log = np.log(np.maximum(pattern_values[:horizon], 1e-8) / current_price)
    ramp = np.linspace(1.0 / horizon, 1.0, horizon, dtype=np.float64)
    deep_residual = deep_log - ramp * deep_log[-1]
    pattern_residual = pattern_log - ramp * pattern_log[-1]
    adapted = current_price * np.exp(ramp * deep_log[-1] + 0.5 * deep_residual + 0.5 * pattern_residual)
    if len(deep_values) > horizon:
        adapted = np.concatenate([adapted, np.asarray(deep_values[horizon:], dtype=np.float64)])
    return adapted


def _shape_residual_from_paths(
    *,
    current_price: float,
    base_output: np.ndarray,
    close: np.ndarray,
    horizon: int,
    recent_vol: float,
) -> np.ndarray:
    x = np.linspace(1.0 / horizon, 1.0, horizon, dtype=np.float64)
    base_log = np.log(np.maximum(np.asarray(base_output[:horizon], dtype=np.float64), 1e-8) / current_price)
    residual = base_log - x * base_log[-1]
    if (float(np.max(residual) - np.min(residual)) if len(residual) else 0.0) < 0.012 and len(close) >= horizon + 1:
        recent_steps = np.diff(np.log(np.maximum(close[-(horizon + 1) :], 1e-8)))
        recent_cum = np.cumsum(recent_steps)
        residual = recent_cum - x * recent_cum[-1]
    cap = min(0.16, max(0.035, recent_vol * 5.5))
    return np.clip(residual, -cap, cap)


def _geopolitical_supply_shock_score(
    *,
    directional_bias: float,
    impact_score: float,
    bullish_event_score: float,
    bearish_event_score: float,
    geopolitical_event_score: float,
    raw_bullish_pressure: float,
    raw_bearish_pressure: float,
    raw_net_pressure: float,
    raw_energy_pressure: float,
    raw_geopolitical_pressure: float,
    raw_supply_pressure: float,
    source_diversity_score: float,
) -> float:
    geo = max(raw_geopolitical_pressure, geopolitical_event_score)
    bullish = max(raw_bullish_pressure, bullish_event_score, max(directional_bias, 0.0), max(raw_net_pressure, 0.0))
    bearish_offset = max(raw_bearish_pressure, bearish_event_score) - bullish
    score = (
        0.34 * geo
        + 0.24 * raw_supply_pressure
        + 0.12 * raw_energy_pressure
        + 0.13 * impact_score
        + 0.10 * bullish
        + 0.05 * source_diversity_score
        - 0.22 * max(bearish_offset, 0.0)
    )
    return _clip01(score)


def _event_upside_pressure_score(
    *,
    directional_bias: float,
    bullish_event_score: float,
    bearish_event_score: float,
    raw_bullish_pressure: float,
    raw_bearish_pressure: float,
    raw_net_pressure: float,
    raw_energy_pressure: float,
    raw_geopolitical_pressure: float,
    geopolitical_event_score: float,
) -> float:
    geo = max(raw_geopolitical_pressure, geopolitical_event_score)
    bullish = max(raw_bullish_pressure, bullish_event_score, max(directional_bias, 0.0), max(raw_net_pressure, 0.0))
    bearish = max(raw_bearish_pressure, bearish_event_score, max(-directional_bias, 0.0))
    score = 0.42 * max(raw_net_pressure, 0.0) + 0.22 * geo + 0.16 * raw_energy_pressure + 0.16 * bullish - 0.16 * max(bearish - bullish, 0.0)
    return _clip01(score)


def _event_regime_path_adapter(
    *,
    close: np.ndarray,
    interval: str,
    as_of_time: int,
    symbol: str,
    deep_values: np.ndarray,
    pattern_values: np.ndarray | None,
    event_context_frame: pd.DataFrame | None,
    settings: Settings,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(deep_values, dtype=np.float64).reshape(-1)
    if interval != "1d" or len(values) < 4:
        return values, {"applied": False, "reason": "unsupported_interval_or_horizon"}
    close = np.asarray(close, dtype=np.float64)
    close = close[np.isfinite(close) & (close > 0.0)]
    if len(close) < 20:
        return values, {"applied": False, "reason": "not_enough_history"}

    current_price = float(close[-1])
    pattern_arr = np.asarray(pattern_values, dtype=np.float64).reshape(-1) if pattern_values is not None else None
    output = values
    adapter = "none"
    returns = np.diff(np.log(close))
    recent_60 = returns[-min(60, len(returns)) :] if len(returns) else np.asarray([], dtype=np.float64)
    recent_30 = returns[-min(30, len(returns)) :] if len(returns) else np.asarray([], dtype=np.float64)
    sum60 = float(np.sum(recent_60)) if len(recent_60) else 0.0
    sum30 = float(np.sum(recent_30)) if len(recent_30) else 0.0
    recent_vol = max(float(np.std(recent_60)) if len(recent_60) else 0.0, 1e-8)
    rsi14 = _technical_rsi(close)
    max20 = float(np.max(close[-20:]))
    event_frame = event_context_frame
    if event_frame is None:
        event_frame = _load_processed_event_context_for_adapter(str(settings.data_dir))
    event_row = _normalize_event_context_for_adapter(event_frame, symbol=symbol, as_of_time=as_of_time)
    if not event_row.empty:
        row = event_row.iloc[0]
        directional_bias = _finite_float(row.get("directional_bias_score"), 0.0)
        impact_score = _finite_float(row.get("impact_score"), 0.0)
        bullish_event_score = _finite_float(row.get("bullish_event_score"), 0.0)
        bearish_event_score = _finite_float(row.get("bearish_event_score"), 0.0)
        geopolitical_event_score = _finite_float(row.get("geopolitical_event_score"), 0.0)
        raw_bullish = _finite_float(row.get("raw_bullish_pressure"), 0.0)
        raw_bearish = _finite_float(row.get("raw_bearish_pressure"), 0.0)
        raw_net = _finite_float(row.get("raw_net_pressure"), 0.0)
        raw_energy = _finite_float(row.get("raw_energy_pressure"), 0.0)
        geopolitical = _finite_float(row.get("raw_geopolitical_pressure"), 0.0)
        raw_supply = _finite_float(row.get("raw_supply_pressure"), 0.0)
        source_diversity = _finite_float(row.get("source_diversity_score"), 0.0)
    else:
        directional_bias = 0.0
        impact_score = 0.0
        bullish_event_score = 0.0
        bearish_event_score = 0.0
        geopolitical_event_score = 0.0
        raw_bullish = 0.0
        raw_bearish = 0.0
        raw_net = 0.0
        raw_energy = 0.0
        geopolitical = 0.0
        raw_supply = 0.0
        source_diversity = 0.0

    supply_shock_score = _geopolitical_supply_shock_score(
        directional_bias=directional_bias,
        impact_score=impact_score,
        bullish_event_score=bullish_event_score,
        bearish_event_score=bearish_event_score,
        geopolitical_event_score=geopolitical_event_score,
        raw_bullish_pressure=raw_bullish,
        raw_bearish_pressure=raw_bearish,
        raw_net_pressure=raw_net,
        raw_energy_pressure=raw_energy,
        raw_geopolitical_pressure=geopolitical,
        raw_supply_pressure=raw_supply,
        source_diversity_score=source_diversity,
    )
    event_upside_score = _event_upside_pressure_score(
        directional_bias=directional_bias,
        bullish_event_score=bullish_event_score,
        bearish_event_score=bearish_event_score,
        raw_bullish_pressure=raw_bullish,
        raw_bearish_pressure=raw_bearish,
        raw_net_pressure=raw_net,
        raw_energy_pressure=raw_energy,
        raw_geopolitical_pressure=geopolitical,
        geopolitical_event_score=geopolitical_event_score,
    )

    horizon = len(values)
    x = np.linspace(1.0 / horizon, 1.0, horizon, dtype=np.float64)
    base_output = (
        _detemplated_pattern_residual_path(current_price, values, pattern_arr)
        if pattern_arr is not None and len(pattern_arr) >= len(values)
        else values
    )
    shape_residual = _shape_residual_from_paths(
        current_price=current_price,
        base_output=base_output,
        close=close,
        horizon=horizon,
        recent_vol=recent_vol,
    )
    if rsi14 > 68.0 and sum30 > 0.45 and current_price >= max20 * 0.995:
        terminal = -min(0.08, max(0.04, recent_vol * 1.6))
        depth = -min(0.24, max(0.12, recent_vol * 5.8))
        dip = np.exp(-((x - 0.28) / 0.22) ** 2)
        output = current_price * np.exp(terminal * x + depth * dip * (1.0 - x * 0.30))
        adapter = "overextended_mean_reversion"
    elif (
        supply_shock_score >= 0.55
        and raw_supply >= 0.55
        and max(geopolitical, geopolitical_event_score) >= 0.55
        and max(raw_bullish, bullish_event_score, raw_net) > -0.10
    ):
        momentum_penalty = 0.035 * max(sum30 - 0.20, 0.0) / 0.20
        terminal = min(0.28, max(0.07, 0.04 + 0.22 * supply_shock_score + 0.08 * max(raw_net, 0.0) - momentum_penalty))
        jump = 1.0 / (1.0 + np.exp(-(x - 0.16) / 0.055))
        jump = (jump - jump[0]) / max(float(jump[-1] - jump[0]), 1e-8)
        trend = 0.68 * jump + 0.32 * np.power(x, 0.78)
        wave = 0.014 * supply_shock_score * np.sin(np.linspace(0.0, 2.6 * np.pi, horizon)) * (1.0 - 0.45 * x)
        output = current_price * np.exp(terminal * trend + wave + 0.45 * shape_residual * (1.0 - 0.10 * x))
        adapter = "geopolitical_supply_shock"
    elif (
        directional_bias > 0.0
        and raw_net > 0.72
        and geopolitical > 0.65
        and 0.04 < sum60 < 0.25
        and rsi14 < 66.0
        and current_price <= max20 * 1.002
    ):
        terminal = min(0.36, max(0.10, sum60 * (2.5 + raw_net)))
        progress = 1.0 / (1.0 + np.exp(-(x - 0.18) / 0.08))
        progress = (progress - progress[0]) / max(float(progress[-1] - progress[0]), 1e-8)
        output = current_price * np.exp(terminal * progress + 0.55 * shape_residual * (1.0 - 0.12 * x))
        adapter = "bullish_event_breakout"
    elif (
        event_upside_score >= 0.42
        and max(geopolitical, geopolitical_event_score, raw_energy) >= 0.55
        and 0.035 < sum60 < 0.35
        and rsi14 < 95.0
    ):
        blend = min(max((event_upside_score - 0.42) / 0.20, 0.0), 1.0)
        terminal = min(0.34, max(0.06, sum60 * (1.55 + 1.85 * event_upside_score) + 0.06 * event_upside_score))
        progress = 1.0 / (1.0 + np.exp(-(x - 0.19) / 0.09))
        progress = (progress - progress[0]) / max(float(progress[-1] - progress[0]), 1e-8)
        event_path = current_price * np.exp(terminal * progress + 0.70 * shape_residual * (1.0 - 0.10 * x))
        base_log = np.log(np.maximum(base_output, 1e-8) / current_price)
        event_log = np.log(np.maximum(event_path, 1e-8) / current_price)
        output = current_price * np.exp((1.0 - blend) * base_log + blend * event_log)
        adapter = "event_risk_premium"
    elif pattern_arr is not None and len(pattern_arr) >= len(values):
        output = base_output
        adapter = "pattern_residual_detemplate"

    return np.asarray(output, dtype=np.float64), {
        "applied": adapter != "none",
        "adapter": adapter,
        "rationale": (
            "Geopolitical/supply shock context lifted the model median path."
            if adapter == "geopolitical_supply_shock"
            else "Recent price pattern residuals reduced repeated horizon template shape."
            if adapter == "pattern_residual_detemplate"
            else "Overextended recent price action favored a mean-reversion path."
            if adapter == "overextended_mean_reversion"
            else "Bullish event context and momentum supported a breakout path."
            if adapter == "bullish_event_breakout"
            else "Persistent geopolitical/energy news pressure added an upside risk premium."
            if adapter == "event_risk_premium"
            else ""
        ),
        "geopolitical_supply_shock_score": supply_shock_score,
        "event_upside_pressure_score": event_upside_score,
        "rsi14": rsi14,
        "recent_sum_30": sum30,
        "recent_sum_60": sum60,
        "impact_score": impact_score,
        "raw_bullish_pressure": raw_bullish,
        "raw_bearish_pressure": raw_bearish,
        "raw_net_pressure": raw_net,
        "raw_energy_pressure": raw_energy,
        "raw_geopolitical_pressure": geopolitical,
        "raw_supply_pressure": raw_supply,
        "bullish_event_score": bullish_event_score,
        "bearish_event_score": bearish_event_score,
        "geopolitical_event_score": geopolitical_event_score,
        "directional_bias_score": directional_bias,
    }


def _deep_comparison_models(
    *,
    close: np.ndarray,
    interval: str,
    horizon: int,
    market: MarketDataWindow,
    selected: list[str],
    settings: Settings,
    warnings: list[str],
    warning_objects: list[ForecastWarning],
    artifact_status: dict[str, str],
    availabilities: dict[str, DeepArtifactAvailability],
    deep_model_info: dict[str, Any],
    explicit_request: bool,
    event_context_frame: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    frame = _candle_frame_from_market(market)
    for model_name in selected:
        if model_name not in DEEP_MODELS:
            continue
        try:
            model = forecast_with_deep_model(
                model_name=model_name,
                close=close,
                interval=interval,
                horizon=horizon,
                settings=settings,
                symbol=market.symbol.provider_symbol,
                candles=frame,
                event_context_frame=event_context_frame if model_name in {"oil_context_fusion", "llm_context_seq_moe"} else None,
            )
        except DeepModelUnavailable as exc:
            availability = availabilities.get(model_name)
            artifact_status[model_name] = availability.status if availability is not None else "unavailable"
            should_warn = explicit_request or model_name in USER_FACING_MODELS or (availability is not None and availability.is_available)
            if should_warn:
                action = availability.training_command if availability is not None else None
                expected = availability.expected_artifact_file if availability is not None else "expected deep artifact"
                message = (
                    f"{model_name} artifact unavailable for {interval}/horizon {horizon}; "
                    f"falling back to internal benchmark models. Expected {expected}. Detail: {exc}"
                )
                _add_warning(
                    warnings=warnings,
                    warning_objects=warning_objects,
                    code="deep_artifact_unavailable",
                    severity="warning",
                    message=message,
                    action=action,
                )
            continue
        artifact_status[model_name] = "available"
        deep_model_info[model_name] = {
            "artifact_file": model.get("artifact_file"),
            "metadata": model.get("metadata", {}),
        }
        out.append(model)
    return out


def _forecast_points_from_model(
    *,
    model: dict[str, Any],
    current_price: float,
    future_times: list[datetime],
    fallback_band: np.ndarray,
    data_status: DataStatusKind | str,
    max_log_band: float | None = None,
) -> list[ForecastPoint]:
    q_prices = model.get("quantile_prices")
    if isinstance(q_prices, dict) and {"p05", "p10", "p25", "p50", "p75", "p90", "p95"}.issubset(q_prices):
        points: list[ForecastPoint] = []
        prob_up = np.asarray(model.get("prob_up", []), dtype=np.float64)
        expected_vol = np.asarray(model.get("expected_volatility", []), dtype=np.float64)
        confidence = np.asarray(model.get("confidence", []), dtype=np.float64).reshape(-1)
        display_values = np.asarray(model.get("values", []), dtype=np.float64).reshape(-1)
        fallback_band = np.asarray(fallback_band, dtype=np.float64).reshape(-1)
        quantile_arrays = {key: np.asarray(q_prices[key], dtype=np.float64).reshape(-1) for key in ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]}
        if any(len(values) < len(future_times) for values in quantile_arrays.values()):
            return _quantile_points(
                current_price=current_price,
                future_times=future_times,
                p50_prices=np.asarray(model["values"], dtype=np.float64),
                log_band=fallback_band,
                data_status=data_status,
            )
        max_band = max(_finite_float(max_log_band, 0.0), 0.0) or None
        for idx, dt_value in enumerate(future_times):
            raw = {key: max(_finite_float(quantile_arrays[key][idx], 0.0), 1e-8) for key in quantile_arrays}
            raw_p50 = max(_finite_float(raw.get("p50"), current_price), 1e-8)
            display_p50 = _finite_float(display_values[idx] if idx < len(display_values) else raw_p50, raw_p50)
            p50 = max(display_p50, 1e-8)
            fallback = abs(_finite_float(fallback_band[idx] if idx < len(fallback_band) else None, 0.02))

            def distance(value: float, anchor: float = raw_p50) -> float:
                return abs(float(np.log(max(value, 1e-8) / max(anchor, 1e-8))))

            band80 = max(distance(raw["p10"]), distance(raw["p90"]), fallback, 1e-6)
            band50 = max(distance(raw["p25"]), distance(raw["p75"]), band80 * 0.45, 1e-6)
            tail95 = max(distance(raw["p05"]), distance(raw["p95"]), band80 * 1.25, 1e-6)
            if max_band is not None:
                band80 = min(band80, max_band)
                band50 = min(band50, band80 * 0.9)
                tail95 = min(max(tail95, band80 * 1.15), max_band * 1.25)
            band50 = min(band50, band80 * 0.9)
            tail95 = max(tail95, band80 * 1.15)
            mid_log = float(np.log(p50 / current_price))
            p05 = float(p50 * np.exp(-tail95))
            p10 = float(p50 * np.exp(-band80))
            p25 = float(p50 * np.exp(-band50))
            p75 = float(p50 * np.exp(band50))
            p90 = float(p50 * np.exp(band80))
            p95 = float(p50 * np.exp(tail95))
            points.append(
                ForecastPoint(
                    time=int(pd.Timestamp(dt_value).timestamp()),
                    p05=p05,
                    p10=p10,
                    p25=p25,
                    p50=p50,
                    p75=p75,
                    p90=p90,
                    p95=p95,
                    expected_return=mid_log,
                    expected_volatility=max(_finite_float(expected_vol[idx] if idx < len(expected_vol) else abs(mid_log), 0.0), band80),
                    prob_up=_clip01(prob_up[idx] if idx < len(prob_up) else 0.5),
                    confidence=_clip01(confidence[idx] if idx < len(confidence) else confidence[0] if len(confidence) else 0.55),
                )
            )
        return points
    return _quantile_points(
        current_price=current_price,
        future_times=future_times,
        p50_prices=np.asarray(model["values"], dtype=np.float64),
        log_band=fallback_band,
        data_status=data_status,
    )


def _band_explanation(
    *,
    primary_model: dict[str, Any],
    model_info: dict[str, Any],
    calibration_status: dict[str, Any],
    interval: str,
    horizon: int,
) -> dict[str, Any]:
    status = str(calibration_status.get("calibration_status") or "uncalibrated")
    display_status = "calibrated" if status == "calibrated" else "volatility_estimated"
    return {
        "status": display_status,
        "source": "conformal_calibration" if display_status == "calibrated" else "model_quantiles_and_recent_volatility",
        "primary_model": str(primary_model.get("id") or "unknown"),
        "interval": interval,
        "horizon": horizon,
        "band_scale": model_info.get("band_scale"),
        "recent_step_vol": model_info.get("recent_step_vol"),
        "n_origins": calibration_status.get("n_origins"),
        "coverage_80": calibration_status.get("coverage_80"),
        "coverage_status": calibration_status.get("coverage_status") or ("measured" if display_status == "calibrated" else "not_measured"),
        "raw_calibration_status": calibration_status.get("raw_calibration_status"),
        "method": (
            "Rolling backtest conformal adjustment widens or narrows the selected model quantile path."
            if display_status == "calibrated"
            else (
                "The selected model supplies a median path and quantile/residual scale; the service combines that "
                "with recent realized volatility so every supported symbol still receives P10-P90 and P05-P95 bands."
            )
        ),
    }


def build_forecast(
    *,
    symbol: str,
    interval: str,
    horizon: int | None = None,
    models: str | None = None,
    include_explanation: bool = False,
    include_scenarios: bool = True,
    allow_removed_models_as_warning: bool = False,
    settings: Settings | None = None,
    market_override: MarketDataWindow | None = None,
    event_context_frame_override: pd.DataFrame | None = None,
    llm_context_summary_override: dict[str, Any] | None = None,
    apply_event_path_adapter: bool = True,
) -> ForecastBundle:
    del include_explanation
    settings = settings or get_settings()
    requested_symbol_input = symbol or settings.default_symbol
    forecast_symbol = settings.default_symbol or "CL=F"
    oil_symbol_forced = str(requested_symbol_input or "").strip() != str(forecast_symbol).strip()
    requested_models = split_model_query(models)
    try:
        market = market_override or load_market_data_window(forecast_symbol, interval, settings=settings)
    except MarketDataUnavailable:
        raise

    resolved_interval = market.timeframe.normalized
    model_horizon, output_horizon = _resolve_horizons(resolved_interval, horizon)
    deep_availabilities = _deep_availability_by_model(settings, resolved_interval, model_horizon)
    default_models = _default_models_for_artifacts(deep_availabilities)
    selection = resolve_model_selection(
        models,
        supported=USER_FACING_MODELS,
        default=default_models,
        allow_removed_as_warning=allow_removed_models_as_warning,
    )
    close = np.asarray([candle.close for candle in market.candles], dtype=np.float64)
    if len(close) == 0:
        raise ForecastUnavailable("Forecast requires at least one close price.")

    warnings: list[str] = []
    warning_objects: list[ForecastWarning] = []
    if oil_symbol_forced:
        _add_warning(
            warnings=warnings,
            warning_objects=warning_objects,
            code="oil_symbol_forced",
            severity="info",
            message=f"Oil-only mode is enabled; requested symbol {requested_symbol_input} was mapped to {forecast_symbol}.",
        )
    pretrained_missing_message: str | None = None
    pretrained_missing_action: str | None = None
    try:
        comparison_models, model_info = forecast_model_comparison(
            close=close,
            interval=resolved_interval,
            horizon=model_horizon,
            z_value=CONFIDENCE_Z,
            return_clip=INTERVAL_TO_RETURN_CLIP.get(resolved_interval, 0.05),
            max_log_band=INTERVAL_TO_MAX_LOG_BAND.get(resolved_interval, 0.25),
        )
    except PretrainedModelNotFoundError as exc:
        comparison_models = []
        fallback_band = min(INTERVAL_TO_MAX_LOG_BAND.get(resolved_interval, 0.1), 0.02)
        model_info = {
            "model_name": "Deep/baseline only forecast",
            "band_calibration": "fallback_recent_volatility",
            "_ci_band_values": np.full(model_horizon, fallback_band, dtype=np.float64),
        }
        pretrained_missing_message = (
            f"Pattern/Motif pretrained artifact is unavailable for {resolved_interval}/horizon {model_horizon}; "
            "serving available deep and baseline models only."
        )
        pretrained_missing_action = str(exc)
    except Exception as exc:
        raise ForecastUnavailable(str(exc)) from exc

    if pretrained_missing_message:
        _add_warning(
            warnings=warnings,
            warning_objects=warning_objects,
            code="pretrained_pattern_artifact_unavailable",
            severity="warning",
            message=pretrained_missing_message,
            action=pretrained_missing_action,
        )
    for message in selection.warnings:
        _add_warning(
            warnings=warnings,
            warning_objects=warning_objects,
            code="removed_model_requested",
            severity="warning",
            message=message,
        )
    artifact_status: dict[str, str] = {
        model_name: availability.status for model_name, availability in deep_availabilities.items()
    }
    deep_model_info: dict[str, Any] = {}
    live_context: dict[str, Any] | None = None
    live_event_context_frame: pd.DataFrame | None = event_context_frame_override
    use_live_event_context = market_override is None and event_context_frame_override is None
    if use_live_event_context and any(model_name in {"oil_context_fusion", "llm_context_seq_moe"} for model_name in selection.selected):
        try:
            live_context = build_live_event_context(
                symbol=market.symbol.provider_symbol,
                settings=settings,
                as_of_time=datetime.fromtimestamp(market.candles[-1].time, tz=timezone.utc),
            )
            live_event_context_frame = live_context.get("context_frame")
        except Exception as exc:
            _add_warning(
                warnings=warnings,
                warning_objects=warning_objects,
                code="live_event_context_unavailable",
                severity="warning",
                message=f"Live news event context unavailable; oil_context_fusion will use cached or file context if available. Detail: {exc}",
                action="Check public news connectivity and LLM context settings.",
            )
    explicit_deep_request = bool(requested_models) and any(model_name in DEEP_MODELS for model_name in selection.selected)
    comparison_models = (
        _deep_comparison_models(
            close=close,
            interval=resolved_interval,
            horizon=model_horizon,
            market=market,
            selected=selection.selected,
            settings=settings,
            warnings=warnings,
            warning_objects=warning_objects,
            artifact_status=artifact_status,
            availabilities=deep_availabilities,
            deep_model_info=deep_model_info,
            explicit_request=explicit_deep_request,
            event_context_frame=live_event_context_frame,
        )
        + comparison_models
        + _baseline_comparison_models(close, resolved_interval, model_horizon)
    )
    model_by_id: dict[str, dict[str, Any]] = {}
    for item in comparison_models:
        model_by_id[str(item.get("id"))] = item
    oil_model = model_by_id.get("oil_context_fusion")
    if oil_model is not None and apply_event_path_adapter:
        pattern_model = model_by_id.get("pattern_mlp")
        adapted_values, adapter_info = _event_regime_path_adapter(
            close=close,
            interval=resolved_interval,
            as_of_time=market.candles[-1].time,
            symbol=market.symbol.provider_symbol,
            deep_values=np.asarray(oil_model.get("values", []), dtype=np.float64),
            pattern_values=np.asarray(pattern_model.get("values", []), dtype=np.float64) if pattern_model is not None else None,
            event_context_frame=live_event_context_frame,
            settings=settings,
        )
        if adapter_info.get("applied"):
            oil_model["values"] = adapted_values
            oil_model["median_values"] = adapted_values
            oil_model["point_path_kind"] = adapter_info.get("adapter")
            oil_model["path_adapter"] = adapter_info
            if "oil_context_fusion" in deep_model_info:
                deep_model_info["oil_context_fusion"]["path_adapter"] = adapter_info
    comparison_models = [model_by_id[name] for name in selection.selected if name in model_by_id]
    if not comparison_models:
        fallback_order = ["motif", "pattern_mlp", "random_walk"]
        comparison_models = [model_by_id[name] for name in fallback_order if name in model_by_id]
    if not comparison_models:
        raise ForecastUnavailable("No selected forecast model output was available.")

    primary_priority = (
        ["oil_context_fusion", "motif", "pattern_mlp", "random_walk"]
        if not selection.requested
        else selection.selected
    )
    primary = next((model_by_id[name] for name in primary_priority if name in model_by_id and model_by_id[name] in comparison_models), comparison_models[0])
    p50_prices = np.asarray(primary["values"], dtype=np.float64)[:output_horizon]
    band = np.asarray(model_info.get("_ci_band_values", []), dtype=np.float64)[:output_horizon]
    if len(band) < output_horizon:
        fallback_band = min(INTERVAL_TO_MAX_LOG_BAND.get(resolved_interval, 0.1), 0.02)
        band = np.pad(band, (0, output_horizon - len(band)), constant_values=fallback_band)

    last_candle = market.candles[-1]
    current_price = float(last_candle.close)
    future_times = _future_datetimes(last_candle.time, resolved_interval, output_horizon)
    forecast_points = _quantile_points(
        current_price=current_price,
        future_times=future_times,
        p50_prices=p50_prices,
        log_band=band,
        data_status=market.data_status.status,
    )
    forecast_points = _forecast_points_from_model(
        model=primary,
        current_price=current_price,
        future_times=future_times,
        fallback_band=band,
        data_status=market.data_status.status,
        max_log_band=INTERVAL_TO_MAX_LOG_BAND.get(resolved_interval),
    )
    calibration_artifact = load_calibration_artifact(str(primary.get("id")), market.symbol.provider_symbol, resolved_interval)
    forecast_points = apply_calibration_to_points(
        forecast_points,
        current_price=current_price,
        artifact=calibration_artifact,
    )

    for message in market.data_status.warnings:
        _add_warning(
            warnings=warnings,
            warning_objects=warning_objects,
            code="data_status_warning",
            severity="warning",
            message=message,
        )
    calibration_status = (
        calibration_artifact.as_dict()
        if calibration_artifact is not None
        else {
            "model": str(primary.get("id")),
            "symbol": market.symbol.provider_symbol,
            "interval": resolved_interval,
            "calibration_status": "uncalibrated",
        }
    )
    if calibration_artifact is None or calibration_artifact.calibration_status != "calibrated":
        raw_status = str(calibration_status.get("calibration_status") or "uncalibrated")
        calibration_status = {
            **calibration_status,
            "calibration_status": "volatility_estimated",
            "coverage_status": "not_measured" if calibration_artifact is None else "insufficient_origins",
            "raw_calibration_status": raw_status,
        }
        _add_warning(
            warnings=warnings,
            warning_objects=warning_objects,
            code="quantile_bands_volatility_estimated",
            severity="info",
            message=(
                "Forecast bands are built from the selected model quantile/residual scale and recent realized "
                "volatility for this model/symbol/interval; measured coverage calibration is not available yet."
            ),
            action="Run rolling coverage calibration before labeling bands as validated confidence intervals.",
        )
    band_explanation = _band_explanation(
        primary_model=primary,
        model_info=model_info,
        calibration_status=calibration_status,
        interval=resolved_interval,
        horizon=output_horizon,
    )
    status_value = str(market.data_status.status)
    if status_value in {DataStatusKind.mock.value, DataStatusKind.fallback.value, DataStatusKind.stale.value}:
        _add_warning(
            warnings=warnings,
            warning_objects=warning_objects,
            code="data_quality_reduced",
            severity="warning",
            message=f"Data status is {market.data_status.status}; confidence is reduced.",
            action="Verify live data availability or explicitly enable development fallback only outside production.",
        )

    forecast_models = []
    for model in comparison_models:
        values = np.asarray(model["values"], dtype=np.float64)[:output_horizon]
        forecast_models.append(
            {
                "id": model.get("id"),
                "label": model.get("label"),
                "description": model.get("description"),
                "color": model.get("color"),
                "points": [{"time": last_candle.time, "value": current_price}]
                + [
                    {"time": int(pd.Timestamp(dt_value).timestamp()), "value": float(value)}
                    for dt_value, value in zip(future_times, values)
                ],
            }
        )

    registry_infos = ModelRegistry(settings).list_model_info()
    logical_infos = _logical_model_infos(comparison_models, model_info, resolved_interval)
    metadata = asset_metadata(market.symbol)
    regime_detection = detect_regime(close)
    cross_asset_context = cross_asset_context_summary(
        market.symbol.provider_symbol,
        metadata.asset_class,
        settings.enable_cross_asset_features,
    )
    if settings.enable_cross_asset_features and cross_asset_context.get("warning"):
        _add_warning(
            warnings=warnings,
            warning_objects=warning_objects,
            code="cross_asset_context_warning",
            severity="warning",
            message=cross_asset_context["warning"],
        )
    llm_context_summary = {
        "enabled": settings.enable_llm_context,
        "external_calls_enabled": settings.enable_external_llm_calls,
        "role": "context/event encoder only",
        "event_context_source": (
            live_context.get("source")
            if live_context
            else "scenario_override"
            if event_context_frame_override is not None
            else "processed_or_file"
        ),
        "live_news_count": int(len(live_context.get("news"))) if live_context is not None and live_context.get("news") is not None else 0,
        "event_count": int((live_context.get("context_points") or [{}])[0].get("event_count", 0)) if live_context else 0,
        "overall_bias": (live_context.get("context_points") or [{}])[0].get("overall_bias") if live_context else None,
        "impact_score": (live_context.get("context_points") or [{}])[0].get("impact_score") if live_context else None,
    }
    if llm_context_summary_override:
        llm_context_summary.update(llm_context_summary_override)

    response = ForecastResponse(
        symbol=market.symbol.provider_symbol,
        asset_metadata=metadata,
        interval=resolved_interval,
        generated_at=datetime.now(timezone.utc),
        current_price=current_price,
        model_version=model_info.get("feature_version") or "phase3_adapter",
        training_cutoff=model_info.get("trained_at"),
        data_status=market.data_status,
        candles=market.candles,
        forecast=forecast_points,
        scenarios=_scenario_response(forecast_points) if include_scenarios else ScenarioResponse(),
        regime=regime_detection.probabilities,
        models=logical_infos + registry_infos,
        cross_asset_context=cross_asset_context,
        warnings=warnings,
        warning_objects=warning_objects,
        model_paths=forecast_models,
        selected_models=selection.selected,
        primary_model=str(primary.get("id")),
        deprecated_models_requested=selection.deprecated_requested,
        removed_models_requested=selection.removed_requested,
        llm_context_summary=llm_context_summary,
        deep_model_info=deep_model_info,
        feature_version=model_info.get("feature_version") or model_info.get("feature_set"),
        artifact_status=artifact_status,
        calibration_status=calibration_status,
        band_explanation=band_explanation,
    )
    return ForecastBundle(
        response=response,
        market_data=market,
        forecast_models=forecast_models,
        metrics=_model_metrics(
            model_info,
            market.symbol.provider_symbol,
            primary_model=primary,
            deep_model_info=deep_model_info,
        ),
        model_info=model_info,
        horizon=output_horizon,
    )


def chart_payload_from_forecast(bundle: ForecastBundle) -> dict[str, Any]:
    response = bundle.response
    actionable_warnings = [item.message for item in response.warning_objects if item.severity in {"warning", "error"}]
    candles = [candle.model_dump() for candle in bundle.market_data.candles]
    anchor = {"time": bundle.market_data.candles[-1].time, "value": response.current_price}
    predicted = [anchor] + [{"time": point.time, "value": point.p50} for point in response.forecast]
    primary_path = next((model for model in bundle.forecast_models if str(model.get("id")) == str(response.primary_model)), None)
    if primary_path and primary_path.get("points"):
        predicted = list(primary_path["points"])
    predicted_lower = [anchor] + [{"time": point.time, "value": point.p10} for point in response.forecast]
    predicted_upper = [anchor] + [{"time": point.time, "value": point.p90} for point in response.forecast]
    primary_deep_metadata = (response.deep_model_info.get(response.primary_model or "", {}) or {}).get("metadata", {})
    model_training_cutoff = (
        primary_deep_metadata.get("training_cutoff")
        or primary_deep_metadata.get("train_end")
        or bundle.model_info.get("training_cutoff")
        or bundle.model_info.get("trained_at")
    )

    return {
        "candles": candles,
        "predicted": predicted,
        "predicted_lower": predicted_lower,
        "predicted_upper": predicted_upper,
        "predicted_tail_lower": [anchor] + [{"time": point.time, "value": point.p05} for point in response.forecast],
        "predicted_tail_upper": [anchor] + [{"time": point.time, "value": point.p95} for point in response.forecast],
        "forecast_models": bundle.forecast_models,
        "metrics": bundle.metrics,
        "symbol_input": response.data_status.symbol_requested,
        "symbol_resolved": response.data_status.symbol_resolved,
        "interval_resolved": response.data_status.interval_resolved,
        "interval_requested": response.data_status.interval_requested,
        "updated_at": response.generated_at.isoformat(),
        "data_source": response.data_status.source,
        "data_status": response.data_status.model_dump(),
        "warning": " ".join(actionable_warnings) if actionable_warnings else None,
        "warnings": response.warnings,
        "warning_objects": [item.model_dump() for item in response.warning_objects],
        "forecast_horizon": bundle.horizon,
        "selected_models": response.selected_models,
        "primary_model": response.primary_model,
        "deprecated_models_requested": response.deprecated_models_requested,
        "removed_models_requested": response.removed_models_requested,
        "llm_context_summary": response.llm_context_summary,
        "deep_model_info": response.deep_model_info,
        "feature_version": response.feature_version,
        "artifact_status": response.artifact_status,
        "calibration_status": response.calibration_status,
        "band_explanation": response.band_explanation,
        "confidence_level": None,
        "band_label": "calibrated conformal quantile band"
        if response.calibration_status.get("calibration_status") == "calibrated"
        else "volatility-estimated quantile band",
        "model_trained_at": model_training_cutoff,
        "model_train_symbols": bundle.model_info.get("train_symbols"),
        "model_sample_info": {
            "n_train": bundle.model_info.get("n_train"),
            "n_val": bundle.model_info.get("n_val"),
        },
        "asset_metadata": response.asset_metadata.model_dump(),
        "regime": response.regime.model_dump(),
        "regime_label": detect_regime([candle.close for candle in bundle.market_data.candles]).label,
        "cross_asset_context": response.cross_asset_context,
    }
