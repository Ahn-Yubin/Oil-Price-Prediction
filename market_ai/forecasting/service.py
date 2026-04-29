from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
)
from market_ai.modeling.forecasters.baselines import BASELINE_FORECASTERS, ForecastContext
from market_ai.modeling.regimes.moe import regime_ensemble_forecast
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError
from market_ai.modeling.forecasters.motif import forecast_model_comparison
from market_ai.schemas.market import (
    DataStatusKind,
    ForecastPoint,
    ForecastResponse,
    MarketDataWindow,
    ModelInfo,
    RegimeProbabilities,
    ScenarioPoint,
    ScenarioResponse,
)
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable, load_market_data_window
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


def _model_metrics(model_info: dict[str, Any], symbol: str) -> dict[str, Any]:
    return {
        "mae": model_info.get("val_mae_ret"),
        "rmse": model_info.get("val_rmse_ret"),
        "mape": model_info.get("val_mape_pct"),
        "symbol": symbol,
        "model": model_info.get("model_name", "Global DL model"),
        "band_calibration": model_info.get("band_calibration"),
        "band_scale": model_info.get("band_scale"),
        "feature_version": model_info.get("feature_version"),
        "target_mode": model_info.get("target_mode"),
        "path_gain": model_info.get("path_gain"),
        "pattern_engine": model_info.get("pattern_engine"),
        "motif_matches": model_info.get("motif_matches"),
    }


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
        "seasonal_naive": "#f2cc60",
        "volatility_scaled_naive": "#db6d28",
        "simple_moving_average_path": "#2dd4bf",
    }
    labels = {
        "random_walk": "Random Walk",
        "seasonal_naive": "Seasonal Naive",
        "volatility_scaled_naive": "Vol-Scaled Naive",
        "simple_moving_average_path": "SMA Path",
    }
    out: list[dict[str, Any]] = []
    for name in ["random_walk", "seasonal_naive", "volatility_scaled_naive", "simple_moving_average_path"]:
        result = BASELINE_FORECASTERS[name](context)
        prices = context.current_price * np.exp(result.cum_log_path)
        out.append(
            {
                "id": name,
                "label": labels[name],
                "description": result.metadata.get("description") or "Baseline forecast",
                "color": colors[name],
                "values": np.asarray(prices, dtype=np.float64),
            }
        )
    moe = regime_ensemble_forecast(context)
    out.append(
        {
            "id": "regime_ensemble",
            "label": "Regime Ensemble",
            "description": f"Regime-aware MoE baseline ({moe.metadata.get('regime', 'unknown')})",
            "color": "#7ee787",
            "values": np.asarray(context.current_price * np.exp(moe.cum_log_path), dtype=np.float64),
        }
    )
    return out


def build_forecast(
    *,
    symbol: str,
    interval: str,
    horizon: int | None = None,
    models: str | None = None,
    include_explanation: bool = False,
    include_scenarios: bool = True,
    settings: Settings | None = None,
) -> ForecastBundle:
    del models, include_explanation
    settings = settings or get_settings()
    try:
        market = load_market_data_window(symbol, interval, settings=settings)
    except MarketDataUnavailable:
        raise

    resolved_interval = market.timeframe.normalized
    model_horizon = INTERVAL_TO_HORIZON.get(resolved_interval, INTERVAL_TO_HORIZON[FALLBACK_INTERVAL])
    output_horizon = model_horizon if horizon is None or horizon <= 0 else min(int(horizon), model_horizon)
    close = np.asarray([candle.close for candle in market.candles], dtype=np.float64)
    if len(close) == 0:
        raise ForecastUnavailable("Forecast requires at least one close price.")

    try:
        comparison_models, model_info = forecast_model_comparison(
            close=close,
            interval=resolved_interval,
            horizon=model_horizon,
            z_value=CONFIDENCE_Z,
            return_clip=INTERVAL_TO_RETURN_CLIP.get(resolved_interval, 0.05),
            max_log_band=INTERVAL_TO_MAX_LOG_BAND.get(resolved_interval, 0.25),
        )
    except PretrainedModelNotFoundError:
        raise
    except Exception as exc:
        raise ForecastUnavailable(str(exc)) from exc

    if not comparison_models:
        raise ForecastUnavailable("No forecast model output was produced.")
    comparison_models = comparison_models + _baseline_comparison_models(close, resolved_interval, model_horizon)

    primary = next((item for item in comparison_models if item.get("id") == "motif"), comparison_models[0])
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

    warnings = list(market.data_status.warnings)
    warnings.append("Quantile bands are residual-volatility adapters and are not validated coverage intervals yet.")
    status_value = str(market.data_status.status)
    if status_value in {DataStatusKind.mock.value, DataStatusKind.fallback.value, DataStatusKind.stale.value}:
        warnings.append(f"Data status is {market.data_status.status}; confidence is reduced.")

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
        warnings.append(cross_asset_context["warning"])
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
    )
    return ForecastBundle(
        response=response,
        market_data=market,
        forecast_models=forecast_models,
        metrics=_model_metrics(model_info, market.symbol.provider_symbol),
        model_info=model_info,
        horizon=output_horizon,
    )


def chart_payload_from_forecast(bundle: ForecastBundle) -> dict[str, Any]:
    response = bundle.response
    candles = [candle.model_dump() for candle in bundle.market_data.candles]
    anchor = {"time": bundle.market_data.candles[-1].time, "value": response.current_price}
    predicted = [anchor] + [{"time": point.time, "value": point.p50} for point in response.forecast]
    predicted_lower = [anchor] + [{"time": point.time, "value": point.p10} for point in response.forecast]
    predicted_upper = [anchor] + [{"time": point.time, "value": point.p90} for point in response.forecast]

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
        "warning": " ".join(response.warnings) if response.warnings else None,
        "warnings": response.warnings,
        "forecast_horizon": bundle.horizon,
        "confidence_level": None,
        "band_label": "uncalibrated residual-volatility quantile adapter",
        "model_trained_at": bundle.model_info.get("trained_at"),
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
