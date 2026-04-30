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
from market_ai.modeling.deep.availability import DeepArtifactAvailability, deep_artifact_availability
from market_ai.modeling.forecasters.deep_fusion import DeepModelUnavailable, forecast_with_deep_model
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError
from market_ai.modeling.forecasters.motif import forecast_model_comparison
from market_ai.modeling.calibration.conformal import apply_calibration_to_points, load_calibration_artifact
from market_ai.modeling.model_catalog import DEEP_MODELS, USER_FACING_MODELS, InvalidModelRequest, resolve_model_selection, split_model_query
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
    return tuple(
        model_name
        for model_name in USER_FACING_MODELS
        if model_name not in DEEP_MODELS or availabilities[model_name].is_available
    )


def _candle_frame_from_market(market: MarketDataWindow) -> pd.DataFrame:
    return pd.DataFrame([candle.model_dump() for candle in market.candles]).assign(
        date=lambda frame: pd.to_datetime(frame["time"], unit="s", utc=True)
    )


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
            )
        except DeepModelUnavailable as exc:
            availability = availabilities.get(model_name)
            artifact_status[model_name] = availability.status if availability is not None else "unavailable"
            should_warn = explicit_request or (availability is not None and availability.is_available)
            if should_warn:
                action = availability.training_command if availability is not None else None
                expected = availability.expected_artifact_file if availability is not None else "expected deep artifact"
                message = (
                    f"{model_name} artifact unavailable for {interval}/horizon {horizon}; "
                    f"falling back to available non-deep models. Expected {expected}. Detail: {exc}"
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
) -> list[ForecastPoint]:
    q_prices = model.get("quantile_prices")
    if isinstance(q_prices, dict) and {"p05", "p10", "p25", "p50", "p75", "p90", "p95"}.issubset(q_prices):
        points: list[ForecastPoint] = []
        prob_up = np.asarray(model.get("prob_up", []), dtype=np.float64)
        expected_vol = np.asarray(model.get("expected_volatility", []), dtype=np.float64)
        confidence = np.asarray(model.get("confidence", []), dtype=np.float64).reshape(-1)
        for idx, dt_value in enumerate(future_times):
            raw = {key: float(np.asarray(q_prices[key], dtype=np.float64)[idx]) for key in ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]}
            ordered = sorted(raw.values())
            p50 = max(ordered[3], 1e-8)
            mid_log = float(np.log(p50 / current_price))
            points.append(
                ForecastPoint(
                    time=int(pd.Timestamp(dt_value).timestamp()),
                    p05=ordered[0],
                    p10=ordered[1],
                    p25=ordered[2],
                    p50=ordered[3],
                    p75=ordered[4],
                    p90=ordered[5],
                    p95=ordered[6],
                    expected_return=mid_log,
                    expected_volatility=_finite_float(expected_vol[idx] if idx < len(expected_vol) else abs(mid_log), 0.0),
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
) -> ForecastBundle:
    del include_explanation
    settings = settings or get_settings()
    requested_models = split_model_query(models)
    try:
        market = load_market_data_window(symbol, interval, settings=settings)
    except MarketDataUnavailable:
        raise

    resolved_interval = market.timeframe.normalized
    model_horizon = INTERVAL_TO_HORIZON.get(resolved_interval, INTERVAL_TO_HORIZON[FALLBACK_INTERVAL])
    output_horizon = model_horizon if horizon is None or horizon <= 0 else min(int(horizon), model_horizon)
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
    warnings: list[str] = []
    warning_objects: list[ForecastWarning] = []
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
        )
        + comparison_models
        + _baseline_comparison_models(close, resolved_interval, model_horizon)
    )
    model_by_id: dict[str, dict[str, Any]] = {}
    for item in comparison_models:
        model_by_id[str(item.get("id"))] = item
    comparison_models = [model_by_id[name] for name in selection.selected if name in model_by_id]
    if not comparison_models:
        fallback_order = ["motif", "pattern_mlp", "random_walk"]
        comparison_models = [model_by_id[name] for name in fallback_order if name in model_by_id]
    if not comparison_models:
        raise ForecastUnavailable("No selected forecast model output was available.")

    primary_priority = (
        ["deep_lstm_tcn_fusion", "motif", "pattern_mlp", "random_walk"]
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
        _add_warning(
            warnings=warnings,
            warning_objects=warning_objects,
            code="quantile_bands_uncalibrated",
            severity="info",
            message="Quantile bands are residual-volatility adapters and are not validated coverage intervals yet.",
            action="Run rolling coverage calibration before labeling bands as validated confidence intervals.",
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
        llm_context_summary={
            "enabled": settings.enable_llm_context,
            "external_calls_enabled": settings.enable_external_llm_calls,
            "role": "context/event encoder only",
        },
        deep_model_info=deep_model_info,
        feature_version=model_info.get("feature_version") or model_info.get("feature_set"),
        artifact_status=artifact_status,
        calibration_status=calibration_status,
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
    actionable_warnings = [item.message for item in response.warning_objects if item.severity in {"warning", "error"}]
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
        "confidence_level": None,
        "band_label": "calibrated conformal quantile band"
        if response.calibration_status.get("calibration_status") == "calibrated"
        else "uncalibrated residual-volatility quantile adapter",
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
