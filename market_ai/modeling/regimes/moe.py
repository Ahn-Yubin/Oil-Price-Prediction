from __future__ import annotations

import numpy as np

from market_ai.modeling.forecasters.baselines import ForecastContext, ForecastResult, _ensure_monotonic, forecast_drift_baseline, forecast_random_walk, forecast_seasonal_naive, forecast_volatility_scaled_naive
from market_ai.modeling.regimes.detector import detect_regime


def regime_ensemble_forecast(context: ForecastContext) -> ForecastResult:
    regime = detect_regime(context.close)
    experts = {
        "trend": forecast_drift_baseline(context),
        "mean_reversion": forecast_seasonal_naive(context),
        "high_volatility": forecast_volatility_scaled_naive(context),
        "fallback": forecast_random_walk(context),
    }
    weights = {
        "trend": regime.probabilities.trend_up + regime.probabilities.trend_down,
        "mean_reversion": regime.probabilities.range,
        "high_volatility": regime.probabilities.high_volatility + regime.probabilities.event_driven,
        "fallback": 0.10,
    }
    total = max(sum(weights.values()), 1e-8)
    weights = {key: val / total for key, val in weights.items()}
    path = sum(weights[key] * experts[key].cum_log_path for key in weights)
    quantiles = _ensure_monotonic(
        {
            q: sum(weights[key] * experts[key].quantiles[q] for key in weights)
            for q in ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]
        }
    )
    if regime.probabilities.high_volatility > 0.28 or regime.label in {"shock", "high_volatility"}:
        center = quantiles["p50"]
        for q in ["p05", "p10", "p25"]:
            quantiles[q] = center - (center - quantiles[q]) * 1.25
        for q in ["p75", "p90", "p95"]:
            quantiles[q] = center + (quantiles[q] - center) * 1.25
        quantiles = _ensure_monotonic(quantiles)
    prob_up = np.clip(sum(weights[key] * experts[key].prob_up for key in weights), 0.0, 1.0)
    expected_vol = sum(weights[key] * experts[key].expected_volatility for key in weights)
    confidence = np.repeat(regime.confidence, context.horizon)
    return ForecastResult(
        name="regime_ensemble",
        cum_log_path=np.asarray(path, dtype=np.float64),
        quantiles=quantiles,
        prob_up=prob_up,
        expected_volatility=np.asarray(expected_vol, dtype=np.float64),
        confidence=confidence,
        metadata={"regime": regime.label, "weights": weights, "current_price": context.current_price or float(context.close[-1])},
    )
