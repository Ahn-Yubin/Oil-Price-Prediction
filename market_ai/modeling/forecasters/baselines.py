from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


QUANTILE_Z = {
    "p05": -1.6448536269514729,
    "p10": -1.2815515655446004,
    "p25": -0.6744897501960817,
    "p50": 0.0,
    "p75": 0.6744897501960817,
    "p90": 1.2815515655446004,
    "p95": 1.6448536269514729,
}
QUANTILE_ORDER = tuple(QUANTILE_Z.keys())


@dataclass(frozen=True)
class ForecastContext:
    close: np.ndarray
    interval: str
    horizon: int
    current_price: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ForecastResult:
    name: str
    cum_log_path: np.ndarray
    quantiles: dict[str, np.ndarray]
    prob_up: np.ndarray
    expected_volatility: np.ndarray
    confidence: np.ndarray
    metadata: dict = field(default_factory=dict)

    def price_quantiles(self, base_price: float | None = None) -> dict[str, np.ndarray]:
        base = float(base_price if base_price is not None else self.metadata.get("current_price", 1.0))
        return {key: base * np.exp(path) for key, path in self.quantiles.items()}


def _clean_close(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) == 0:
        raise ValueError("Forecast requires at least one positive close")
    return arr


def _returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(_clean_close(close)))


def _safe_std(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    value = float(np.std(values))
    return value if np.isfinite(value) else 0.0


def _recent_volatility(returns: np.ndarray, interval: str) -> float:
    windows = {"1d": 20, "1h": 48, "30m": 96, "15m": 120}
    lookback = min(len(returns), windows.get(interval, 32))
    if lookback <= 1:
        return 1e-5
    return max(_safe_std(returns[-lookback:]), 1e-5)


def _confidence_from_history(returns: np.ndarray, interval: str) -> float:
    if len(returns) < 8:
        return 0.35
    vol = _recent_volatility(returns, interval)
    long_vol = max(_safe_std(returns[-min(len(returns), 120) :]), 1e-5)
    penalty = min(max(vol / long_vol - 1.0, 0.0) * 0.18, 0.25)
    sample_bonus = min(len(returns) / 260.0, 1.0) * 0.18
    return float(np.clip(0.45 + sample_bonus - penalty, 0.1, 0.85))


def _ensure_monotonic(quantiles: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    stacked = np.vstack([np.asarray(quantiles[key], dtype=np.float64) for key in QUANTILE_ORDER])
    stacked = np.sort(stacked, axis=0)
    return {key: stacked[idx] for idx, key in enumerate(QUANTILE_ORDER)}


def _result_from_path(name: str, context: ForecastContext, cum_path: np.ndarray, metadata: dict | None = None) -> ForecastResult:
    close = _clean_close(context.close)
    returns = _returns(close)
    horizon = int(context.horizon)
    cum_path = np.asarray(cum_path, dtype=np.float64)[:horizon]
    if len(cum_path) < horizon:
        last = float(cum_path[-1]) if len(cum_path) else 0.0
        cum_path = np.pad(cum_path, (0, horizon - len(cum_path)), constant_values=last)
    step_vol = _recent_volatility(returns, context.interval)
    steps = np.sqrt(np.arange(1, horizon + 1, dtype=np.float64))
    cum_vol = np.maximum(step_vol * steps, 1e-6)
    quantiles = _ensure_monotonic({key: cum_path + z_value * cum_vol for key, z_value in QUANTILE_Z.items()})
    score = np.divide(cum_path, cum_vol, out=np.zeros_like(cum_path), where=cum_vol > 0)
    prob_up = np.clip(0.5 + 0.25 * np.tanh(score), 0.0, 1.0)
    confidence = np.repeat(_confidence_from_history(returns, context.interval), horizon)
    current_price = float(context.current_price if context.current_price is not None else close[-1])
    return ForecastResult(
        name=name,
        cum_log_path=cum_path,
        quantiles=quantiles,
        prob_up=prob_up,
        expected_volatility=cum_vol,
        confidence=confidence,
        metadata={"current_price": current_price, **(metadata or {})},
    )


def forecast_random_walk(context: ForecastContext) -> ForecastResult:
    return _result_from_path("random_walk", context, np.zeros(context.horizon), {"description": "No-drift random walk baseline"})


def forecast_drift_baseline(context: ForecastContext) -> ForecastResult:
    returns = _returns(context.close)
    lookback = min(len(returns), max(12, context.horizon // 2))
    drift = float(np.mean(returns[-lookback:])) if lookback else 0.0
    drift = float(np.clip(drift, -0.02, 0.02))
    return _result_from_path("drift", context, np.cumsum(np.repeat(drift, context.horizon)), {"drift": drift})


def forecast_seasonal_naive(context: ForecastContext) -> ForecastResult:
    returns = _returns(context.close)
    season = {"1d": 5, "1h": 24, "30m": 48, "15m": 96}.get(context.interval, 5)
    if len(returns) < season:
        return forecast_random_walk(context)
    pattern = returns[-season:]
    repeated = np.resize(pattern, context.horizon)
    path = np.cumsum(np.clip(repeated, -0.04, 0.04))
    return _result_from_path("seasonal_naive", context, path, {"season_length": season})


def forecast_volatility_scaled_naive(context: ForecastContext) -> ForecastResult:
    returns = _returns(context.close)
    if len(returns) < 4:
        return forecast_random_walk(context)
    vol = _recent_volatility(returns, context.interval)
    momentum = float(np.sum(returns[-min(len(returns), 10) :]))
    direction = float(np.tanh(momentum / max(vol * np.sqrt(min(len(returns), 10)), 1e-6)))
    steps = np.sqrt(np.arange(1, context.horizon + 1, dtype=np.float64))
    path = direction * vol * 0.35 * steps
    return _result_from_path("volatility_scaled_naive", context, path, {"direction_score": direction})


def forecast_simple_moving_average_path(context: ForecastContext) -> ForecastResult:
    close = _clean_close(context.close)
    lookback = min(len(close), {"1d": 20, "1h": 48, "30m": 96, "15m": 120}.get(context.interval, 32))
    if lookback < 2:
        return forecast_random_walk(context)
    base = float(close[-1])
    sma = float(np.mean(close[-lookback:]))
    target_log = float(np.clip(np.log(max(sma, 1e-8) / base), -0.2, 0.2))
    progress = 1.0 - np.exp(-np.arange(1, context.horizon + 1, dtype=np.float64) / max(context.horizon / 3.0, 1.0))
    return _result_from_path("simple_moving_average_path", context, target_log * progress, {"lookback": lookback})


BASELINE_FORECASTERS: dict[str, Callable[[ForecastContext], ForecastResult]] = {
    "random_walk": forecast_random_walk,
    "drift": forecast_drift_baseline,
    "seasonal_naive": forecast_seasonal_naive,
    "volatility_scaled_naive": forecast_volatility_scaled_naive,
    "simple_moving_average_path": forecast_simple_moving_average_path,
}
