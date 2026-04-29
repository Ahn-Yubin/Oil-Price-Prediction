#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from market_ai.modeling.forecasters.neural_npz import forecast_with_global_model
from market_ai.modeling.forecasters.baselines import ForecastContext, BASELINE_FORECASTERS
from market_ai.config import PROJECT_DIR
from market_ai.constants import (
    CONFIDENCE_Z,
    INTERVAL_TO_HORIZON,
    INTERVAL_TO_MAX_LOG_BAND,
    INTERVAL_TO_RETURN_CLIP,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


OUT_DIR = PROJECT_DIR / "outputs" / "backtests"
PLOT_DIR = OUT_DIR / "plots"
DEFAULT_EVAL_HORIZONS = (1, 3, 5, 10, 20)
BACKTEST_PERIOD_CANDIDATES = {
    "1d": ["10y", "5y", "2y"],
    "1h": ["730d", "365d", "180d"],
    "30m": ["60d", "30d", "14d"],
    "15m": ["60d", "30d", "14d"],
}
WINDOW_BY_INTERVAL = {"1d": 64, "1h": 96, "30m": 120, "15m": 144}
_TORCH_MODEL_CACHE: dict[tuple[str, str, int], object] = {}


@dataclass(frozen=True)
class ForecastResult:
    name: str
    cum_log_path: np.ndarray


def _clean_close(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return arr[np.isfinite(arr) & (arr > 0.0)]


def download_close(symbol: str, interval: str, start: str | None = None, end: str | None = None) -> np.ndarray:
    if start or end:
        data = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        if not data.empty:
            close = data["Close"].iloc[:, 0] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
            arr = _clean_close(close)
            if len(arr) > INTERVAL_TO_HORIZON[interval] + 60:
                return arr
    for period in BACKTEST_PERIOD_CANDIDATES.get(interval, ["2y"]):
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if data.empty:
            continue
        close = data["Close"].iloc[:, 0] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
        arr = _clean_close(close)
        if len(arr) > INTERVAL_TO_HORIZON[interval] + 180:
            return arr
    raise RuntimeError(f"No usable data for {symbol} {interval}")


def _returns(close: np.ndarray) -> np.ndarray:
    close = _clean_close(close)
    return np.diff(np.log(close))


def _safe_std(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    val = float(np.std(values))
    return val if np.isfinite(val) else 0.0


def _recent_vol(returns: np.ndarray, window: int) -> float:
    lookback = min(len(returns), max(12, window // 2))
    return max(_safe_std(returns[-lookback:]), 1e-5)


def _window_signature(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    vol = max(_safe_std(values), 1e-6)
    norm_ret = np.clip((values - float(np.mean(values))) / vol, -6.0, 6.0)
    path = np.cumsum(values)
    path = path - float(path[-1])
    path_std = max(_safe_std(path), vol * np.sqrt(len(values)), 1e-6)
    norm_path = np.clip(path / path_std, -6.0, 6.0)
    return np.concatenate([norm_ret, norm_path])


def _sequence_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    vol = max(_safe_std(values), 1e-6)
    norm_ret = np.clip((values - float(np.mean(values))) / vol, -6.0, 6.0)
    path = np.cumsum(values)
    path = path - float(path[-1])
    path_std = max(_safe_std(path), vol * np.sqrt(len(values)), 1e-6)
    norm_path = np.clip(path / path_std, -6.0, 6.0)

    short = np.zeros_like(values)
    medium = np.zeros_like(values)
    for i in range(len(values)):
        short_tail = values[max(0, i - 4) : i + 1]
        medium_tail = values[max(0, i - 12) : i + 1]
        short[i] = np.mean(short_tail) / vol
        medium[i] = np.mean(medium_tail) / vol
    return np.stack([norm_ret, norm_path, np.clip(short, -6.0, 6.0), np.clip(medium, -6.0, 6.0)], axis=1).astype(np.float32)


def _target_range(returns: np.ndarray, interval: str, horizon: int) -> float:
    lookbacks = {"1d": 45, "1h": 120, "30m": 120, "15m": 160}
    ratios = {"1d": 0.45, "1h": 0.28, "30m": 0.28, "15m": 0.25}
    floors = {"1d": 0.06, "1h": 0.018, "30m": 0.018, "15m": 0.014}
    lookback = min(len(returns), lookbacks.get(interval, horizon))
    recent_path = np.cumsum(returns[-lookback:])
    recent_range = float(np.max(recent_path) - np.min(recent_path)) if len(recent_path) else 0.0
    return max(floors.get(interval, 0.015), recent_range * ratios.get(interval, 0.25))


def calibrate_amplitude(path: np.ndarray, returns: np.ndarray, interval: str, horizon: int) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 4:
        return path
    target_range = _target_range(returns, interval, horizon)
    current_range = float(np.max(path) - np.min(path))
    if current_range >= target_range:
        return path
    trend = np.linspace(0.0, float(path[-1]), len(path))
    residual = path - trend
    residual_range = float(np.max(residual) - np.min(residual))
    if residual_range <= 1e-8:
        return path
    gain = min(target_range / residual_range, 8.0)
    return trend + residual * gain


def forecast_flat(close: np.ndarray, interval: str, horizon: int) -> ForecastResult:
    return ForecastResult("flat", np.zeros(horizon, dtype=np.float64))


def forecast_random_walk(close: np.ndarray, interval: str, horizon: int) -> ForecastResult:
    context = ForecastContext(close=close, interval=interval, horizon=horizon, current_price=float(close[-1]))
    return ForecastResult("random_walk", BASELINE_FORECASTERS["random_walk"](context).cum_log_path)


def forecast_drift(close: np.ndarray, interval: str, horizon: int) -> ForecastResult:
    returns = _returns(close)
    lookback = min(len(returns), max(12, horizon // 2))
    drift = float(np.mean(returns[-lookback:])) if lookback else 0.0
    return ForecastResult("drift", np.cumsum(np.repeat(drift, horizon)))


def forecast_seasonal_naive(close: np.ndarray, interval: str, horizon: int) -> ForecastResult:
    context = ForecastContext(close=close, interval=interval, horizon=horizon, current_price=float(close[-1]))
    return ForecastResult("seasonal_naive", BASELINE_FORECASTERS["seasonal_naive"](context).cum_log_path)


def forecast_volatility_scaled_naive(close: np.ndarray, interval: str, horizon: int) -> ForecastResult:
    context = ForecastContext(close=close, interval=interval, horizon=horizon, current_price=float(close[-1]))
    return ForecastResult("volatility_scaled_naive", BASELINE_FORECASTERS["volatility_scaled_naive"](context).cum_log_path)


def forecast_simple_moving_average_path(close: np.ndarray, interval: str, horizon: int) -> ForecastResult:
    context = ForecastContext(close=close, interval=interval, horizon=horizon, current_price=float(close[-1]))
    return ForecastResult(
        "simple_moving_average_path",
        BASELINE_FORECASTERS["simple_moving_average_path"](context).cum_log_path,
    )


def forecast_cycle(close: np.ndarray, interval: str, horizon: int) -> ForecastResult:
    returns = _returns(close)
    if len(returns) < 24:
        return ForecastResult("cycle", np.zeros(horizon, dtype=np.float64))
    lookback = min(len(returns), max(48, horizon))
    hist = returns[-lookback:]
    vol = _recent_vol(returns, lookback)
    demeaned = hist - float(np.mean(hist))
    spectrum = np.fft.rfft(demeaned)
    if len(spectrum) <= 2:
        return ForecastResult("cycle", np.zeros(horizon, dtype=np.float64))
    idx = int(np.argmax(np.abs(spectrum[1:])) + 1)
    amp = min(float(np.abs(spectrum[idx]) / len(hist)) * 2.0, vol * 1.25)
    phase = float(np.angle(spectrum[idx]))
    t = np.arange(len(hist), len(hist) + horizon, dtype=np.float64)
    step = float(np.mean(hist)) + amp * np.cos(2.0 * np.pi * idx * t / len(hist) + phase)
    path = np.cumsum(step)
    return ForecastResult("cycle", calibrate_amplitude(path, returns, interval, horizon))


def forecast_motif(close: np.ndarray, interval: str, horizon: int, k: int = 12) -> ForecastResult:
    returns = _returns(close)
    window = {"1d": 64, "1h": 96, "30m": 120, "15m": 144}.get(interval, 96)
    if len(returns) < window + horizon + 10:
        return forecast_cycle(close, interval, horizon)

    current = _window_signature(returns[-window:])
    recent_vol = _recent_vol(returns, window)
    candidates: list[tuple[float, np.ndarray]] = []
    last_start = len(returns) - window - horizon + 1
    for start in range(0, max(0, last_start)):
        hist = returns[start : start + window]
        fut = returns[start + window : start + window + horizon]
        if len(fut) != horizon:
            continue
        sig = _window_signature(hist)
        dist = float(np.mean((current - sig) ** 2))
        fut_scale = max(_safe_std(hist[-max(12, window // 2) :]), 1e-5)
        path = np.cumsum(fut) / fut_scale * recent_vol
        candidates.append((dist, path))

    if not candidates:
        return forecast_cycle(close, interval, horizon)

    candidates.sort(key=lambda item: item[0])
    top = candidates[: min(k, len(candidates))]
    dists = np.asarray([d for d, _ in top], dtype=np.float64)
    weights = np.exp(-dists / max(float(np.median(dists)), 1e-6))
    weights = weights / max(float(np.sum(weights)), 1e-8)
    path = np.sum([w * p for w, (_, p) in zip(weights, top)], axis=0)
    path = calibrate_amplitude(path, returns, interval, horizon)
    return ForecastResult("motif", path)


def forecast_mlp(close: np.ndarray, interval: str, horizon: int) -> ForecastResult:
    mean, _low, _high, _info = forecast_with_global_model(
        close=close,
        interval=interval,
        horizon=horizon,
        z_value=CONFIDENCE_Z,
        return_clip=INTERVAL_TO_RETURN_CLIP[interval],
        max_log_band=INTERVAL_TO_MAX_LOG_BAND[interval],
    )
    base = float(close[-1])
    return ForecastResult("pattern_mlp", np.log(np.asarray(mean, dtype=np.float64) / base))


if nn is not None:

    class LSTMPathModel(nn.Module):
        def __init__(self, in_dim: int, hidden: int, horizon: int):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, num_layers=1)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, horizon),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])


    class TCNPathModel(nn.Module):
        def __init__(self, in_dim: int, hidden: int, horizon: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=9, padding=4),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=13, padding=6),
                nn.GELU(),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, horizon),
            )

        def forward(self, x):
            y = self.net(x.transpose(1, 2)).mean(dim=-1)
            return self.head(y)


def _build_torch_dataset(
    returns: np.ndarray,
    interval: str,
    horizon: int,
    max_train_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = WINDOW_BY_INTERVAL.get(interval, 96)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    scales: list[float] = []
    for t in range(window, len(returns) - horizon + 1):
        hist = returns[t - window : t]
        fut = returns[t : t + horizon]
        scale = max(_recent_vol(hist, window), 1e-5)
        xs.append(_sequence_features(hist))
        ys.append(np.clip(np.cumsum(fut) / scale, -12.0, 12.0).astype(np.float32))
        scales.append(float(scale))

    if not xs:
        raise RuntimeError("Not enough samples for torch sequence model")
    X = np.stack(xs).astype(np.float32)
    Y = np.stack(ys).astype(np.float32)
    S = np.asarray(scales, dtype=np.float32)
    if len(X) > max_train_samples:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(X), size=max_train_samples, replace=False))
        X, Y, S = X[idx], Y[idx], S[idx]
    return X, Y, S


def _train_torch_path_model(
    close: np.ndarray,
    interval: str,
    horizon: int,
    kind: str,
    epochs: int,
    max_train_samples: int,
) -> np.ndarray:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is not installed")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)

    returns = _returns(close)
    cache_key = (kind, interval, horizon)
    model = _TORCH_MODEL_CACHE.get(cache_key)
    if model is None:
        X, Y, _S = _build_torch_dataset(returns, interval, horizon, max_train_samples)
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(Y, dtype=torch.float32)
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        hidden = 24 if kind == "lstm" else 32
        model = LSTMPathModel(X.shape[-1], hidden, horizon) if kind == "lstm" else TCNPathModel(X.shape[-1], hidden, horizon)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
        weights = torch.tensor(1.0 / np.sqrt(np.arange(1, horizon + 1, dtype=np.float32)), dtype=torch.float32)
        weights = weights / weights.mean()

        model.train()
        for _epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = (((pred - yb) ** 2) * weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        _TORCH_MODEL_CACHE[cache_key] = model

    window = WINDOW_BY_INTERVAL.get(interval, 96)
    features = _sequence_features(returns[-window:])
    recent_scale = max(_recent_vol(returns, window), 1e-5)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(features[None, :, :], dtype=torch.float32)).cpu().numpy()[0]
    path = pred_scaled.astype(np.float64) * recent_scale
    path = calibrate_amplitude(path, returns, interval, horizon)
    return path


def forecast_lstm(close: np.ndarray, interval: str, horizon: int) -> ForecastResult:
    epochs = {"1d": 8, "1h": 6, "30m": 6, "15m": 5}.get(interval, 6)
    max_samples = {"1d": 1000, "1h": 1200, "30m": 1000, "15m": 1000}.get(interval, 1000)
    return ForecastResult("lstm", _train_torch_path_model(close, interval, horizon, "lstm", epochs, max_samples))


def forecast_tcn(close: np.ndarray, interval: str, horizon: int) -> ForecastResult:
    epochs = {"1d": 8, "1h": 6, "30m": 6, "15m": 5}.get(interval, 6)
    max_samples = {"1d": 1000, "1h": 1200, "30m": 1000, "15m": 1000}.get(interval, 1000)
    return ForecastResult("tcn", _train_torch_path_model(close, interval, horizon, "tcn", epochs, max_samples))


def forecast_ensemble(close: np.ndarray, interval: str, horizon: int) -> ForecastResult:
    motif = forecast_motif(close, interval, horizon).cum_log_path
    cycle = forecast_cycle(close, interval, horizon).cum_log_path
    mlp = forecast_mlp(close, interval, horizon).cum_log_path
    path = 0.60 * motif + 0.20 * cycle + 0.20 * mlp
    path = calibrate_amplitude(path, _returns(close), interval, horizon)
    return ForecastResult("ensemble", path)


FORECASTERS = {
    "random_walk": forecast_random_walk,
    "flat": forecast_flat,
    "drift": forecast_drift,
    "seasonal_naive": forecast_seasonal_naive,
    "volatility_scaled_naive": forecast_volatility_scaled_naive,
    "simple_moving_average_path": forecast_simple_moving_average_path,
    "cycle": forecast_cycle,
    "motif": forecast_motif,
    "pattern_mlp": forecast_mlp,
    "lstm": forecast_lstm,
    "tcn": forecast_tcn,
    "ensemble": forecast_ensemble,
}


def _turn_count(values: np.ndarray) -> int:
    diff = np.diff(np.asarray(values, dtype=np.float64))
    diff = diff[diff != 0.0]
    if len(diff) < 2:
        return 0
    return int(np.sum(np.diff(np.sign(diff)) != 0))


def _metrics(pred_path: np.ndarray, actual_path: np.ndarray) -> dict:
    pred_price = np.exp(pred_path)
    actual_price = np.exp(actual_path)
    ape = np.abs(pred_price - actual_price) / np.maximum(np.abs(actual_price), 1e-8)
    pred_turns = _turn_count(pred_price)
    actual_turns = _turn_count(actual_price)
    actual_range = float(np.max(actual_price) - np.min(actual_price))
    pred_range = float(np.max(pred_price) - np.min(pred_price))
    range_ratio = pred_range / max(actual_range, 1e-8)
    turn_error = abs(pred_turns - actual_turns) / max(actual_turns, 1)
    direction = float(np.mean(np.sign(np.diff(pred_price)) == np.sign(np.diff(actual_price)))) if len(pred_price) > 1 else 0.0
    return {
        "mae_pct": float(np.mean(ape) * 100.0),
        "rmse_pct": float(np.sqrt(np.mean((pred_price - actual_price) ** 2)) * 100.0),
        "final_ape_pct": float(ape[-1] * 100.0),
        "direction_acc": direction,
        "pred_turns": float(pred_turns),
        "actual_turns": float(actual_turns),
        "turn_error": float(turn_error),
        "range_ratio": float(range_ratio),
        "shape_score": float(max(0.0, 100.0 - 45.0 * min(turn_error, 2.0) - 35.0 * min(abs(np.log(max(range_ratio, 1e-6))), 2.0) + 20.0 * direction)),
    }


def rolling_origin_indices(
    n_rows: int,
    horizon: int,
    *,
    lookback: int,
    step: int,
    max_origins: int | None,
) -> list[int]:
    min_origin = max(lookback - 1, 1)
    max_origin = n_rows - horizon - 1
    if max_origin <= min_origin:
        raise RuntimeError(f"Not enough rows for rolling backtest: rows={n_rows}, lookback={lookback}, horizon={horizon}")
    origins = list(range(min_origin, max_origin + 1, max(1, step)))
    if max_origins is not None and max_origins > 0:
        origins = origins[-max_origins:]
    return origins


def _mase_denominator(train_close: np.ndarray) -> float:
    if len(train_close) < 2:
        return 1.0
    diffs = np.abs(np.diff(train_close.astype(np.float64)))
    denom = float(np.mean(diffs)) if len(diffs) else 1.0
    return max(denom, 1e-8)


def point_metrics(pred_price: np.ndarray, actual_price: np.ndarray, train_close: np.ndarray) -> dict[str, float]:
    pred_price = np.asarray(pred_price, dtype=np.float64)
    actual_price = np.asarray(actual_price, dtype=np.float64)
    err = pred_price - actual_price
    abs_err = np.abs(err)
    denom = np.maximum((np.abs(pred_price) + np.abs(actual_price)) / 2.0, 1e-8)
    direction = (
        float(np.mean(np.sign(np.diff(pred_price)) == np.sign(np.diff(actual_price))))
        if len(pred_price) > 1
        else float(np.sign(pred_price[-1] - train_close[-1]) == np.sign(actual_price[-1] - train_close[-1]))
    )
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "smape": float(np.mean(abs_err / denom) * 100.0),
        "mase": float(np.mean(abs_err) / _mase_denominator(train_close)),
        "median_absolute_error": float(np.median(abs_err)),
        "directional_accuracy": direction,
    }


def _quantile_paths_from_point(pred_path: np.ndarray, train_close: np.ndarray) -> dict[str, np.ndarray]:
    returns = _returns(train_close)
    vol = _recent_vol(returns, max(20, min(len(returns), 80))) if len(returns) else 1e-5
    steps = np.sqrt(np.arange(1, len(pred_path) + 1, dtype=np.float64))
    z = {
        "p05": -1.6448536269514729,
        "p10": -1.2815515655446004,
        "p25": -0.6744897501960817,
        "p50": 0.0,
        "p75": 0.6744897501960817,
        "p90": 1.2815515655446004,
        "p95": 1.6448536269514729,
    }
    paths = {key: np.asarray(pred_path, dtype=np.float64) + value * vol * steps for key, value in z.items()}
    stacked = np.sort(np.vstack([paths[key] for key in z]), axis=0)
    return {key: stacked[idx] for idx, key in enumerate(z)}


def _pinball(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def probabilistic_metrics(quantile_paths: dict[str, np.ndarray], actual_path: np.ndarray) -> dict[str, float]:
    actual_path = np.asarray(actual_path, dtype=np.float64)
    q_levels = {"p05": 0.05, "p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95}
    pinball = np.mean([_pinball(actual_path, np.asarray(quantile_paths[key]), level) for key, level in q_levels.items()])
    p10 = np.asarray(quantile_paths["p10"], dtype=np.float64)
    p90 = np.asarray(quantile_paths["p90"], dtype=np.float64)
    p05 = np.asarray(quantile_paths["p05"], dtype=np.float64)
    p95 = np.asarray(quantile_paths["p95"], dtype=np.float64)
    coverage_80 = float(np.mean((actual_path >= p10) & (actual_path <= p90)))
    coverage_90 = float(np.mean((actual_path >= p05) & (actual_path <= p95)))
    width_80 = float(np.mean(p90 - p10))
    width_90 = float(np.mean(p95 - p05))
    alpha = 0.2
    winkler = (p90 - p10) + (2.0 / alpha) * (p10 - actual_path) * (actual_path < p10) + (2.0 / alpha) * (actual_path - p90) * (actual_path > p90)
    return {
        "pinball_loss": float(pinball),
        "coverage_80": coverage_80,
        "coverage_90": coverage_90,
        "average_interval_width_80": width_80,
        "average_interval_width_90": width_90,
        "winkler_80": float(np.mean(winkler)),
    }


def classify_regime(train_close: np.ndarray) -> str:
    returns = _returns(train_close)
    if len(returns) < 20:
        return "range"
    recent = returns[-min(len(returns), 20) :]
    longer = returns[-min(len(returns), 80) :]
    trend = float(np.sum(recent))
    short_vol = max(float(np.std(recent)), 1e-8)
    long_vol = max(float(np.std(longer)), 1e-8)
    if abs(recent[-1]) > short_vol * 3.0:
        return "shock"
    if short_vol > long_vol * 1.35:
        return "high_volatility"
    if short_vol < long_vol * 0.7:
        return "low_volatility"
    if trend > short_vol * 1.5:
        return "trend_up"
    if trend < -short_vol * 1.5:
        return "trend_down"
    return "range"


def _eval_horizons(horizon: int) -> list[int]:
    return [h for h in DEFAULT_EVAL_HORIZONS if h <= horizon] or [horizon]


def backtest(
    close: np.ndarray,
    interval: str,
    samples: int,
    stride: int,
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon = INTERVAL_TO_HORIZON[interval]
    min_train = {"1d": 260, "1h": 420, "30m": 420, "15m": 560}.get(interval, 420)
    max_origin = len(close) - horizon - 1
    if max_origin <= min_train:
        raise RuntimeError(f"Not enough rows for backtest: rows={len(close)}, interval={interval}")
    origins = list(range(max_origin, min_train, -stride))[:samples]
    origins = sorted(origins)

    rows = []
    sample_rows = []
    for origin in origins:
        train_close = close[: origin + 1]
        base = float(close[origin])
        actual = close[origin + 1 : origin + 1 + horizon]
        actual_path = np.log(actual / base)

        for name in model_names:
            fn = FORECASTERS[name]
            try:
                pred = fn(train_close, interval, horizon).cum_log_path[:horizon]
            except Exception as exc:
                rows.append({"model": name, "origin": origin, "error": str(exc)})
                continue
            metric = _metrics(pred, actual_path)
            rows.append({"model": name, "origin": origin, **metric})
            if origin == origins[-1]:
                for step, (p, a) in enumerate(zip(pred, actual_path), start=1):
                    sample_rows.append(
                        {
                            "model": name,
                            "step": step,
                            "pred_price": base * float(np.exp(p)),
                            "actual_price": base * float(np.exp(a)),
                        }
                    )

    return pd.DataFrame(rows), pd.DataFrame(sample_rows)


def run_rolling_backtest(
    close: np.ndarray,
    interval: str,
    model_names: list[str],
    *,
    lookback: int,
    horizon: int,
    step: int,
    max_origins: int | None,
    rolling: bool,
    expanding: bool,
    include_regime_breakdown: bool,
) -> dict[str, pd.DataFrame]:
    close = _clean_close(close)
    origins = rolling_origin_indices(
        len(close),
        horizon,
        lookback=lookback,
        step=step,
        max_origins=max_origins,
    )
    details_rows: list[dict] = []
    summary_rows: list[dict] = []
    horizon_rows: list[dict] = []
    probabilistic_rows: list[dict] = []
    sample_rows: list[dict] = []

    del expanding  # current models are inference-only; expanding is equivalent to using all history up to origin.
    for origin in origins:
        train_start = max(0, origin + 1 - lookback) if rolling else 0
        train_close = close[train_start : origin + 1]
        base = float(close[origin])
        actual = close[origin + 1 : origin + 1 + horizon]
        actual_path = np.log(actual / base)
        regime = classify_regime(train_close) if include_regime_breakdown else "all"

        for name in model_names:
            fn = FORECASTERS[name]
            try:
                pred_path = np.asarray(fn(train_close, interval, horizon).cum_log_path[:horizon], dtype=np.float64)
            except Exception as exc:
                summary_rows.append({"model": name, "origin": origin, "regime": regime, "error": str(exc)})
                continue
            if len(pred_path) < horizon:
                pred_path = np.pad(pred_path, (0, horizon - len(pred_path)), constant_values=float(pred_path[-1]) if len(pred_path) else 0.0)

            pred_price = base * np.exp(pred_path)
            metric = point_metrics(pred_price, actual, train_close)
            summary_rows.append({"model": name, "origin": origin, "regime": regime, **metric})

            quantile_paths = _quantile_paths_from_point(pred_path, train_close)
            probabilistic_rows.append({"model": name, "origin": origin, "regime": regime, **probabilistic_metrics(quantile_paths, actual_path)})

            for h in _eval_horizons(horizon):
                h_pred_price = pred_price[:h]
                h_actual = actual[:h]
                h_metrics = point_metrics(h_pred_price, h_actual, train_close)
                horizon_rows.append({"model": name, "origin": origin, "horizon": h, "regime": regime, **h_metrics})

            for step_idx, (p_log, a_log) in enumerate(zip(pred_path, actual_path), start=1):
                row = {
                    "model": name,
                    "origin": origin,
                    "step": step_idx,
                    "regime": regime,
                    "pred_log_return": float(p_log),
                    "actual_log_return": float(a_log),
                    "pred_price": float(base * np.exp(p_log)),
                    "actual_price": float(base * np.exp(a_log)),
                }
                for key, path in quantile_paths.items():
                    row[f"{key}_log_return"] = float(path[step_idx - 1])
                    row[f"{key}_price"] = float(base * np.exp(path[step_idx - 1]))
                details_rows.append(row)
                if origin == origins[-1]:
                    sample_rows.append(row)

    details = pd.DataFrame(details_rows)
    summary_source = pd.DataFrame(summary_rows)
    probabilistic_source = pd.DataFrame(probabilistic_rows)
    horizon_source = pd.DataFrame(horizon_rows)

    ok_summary = summary_source[summary_source["error"].isna()] if "error" in summary_source.columns else summary_source
    summary = (
        ok_summary.groupby("model", as_index=False)
        .agg(
            origins=("origin", "nunique"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            smape=("smape", "mean"),
            mase=("mase", "mean"),
            median_absolute_error=("median_absolute_error", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
        )
        .sort_values(["rmse", "mae"], ascending=[True, True])
        .reset_index(drop=True)
        if not ok_summary.empty
        else pd.DataFrame()
    )
    probabilistic = (
        probabilistic_source.groupby("model", as_index=False)
        .agg(
            pinball_loss=("pinball_loss", "mean"),
            coverage_80=("coverage_80", "mean"),
            coverage_90=("coverage_90", "mean"),
            average_interval_width_80=("average_interval_width_80", "mean"),
            average_interval_width_90=("average_interval_width_90", "mean"),
            winkler_80=("winkler_80", "mean"),
        )
        .sort_values(["pinball_loss", "winkler_80"], ascending=[True, True])
        .reset_index(drop=True)
        if not probabilistic_source.empty
        else pd.DataFrame()
    )
    horizon_metrics = (
        horizon_source.groupby(["model", "horizon"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            smape=("smape", "mean"),
            mase=("mase", "mean"),
            median_absolute_error=("median_absolute_error", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
        )
        .sort_values(["horizon", "rmse", "mae"], ascending=[True, True, True])
        .reset_index(drop=True)
        if not horizon_source.empty
        else pd.DataFrame()
    )
    regime_metrics = (
        ok_summary.groupby(["model", "regime"], as_index=False)
        .agg(
            origins=("origin", "nunique"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            smape=("smape", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
        )
        .sort_values(["regime", "rmse"], ascending=[True, True])
        .reset_index(drop=True)
        if include_regime_breakdown and not ok_summary.empty
        else pd.DataFrame()
    )
    leaderboard = summary.copy()
    if not leaderboard.empty and not probabilistic.empty:
        leaderboard = leaderboard.merge(probabilistic[["model", "pinball_loss", "coverage_80", "winkler_80"]], on="model", how="left")
        leaderboard["rank_score"] = leaderboard["rmse"].rank(method="min") + leaderboard["pinball_loss"].rank(method="min")
        leaderboard = leaderboard.sort_values(["rank_score", "rmse"], ascending=[True, True]).reset_index(drop=True)

    return {
        "summary": summary,
        "details": details,
        "horizon_metrics": horizon_metrics,
        "probabilistic_metrics": probabilistic,
        "regime_metrics": regime_metrics,
        "leaderboard": leaderboard,
        "sample": pd.DataFrame(sample_rows),
        "origin_metrics": summary_source,
    }


def write_backtest_outputs(outputs: dict[str, pd.DataFrame], output_dir: Path, prefix: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in ["summary", "details", "horizon_metrics", "probabilistic_metrics", "regime_metrics", "leaderboard"]:
        path = output_dir / f"{prefix}_{name}.csv"
        outputs.get(name, pd.DataFrame()).to_csv(path, index=False)
        written.append(path)
    sample_path = output_dir / f"{prefix}_last_origin_paths.csv"
    outputs.get("sample", pd.DataFrame()).to_csv(sample_path, index=False)
    written.append(sample_path)
    return written


def plot_sample_paths(sample: pd.DataFrame, symbol: str, interval: str) -> Path | None:
    if sample.empty:
        return None
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / f"{symbol.replace('=', '_')}_{interval}_last_origin_paths.png"
    actual = sample[["step", "actual_price"]].drop_duplicates().sort_values("step")

    plt.figure(figsize=(12, 6), dpi=150)
    plt.plot(actual["step"], actual["actual_price"], color="black", linewidth=2.4, label="Actual")
    for model, group in sample.groupby("model"):
        group = group.sort_values("step")
        plt.plot(group["step"], group["pred_price"], linewidth=1.6, alpha=0.9, label=model)
    plt.title(f"{symbol} {interval} walk-forward last origin: actual vs forecasts")
    plt.xlabel("Forecast step")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.25)
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["error"].isna()] if "error" in results.columns else results
    grouped = ok.groupby("model", as_index=False).agg(
        samples=("origin", "count"),
        mae_pct=("mae_pct", "mean"),
        rmse_pct=("rmse_pct", "mean"),
        final_ape_pct=("final_ape_pct", "mean"),
        direction_acc=("direction_acc", "mean"),
        turn_error=("turn_error", "mean"),
        range_ratio=("range_ratio", "median"),
        shape_score=("shape_score", "mean"),
    )
    return grouped.sort_values(["shape_score", "mae_pct"], ascending=[False, True]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward forecast model backtest")
    p.add_argument("--symbol", default="CL=F")
    p.add_argument("--interval", choices=["1d", "1h", "30m", "15m"], default="")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--lookback", type=int, default=0)
    p.add_argument("--horizon", type=int, default=0)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--max-origins", type=int, default=0)
    p.add_argument("--rolling", action="store_true", help="Use a fixed lookback window for each origin")
    p.add_argument("--expanding", action="store_true", help="Use all data available up to each origin")
    p.add_argument("--output-dir", default=str(OUT_DIR))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--include-regime-breakdown", action="store_true")
    p.add_argument("--samples", type=int, default=0, help="Backward-compatible alias for --max-origins")
    p.add_argument("--stride", type=int, default=0, help="Backward-compatible alias for --step")
    p.add_argument(
        "--models",
        default="motif,lstm,tcn,cycle,pattern_mlp,ensemble,flat,drift,random_walk,seasonal_naive",
        help="Comma-separated model list. Available: " + ",".join(FORECASTERS.keys()),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    intervals = [args.interval] if args.interval else ["1d", "1h", "30m", "15m"]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_names if m not in FORECASTERS]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Available: {sorted(FORECASTERS)}")
    all_summary = []
    all_leaderboards = []
    run_meta = {
        "symbol": args.symbol,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Rolling/expanding origin backtest. Each origin uses only historical close values available at that origin.",
        "models": model_names,
        "start": args.start or None,
        "end": args.end or None,
    }

    for interval in intervals:
        close = download_close(args.symbol, interval, start=args.start or None, end=args.end or None)
        horizon = args.horizon or INTERVAL_TO_HORIZON[interval]
        horizon = max(1, min(horizon, INTERVAL_TO_HORIZON[interval]))
        lookback = args.lookback or {"1d": 260, "1h": 420, "30m": 420, "15m": 560}.get(interval, 420)
        step = args.step or args.stride or max(1, horizon // 4)
        max_origins = args.max_origins or args.samples or 24
        use_rolling = args.rolling or not args.expanding
        outputs = run_rolling_backtest(
            close,
            interval,
            model_names,
            lookback=lookback,
            horizon=horizon,
            step=step,
            max_origins=max_origins,
            rolling=use_rolling,
            expanding=args.expanding,
            include_regime_breakdown=args.include_regime_breakdown,
        )
        prefix = f"{args.symbol.replace('=', '_')}_{interval}"
        write_backtest_outputs(outputs, output_dir, prefix)
        if not args.no_plots:
            plot_path = plot_sample_paths(outputs["sample"], args.symbol, interval)
            if plot_path is not None and plot_path.parent != plot_dir:
                plot_path.replace(plot_dir / plot_path.name)
        summary = outputs["summary"].copy()
        leaderboard = outputs["leaderboard"].copy()
        for frame in [summary, leaderboard]:
            if not frame.empty:
                frame.insert(0, "interval", interval)
                frame.insert(1, "horizon", horizon)
                frame.insert(2, "rows", len(close))
        all_summary.append(summary)
        all_leaderboards.append(leaderboard)

    final = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    final.to_csv(output_dir / f"{args.symbol.replace('=', '_')}_summary.csv", index=False)
    if all_leaderboards:
        pd.concat(all_leaderboards, ignore_index=True).to_csv(output_dir / f"{args.symbol.replace('=', '_')}_leaderboard.csv", index=False)
    (output_dir / f"{args.symbol.replace('=', '_')}_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(final.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
