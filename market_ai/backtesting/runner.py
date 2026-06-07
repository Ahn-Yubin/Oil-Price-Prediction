#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from market_ai.modeling.forecasters.deep_fusion import forecast_with_deep_model
from market_ai.config import PROJECT_DIR, get_settings
from market_ai.constants import (
    CONFIDENCE_Z,
    INTERVAL_TO_HORIZON,
    INTERVAL_TO_MAX_LOG_BAND,
    INTERVAL_TO_RETURN_CLIP,
    select_model_horizon,
)
from market_ai.data.market_panel import load_market_panel
from market_ai.modeling.deep.availability import deep_artifact_availability
from market_ai.modeling.registry import metadata_for_artifact
from market_ai.modeling.model_catalog import REMOVED_LEGACY_MODELS


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
ONLINE_RESIDUAL_WINDOW = 8
ONLINE_RESIDUAL_GAIN = 2.0
ONLINE_RESIDUAL_MAX_ABS_LOG = 0.18


@dataclass(frozen=True)
class ForecastResult:
    name: str
    cum_log_path: np.ndarray


def _clean_close(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return arr[np.isfinite(arr) & (arr > 0.0)]


def _normalize_yfinance_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.reset_index()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in frame.columns]
    date_col = "Date" if "Date" in frame.columns else "Datetime" if "Datetime" in frame.columns else frame.columns[0]
    frame = frame.rename(
        columns={
            date_col: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    keep = ["date", "open", "high", "low", "close", "volume"]
    out = frame[[col for col in keep if col in frame.columns]].copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)
    return out[out["close"] > 0.0].reset_index(drop=True)


def _processed_market_panel_path(interval: str) -> Path | None:
    root = PROJECT_DIR / "data" / "processed" / "market_panel" / interval
    for name in ("panel.parquet", "panel.csv"):
        path = root / name
        if path.exists():
            return path
    return None


def _normalize_processed_panel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "timestamp" in out.columns:
        out["date"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
    else:
        raise ValueError("processed market panel requires timestamp/date column")
    keep = ["date", "open", "high", "low", "close", "volume"]
    out = out[[col for col in keep if col in out.columns]].copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    return out[out["close"] > 0.0].reset_index(drop=True)


def load_processed_candles(symbol: str, interval: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    path = _processed_market_panel_path(interval)
    if path is None:
        raise FileNotFoundError(f"No processed market panel found for {interval}")
    panel = load_market_panel(path)
    frame = panel[panel["symbol"].astype(str) == symbol].copy()
    if frame.empty:
        raise RuntimeError(f"Processed market panel {path.relative_to(PROJECT_DIR)} has no rows for {symbol}")
    frame = _normalize_processed_panel_frame(frame)
    if start:
        start_ts = pd.Timestamp(start)
        start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
        frame = frame[frame["date"] >= start_ts]
    if end:
        end_ts = pd.Timestamp(end)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        frame = frame[frame["date"] < end_ts]
    if frame.empty:
        raise RuntimeError(f"Processed market panel has no rows for {symbol} {interval} in requested window")
    frame.attrs["source"] = "processed_market_panel"
    frame.attrs["source_path"] = str(path.relative_to(PROJECT_DIR))
    return frame.reset_index(drop=True)


def download_candles(symbol: str, interval: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    try:
        frame = load_processed_candles(symbol, interval, start=start, end=end)
        if len(frame) > INTERVAL_TO_HORIZON[interval] + 60:
            return frame
    except Exception:
        pass
    if start or end:
        data = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        if not data.empty:
            frame = _normalize_yfinance_frame(data)
            if len(frame) > INTERVAL_TO_HORIZON[interval] + 60:
                return frame
    for period in BACKTEST_PERIOD_CANDIDATES.get(interval, ["2y"]):
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if data.empty:
            continue
        frame = _normalize_yfinance_frame(data)
        if len(frame) > INTERVAL_TO_HORIZON[interval] + 180:
            return frame
    raise RuntimeError(f"No usable data for {symbol} {interval}")


def download_close(symbol: str, interval: str, start: str | None = None, end: str | None = None) -> np.ndarray:
    return _clean_close(download_candles(symbol, interval, start=start, end=end)["close"])


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


def forecast_motif(close: np.ndarray, interval: str, horizon: int, k: int = 12) -> ForecastResult:
    returns = _returns(close)
    window = {"1d": 64, "1h": 96, "30m": 120, "15m": 144}.get(interval, 96)
    if len(returns) < window + horizon + 10:
        return forecast_random_walk(close, interval, horizon)

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
        return forecast_random_walk(close, interval, horizon)

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


def forecast_oil_context_fusion(close: np.ndarray, interval: str, horizon: int) -> ForecastResult:
    model = forecast_with_deep_model(model_name="oil_context_fusion", close=close, interval=interval, horizon=horizon)
    base = float(close[-1])
    return ForecastResult("oil_context_fusion", np.log(np.asarray(model["values"], dtype=np.float64) / base))


DEEP_MODEL_NAMES = {"oil_context_fusion"}


FORECASTERS = {
    "random_walk": forecast_random_walk,
    "drift": forecast_drift,
    "seasonal_naive": forecast_seasonal_naive,
    "volatility_scaled_naive": forecast_volatility_scaled_naive,
    "flat": forecast_flat,
    "simple_moving_average_path": forecast_simple_moving_average_path,
    "motif": forecast_motif,
    "pattern_mlp": forecast_mlp,
    "oil_context_fusion": forecast_oil_context_fusion,
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


def _price_shape_metrics(pred_price: np.ndarray, actual_price: np.ndarray) -> dict[str, float]:
    pred_price = np.asarray(pred_price, dtype=np.float64)
    actual_price = np.asarray(actual_price, dtype=np.float64)
    mask = np.isfinite(pred_price) & np.isfinite(actual_price) & (actual_price > 0.0)
    pred = pred_price[mask]
    actual = actual_price[mask]
    if len(pred) == 0:
        return {
            "mape": float("nan"),
            "final_ape_pct": float("nan"),
            "step_directional_accuracy": float("nan"),
            "pred_turns": float("nan"),
            "actual_turns": float("nan"),
            "turn_error": float("nan"),
            "range_ratio": float("nan"),
            "shape_score": float("nan"),
        }
    ape = np.abs(pred - actual) / np.maximum(np.abs(actual), 1e-8)
    pred_turns = _turn_count(pred)
    actual_turns = _turn_count(actual)
    actual_range = float(np.max(actual) - np.min(actual))
    pred_range = float(np.max(pred) - np.min(pred))
    range_ratio = pred_range / max(actual_range, 1e-8)
    turn_error = abs(pred_turns - actual_turns) / max(actual_turns, 1)
    step_direction = (
        float(np.mean(np.sign(np.diff(pred)) == np.sign(np.diff(actual))))
        if len(pred) > 1
        else float("nan")
    )
    shape_score = max(
        0.0,
        100.0
        - 45.0 * min(turn_error, 2.0)
        - 35.0 * min(abs(np.log(max(range_ratio, 1e-6))), 2.0)
        + 20.0 * (step_direction if np.isfinite(step_direction) else 0.0),
    )
    return {
        "mape": float(np.mean(ape) * 100.0),
        "final_ape_pct": float(ape[-1] * 100.0),
        "step_directional_accuracy": step_direction,
        "pred_turns": float(pred_turns),
        "actual_turns": float(actual_turns),
        "turn_error": float(turn_error),
        "range_ratio": float(range_ratio),
        "shape_score": float(shape_score),
    }


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
        **_price_shape_metrics(pred_price, actual_price),
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


def _online_residual_correction(
    history: list[tuple[int, np.ndarray]],
    origin: int,
    horizon: int,
) -> tuple[np.ndarray, int]:
    eligible = [residual[:horizon] for available_origin, residual in history if available_origin <= origin and len(residual) >= horizon]
    if len(eligible) < ONLINE_RESIDUAL_WINDOW:
        return np.zeros(horizon, dtype=np.float64), 0
    recent = np.vstack(eligible[-ONLINE_RESIDUAL_WINDOW:])
    correction = ONLINE_RESIDUAL_GAIN * np.mean(recent, axis=0)
    correction = np.clip(correction, -ONLINE_RESIDUAL_MAX_ABS_LOG, ONLINE_RESIDUAL_MAX_ABS_LOG)
    return correction.astype(np.float64), int(len(recent))


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


def _candle_time(candles: pd.DataFrame | None, idx: int) -> pd.Timestamp | None:
    if candles is None or candles.empty or idx < 0 or idx >= len(candles):
        return None
    frame = candles.reset_index(drop=True)
    column = "date" if "date" in frame.columns else "timestamp" if "timestamp" in frame.columns else None
    if column is None:
        return None
    value = pd.to_datetime(frame.loc[idx, column], errors="coerce", utc=True)
    return value if pd.notna(value) else None


def _iso_time(value: pd.Timestamp | None) -> str | None:
    return value.isoformat() if value is not None else None


def _deep_artifact_horizon(interval: str, display_horizon: int) -> int:
    artifact_horizon, _ = select_model_horizon(interval, display_horizon)
    return artifact_horizon


def _deep_metadata_lookup(model_names: list[str], interval: str, horizon: int) -> dict[str, dict[str, Any]]:
    settings = get_settings()
    artifact_horizon = _deep_artifact_horizon(interval, horizon)
    out: dict[str, dict[str, Any]] = {}
    for name in model_names:
        if name not in DEEP_MODEL_NAMES:
            continue
        try:
            availability = deep_artifact_availability(settings=settings, model_name=name, interval=interval, horizon=artifact_horizon)
            if not availability.is_available or availability.artifact_path is None:
                continue
            metadata = metadata_for_artifact(availability.artifact_path, metadata_dir=settings.metadata_dir)
        except Exception:
            continue
        out[name] = {
            "artifact_file": metadata.artifact_file,
            "training_cutoff": metadata.training_cutoff,
            "train_start": metadata.train_start,
            "train_end": metadata.train_end,
            "n_train": metadata.n_train,
            "n_val": metadata.n_val,
            "n_test": metadata.n_test,
        }
    return out


def _leakage_audit_status(origin_time: pd.Timestamp | None, metadata: dict[str, Any] | None) -> str:
    if origin_time is None:
        return "origin_time_unavailable"
    if not metadata or not metadata.get("training_cutoff"):
        return "benchmark_or_metadata_unavailable"
    cutoff = pd.to_datetime(metadata.get("training_cutoff"), errors="coerce", utc=True)
    if pd.isna(cutoff):
        return "training_cutoff_unavailable"
    return "post_artifact_cutoff" if origin_time > cutoff else "overlaps_artifact_sample_window"


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
    symbol: str = "UNKNOWN",
    candles: pd.DataFrame | None = None,
    lookback: int,
    horizon: int,
    step: int,
    max_origins: int | None,
    rolling: bool,
    expanding: bool,
    include_regime_breakdown: bool,
) -> dict[str, pd.DataFrame]:
    close = _clean_close(close)
    removed = [name for name in model_names if name in REMOVED_LEGACY_MODELS and name not in FORECASTERS]
    if removed:
        raise ValueError(f"Removed/deprecated model(s) requested: {removed}")
    unknown = [name for name in model_names if name not in FORECASTERS]
    if unknown:
        raise ValueError(f"Unknown model(s) requested: {unknown}")
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
    metadata_lookup = _deep_metadata_lookup(model_names, interval, horizon)
    deep_artifact_horizon = _deep_artifact_horizon(interval, horizon)
    residual_history: dict[str, list[tuple[int, np.ndarray]]] = {name: [] for name in model_names}

    del expanding  # current models are inference-only; expanding is equivalent to using all history up to origin.
    for origin in origins:
        train_start = max(0, origin + 1 - lookback) if rolling else 0
        train_close = close[train_start : origin + 1]
        train_candles = candles.iloc[train_start : origin + 1].copy() if candles is not None else None
        base = float(close[origin])
        actual = close[origin + 1 : origin + 1 + horizon]
        actual_path = np.log(actual / base)
        regime = classify_regime(train_close) if include_regime_breakdown else "all"
        origin_time = _candle_time(candles, origin)
        train_start_time = _candle_time(candles, train_start)
        actual_end_time = _candle_time(candles, origin + horizon)

        for name in model_names:
            fn = FORECASTERS[name]
            metadata = metadata_lookup.get(name)
            audit_status = _leakage_audit_status(origin_time, metadata)
            audit_fields = {
                "origin_time": _iso_time(origin_time),
                "train_window_start": _iso_time(train_start_time),
                "actual_window_end": _iso_time(actual_end_time),
                "artifact_training_cutoff": metadata.get("training_cutoff") if metadata else None,
                "artifact_train_start": metadata.get("train_start") if metadata else None,
                "artifact_train_end": metadata.get("train_end") if metadata else None,
                "leakage_audit_status": audit_status,
            }
            try:
                if name in DEEP_MODEL_NAMES:
                    model = forecast_with_deep_model(
                        model_name=name,
                        close=train_close,
                        interval=interval,
                        horizon=deep_artifact_horizon,
                        symbol=symbol,
                        candles=train_candles,
                    )
                    pred_path = np.asarray(np.log(np.asarray(model["values"], dtype=np.float64) / base)[:horizon], dtype=np.float64)
                elif name == "pattern_mlp":
                    pred_path = np.asarray(fn(train_close, interval, deep_artifact_horizon).cum_log_path[:horizon], dtype=np.float64)
                else:
                    pred_path = np.asarray(fn(train_close, interval, horizon).cum_log_path[:horizon], dtype=np.float64)
            except Exception as exc:
                summary_rows.append({"model": name, "origin": origin, "regime": regime, **audit_fields, "error": str(exc)})
                continue
            if len(pred_path) < horizon:
                pred_path = np.pad(pred_path, (0, horizon - len(pred_path)), constant_values=float(pred_path[-1]) if len(pred_path) else 0.0)
            online_calibration_samples = 0
            if name in DEEP_MODEL_NAMES and audit_status == "post_artifact_cutoff":
                correction, online_calibration_samples = _online_residual_correction(residual_history.get(name, []), origin, horizon)
                pred_path = pred_path + correction

            pred_price = base * np.exp(pred_path)
            metric = point_metrics(pred_price, actual, train_close)
            calibration_fields = {
                "online_residual_calibration": bool(online_calibration_samples),
                "online_residual_samples": online_calibration_samples,
            }
            summary_rows.append({"model": name, "origin": origin, "regime": regime, **audit_fields, **calibration_fields, **metric})

            quantile_paths = _quantile_paths_from_point(pred_path, train_close)
            probabilistic_rows.append(
                {"model": name, "origin": origin, "regime": regime, **audit_fields, **calibration_fields, **probabilistic_metrics(quantile_paths, actual_path)}
            )

            for h in _eval_horizons(horizon):
                h_pred_price = pred_price[:h]
                h_actual = actual[:h]
                h_metrics = point_metrics(h_pred_price, h_actual, train_close)
                horizon_rows.append({"model": name, "origin": origin, "horizon": h, "regime": regime, **audit_fields, **calibration_fields, **h_metrics})

            for step_idx, (p_log, a_log) in enumerate(zip(pred_path, actual_path), start=1):
                row = {
                    "model": name,
                    "origin": origin,
                    "step": step_idx,
                    "regime": regime,
                    **audit_fields,
                    "pred_log_return": float(p_log),
                    "actual_log_return": float(a_log),
                    "pred_price": float(base * np.exp(p_log)),
                    "actual_price": float(base * np.exp(a_log)),
                    **calibration_fields,
                }
                for key, path in quantile_paths.items():
                    row[f"{key}_log_return"] = float(path[step_idx - 1])
                    row[f"{key}_price"] = float(base * np.exp(path[step_idx - 1]))
                details_rows.append(row)
                if origin == origins[-1]:
                    sample_rows.append(row)
            if len(actual_path) >= horizon and (name not in DEEP_MODEL_NAMES or audit_status == "post_artifact_cutoff"):
                residual_history.setdefault(name, []).append((origin + horizon, np.asarray(actual_path[:horizon] - pred_path[:horizon], dtype=np.float64)))

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
            mape=("mape", "mean"),
            smape=("smape", "mean"),
            mase=("mase", "mean"),
            median_absolute_error=("median_absolute_error", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
            step_directional_accuracy=("step_directional_accuracy", "mean"),
            final_ape_pct=("final_ape_pct", "mean"),
            pred_turns=("pred_turns", "mean"),
            actual_turns=("actual_turns", "mean"),
            turn_error=("turn_error", "mean"),
            range_ratio=("range_ratio", "mean"),
            shape_score=("shape_score", "mean"),
        )
        .sort_values(["mape", "shape_score", "rmse"], ascending=[True, False, True])
        .reset_index(drop=True)
        if not ok_summary.empty
        else pd.DataFrame()
    )
    if not summary.empty and "leakage_audit_status" in ok_summary.columns:
        audit = (
            ok_summary.groupby("model", as_index=False)
            .agg(
                origin_start=("origin_time", "min"),
                origin_end=("origin_time", "max"),
                post_cutoff_origins=("leakage_audit_status", lambda values: int((values == "post_artifact_cutoff").sum())),
                overlap_origins=("leakage_audit_status", lambda values: int((values == "overlaps_artifact_sample_window").sum())),
            )
        )
        summary = summary.merge(audit, on="model", how="left")
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
            mape=("mape", "mean"),
            smape=("smape", "mean"),
            mase=("mase", "mean"),
            median_absolute_error=("median_absolute_error", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
            step_directional_accuracy=("step_directional_accuracy", "mean"),
            final_ape_pct=("final_ape_pct", "mean"),
            pred_turns=("pred_turns", "mean"),
            actual_turns=("actual_turns", "mean"),
            turn_error=("turn_error", "mean"),
            range_ratio=("range_ratio", "mean"),
            shape_score=("shape_score", "mean"),
        )
        .sort_values(["horizon", "mape", "shape_score"], ascending=[True, True, False])
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
            mape=("mape", "mean"),
            smape=("smape", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
            step_directional_accuracy=("step_directional_accuracy", "mean"),
            range_ratio=("range_ratio", "mean"),
            shape_score=("shape_score", "mean"),
        )
        .sort_values(["regime", "mape", "shape_score"], ascending=[True, True, False])
        .reset_index(drop=True)
        if include_regime_breakdown and not ok_summary.empty
        else pd.DataFrame()
    )
    availability_rows: list[dict] = []
    for name in model_names:
        model_rows = summary_source[summary_source["model"] == name] if "model" in summary_source.columns else pd.DataFrame()
        errors = model_rows["error"].dropna().astype(str).tolist() if "error" in model_rows.columns else []
        ok_count = len(model_rows) - len(errors)
        availability_rows.append(
            {
                "model": name,
                "status": "available" if ok_count > 0 else "unavailable",
                "origins_ok": ok_count,
                "origins_error": len(errors),
                "last_error": errors[-1] if errors else None,
            }
        )
    leaderboard = summary.copy()
    if not leaderboard.empty and not probabilistic.empty:
        leaderboard = leaderboard.merge(probabilistic[["model", "pinball_loss", "coverage_80", "winkler_80"]], on="model", how="left")
        leaderboard["rank_score"] = (
            leaderboard["mape"].rank(method="min")
            + leaderboard["pinball_loss"].rank(method="min")
            + leaderboard["shape_score"].rank(method="min", ascending=False)
        )
        leaderboard = leaderboard.sort_values(["rank_score", "mape", "shape_score"], ascending=[True, True, False]).reset_index(drop=True)

    return {
        "summary": summary,
        "details": details,
        "horizon_metrics": horizon_metrics,
        "probabilistic_metrics": probabilistic,
        "regime_metrics": regime_metrics,
        "leaderboard": leaderboard,
        "sample": pd.DataFrame(sample_rows),
        "origin_metrics": summary_source,
        "model_availability": pd.DataFrame(availability_rows),
    }


def write_backtest_outputs(outputs: dict[str, pd.DataFrame], output_dir: Path, prefix: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in ["summary", "details", "horizon_metrics", "probabilistic_metrics", "regime_metrics", "leaderboard", "model_availability"]:
        path = output_dir / f"{prefix}_{name}.csv"
        outputs.get(name, pd.DataFrame()).to_csv(path, index=False)
        written.append(path)
        if name == "model_availability":
            latest_path = output_dir / "latest_model_availability.csv"
            outputs.get(name, pd.DataFrame()).to_csv(latest_path, index=False)
            written.append(latest_path)
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
        default="oil_context_fusion,random_walk,drift,motif,pattern_mlp",
        help="Comma-separated model list. Available: " + ",".join(FORECASTERS.keys()),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    intervals = [args.interval] if args.interval else ["1d", "1h"]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    removed = [m for m in model_names if m in REMOVED_LEGACY_MODELS and m not in FORECASTERS]
    if removed:
        raise SystemExit(
            f"Removed/deprecated model(s): {removed}. "
            f"Supported backtest models: {sorted(FORECASTERS)}"
        )
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
        candles = download_candles(args.symbol, interval, start=args.start or None, end=args.end or None)
        close = _clean_close(candles["close"])
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
            symbol=args.symbol,
            candles=candles,
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
        run_meta.setdefault("data_sources", {})[interval] = {
            "source": candles.attrs.get("source", "yfinance"),
            "path": candles.attrs.get("source_path"),
            "rows": int(len(candles)),
            "first_date": _iso_time(_candle_time(candles, 0)),
            "last_date": _iso_time(_candle_time(candles, len(candles) - 1)),
            "horizon": int(horizon),
            "lookback": int(lookback),
            "step": int(step),
            "max_origins": int(max_origins),
        }
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
