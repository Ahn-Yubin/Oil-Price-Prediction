from __future__ import annotations

import numpy as np
import pandas as pd


def to_log_returns(close) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(close), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(np.log(arr))


def to_vol_scaled_returns(returns, volatility: float | None = None, floor: float = 1e-8) -> tuple[np.ndarray, float]:
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    vol = float(np.std(arr)) if volatility is None else float(volatility)
    vol = max(vol, floor)
    return arr / vol, vol


def cumulative_future_returns(returns, horizon: int) -> np.ndarray:
    arr = np.asarray(returns, dtype=np.float64)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    rows: list[np.ndarray] = []
    for idx in range(0, len(arr) - horizon + 1):
        rows.append(np.cumsum(arr[idx : idx + horizon]))
    return np.vstack(rows) if rows else np.empty((0, horizon), dtype=np.float64)


def volatility_scaled_cumulative_returns(returns, horizon: int, vol_window: int = 20) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(returns, dtype=np.float64)
    targets: list[np.ndarray] = []
    scales: list[float] = []
    for idx in range(vol_window, len(arr) - horizon + 1):
        hist = arr[idx - vol_window : idx]
        scale = max(float(np.std(hist)), 1e-8)
        targets.append(np.cumsum(arr[idx : idx + horizon]) / scale)
        scales.append(scale)
    if not targets:
        return np.empty((0, horizon), dtype=np.float64), np.asarray([], dtype=np.float64)
    return np.vstack(targets), np.asarray(scales, dtype=np.float64)


def reconstruct_price_path(current_price: float, predicted_cumulative_log_return) -> np.ndarray:
    path = np.asarray(predicted_cumulative_log_return, dtype=np.float64)
    return float(current_price) * np.exp(path)
