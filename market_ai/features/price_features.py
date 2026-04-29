from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_SET_VERSION = "price_v1"


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _rolling_autocorr(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    def calc(values: np.ndarray) -> float:
        if len(values) <= lag + 2:
            return 0.0
        a = values[:-lag]
        b = values[lag:]
        a_std = float(np.std(a))
        b_std = float(np.std(b))
        if a_std <= 1e-12 or b_std <= 1e-12:
            return 0.0
        return float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / (a_std * b_std))

    return series.rolling(window, min_periods=max(4, lag + 3)).apply(calc, raw=True)


def build_price_features(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy()
    required = {"open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in frame.columns:
            frame[col] = _safe_numeric(frame[col])
    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    close = frame["close"].clip(lower=1e-12)
    open_ = frame["open"].clip(lower=1e-12)
    high = frame["high"]
    low = frame["low"]
    prev_close = close.shift(1)
    log_return = np.log(close / prev_close)

    out = pd.DataFrame(index=frame.index)
    if "date" in frame.columns:
        out["date"] = frame["date"]
    out["log_return"] = log_return
    out["close_to_open_return"] = np.log(close / open_)
    out["high_low_range"] = (high - low) / close
    out["true_range"] = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1) / close

    for window in [5, 20, 60]:
        out[f"rolling_vol_{window}"] = log_return.rolling(window, min_periods=max(2, window // 2)).std()
        out[f"momentum_{window}"] = log_return.rolling(window, min_periods=max(2, window // 2)).sum()

    rolling_max_20 = close.rolling(20, min_periods=2).max()
    rolling_max_60 = close.rolling(60, min_periods=2).max()
    out["drawdown_20"] = close / rolling_max_20 - 1.0
    out["drawdown_60"] = close / rolling_max_60 - 1.0

    volume = frame["volume"].fillna(0.0)
    volume_mean = volume.rolling(20, min_periods=5).mean()
    volume_std = volume.rolling(20, min_periods=5).std().replace(0.0, np.nan)
    out["volume_zscore_20"] = (volume - volume_mean) / volume_std

    range_ = out["high_low_range"]
    range_mean = range_.rolling(20, min_periods=5).mean()
    range_std = range_.rolling(20, min_periods=5).std().replace(0.0, np.nan)
    out["range_zscore_20"] = (range_ - range_mean) / range_std
    out["autocorr_20"] = _rolling_autocorr(log_return.fillna(0.0), 20, lag=1)

    trend_den = out["rolling_vol_20"].replace(0.0, np.nan) * np.sqrt(20.0)
    out["trend_strength"] = out["momentum_20"].abs() / trend_den
    close_mean = close.rolling(20, min_periods=5).mean()
    close_std = close.rolling(20, min_periods=5).std().replace(0.0, np.nan)
    out["mean_reversion_score"] = -(close - close_mean) / close_std
    out["realized_skew"] = log_return.rolling(20, min_periods=8).skew()
    out["realized_kurtosis"] = log_return.rolling(20, min_periods=8).kurt()

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
