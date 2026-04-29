from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from market_ai.data.event_providers import FileEventProvider, NullEventProvider
from market_ai.features.deep_features import (
    CROSS_ASSET_FEATURE_COLUMNS,
    DEEP_FEATURE_VERSION,
    PRICE_FEATURE_COLUMNS,
    build_deep_price_features,
    build_static_features,
    empty_cross_asset_window,
)
from market_ai.schemas.deep_learning import DeepDatasetConfig, DeepLearningSample


@dataclass(frozen=True)
class DeepDataset:
    samples: list[DeepLearningSample]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    price_feature_names: tuple[str, ...] = PRICE_FEATURE_COLUMNS
    cross_asset_feature_names: tuple[str, ...] = CROSS_ASSET_FEATURE_COLUMNS
    feature_version: str = DEEP_FEATURE_VERSION

    def tensors(self, indices: Iterable[int] | None = None) -> dict[str, np.ndarray]:
        idx = list(indices) if indices is not None else list(range(len(self.samples)))
        selected = [self.samples[i] for i in idx]
        return {
            "x_price": np.asarray([sample.x_price for sample in selected], dtype=np.float32),
            "x_cross_asset": np.asarray([sample.x_cross_asset for sample in selected], dtype=np.float32),
            "x_event_context": np.asarray([sample.x_event_context for sample in selected], dtype=np.float32),
            "x_static": np.asarray([sample.x_static for sample in selected], dtype=np.float32),
            "y_vol_scaled_cum_return": np.asarray([sample.y_vol_scaled_cum_return for sample in selected], dtype=np.float32),
            "y_direction": np.asarray([sample.y_direction for sample in selected], dtype=np.float32),
            "y_future_volatility": np.asarray([sample.y_future_volatility for sample in selected], dtype=np.float32),
        }


def _normalize_candles(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy()
    if "date" not in frame.columns:
        if "time" in frame.columns:
            frame["date"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        else:
            frame["date"] = pd.date_range("2000-01-01", periods=len(frame), freq="D", tz="UTC")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    required = ["open", "high", "low", "close"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    for col in [*required, "volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame = frame.dropna(subset=["date", *required]).sort_values("date").reset_index(drop=True)
    frame = frame[frame["close"] > 0.0].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid positive-close candles")
    return frame


def _future_volatility(returns: np.ndarray, start: int, horizon: int) -> np.ndarray:
    vals: list[float] = []
    for h in range(1, horizon + 1):
        segment = returns[start : start + h]
        vals.append(max(float(np.std(segment)) if len(segment) > 1 else 0.0, 0.0))
    return np.asarray(vals, dtype=np.float32)


def _time_split_indices(n: int, validation_ratio: float, test_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty, empty
    test_n = int(np.floor(n * test_ratio))
    val_n = int(np.floor(n * validation_ratio))
    if n >= 3:
        test_n = max(1, test_n)
        val_n = max(1, val_n)
    train_n = max(n - val_n - test_n, 0)
    if train_n == 0 and n > 0:
        train_n = max(1, n - val_n - test_n)
    train_end = min(train_n, n)
    val_end = min(train_end + val_n, n)
    return (
        np.arange(0, train_end, dtype=np.int64),
        np.arange(train_end, val_end, dtype=np.int64),
        np.arange(val_end, n, dtype=np.int64),
    )


def build_deep_dataset_from_frame(
    *,
    symbol: str,
    interval: str,
    candles: pd.DataFrame,
    config: DeepDatasetConfig,
    event_provider: FileEventProvider | None = None,
) -> DeepDataset:
    frame = _normalize_candles(candles)
    provider = event_provider or (FileEventProvider.from_env() if config.event_context_enabled else NullEventProvider())
    features = build_deep_price_features(frame)
    close = frame["close"].to_numpy(dtype=np.float64)
    log_close = np.log(close)
    returns_by_bar = np.zeros(len(close), dtype=np.float64)
    returns_by_bar[1:] = np.diff(log_close)
    lookback = int(config.lookback)
    horizon = int(config.horizon)
    min_history = max(int(config.min_history), lookback)
    samples: list[DeepLearningSample] = []

    last_origin = len(frame) - horizon - 1
    for origin in range(min_history - 1, last_origin + 1):
        feature_window = features.iloc[origin - lookback + 1 : origin + 1]
        if len(feature_window) != lookback:
            continue
        current_price = float(close[origin])
        recent = returns_by_bar[max(1, origin - 60 + 1) : origin + 1]
        recent_vol = max(float(np.std(recent)) if len(recent) > 1 else 0.0, 1e-8)
        future_log_path = log_close[origin + 1 : origin + horizon + 1] - log_close[origin]
        scaled_target = np.clip(future_log_path / recent_vol, -12.0, 12.0).astype(np.float32)
        future_returns = returns_by_bar[origin + 1 : origin + horizon + 1]
        as_of_time = pd.Timestamp(frame.loc[origin, "date"]).to_pydatetime()
        event_vector = provider.context_vector(symbol=symbol, as_of_time=as_of_time).as_list()
        samples.append(
            DeepLearningSample(
                symbol=symbol,
                interval=interval,
                as_of_time=as_of_time,
                lookback=lookback,
                horizon=horizon,
                x_price=feature_window[list(PRICE_FEATURE_COLUMNS)].to_numpy(dtype=np.float32).tolist(),
                x_cross_asset=empty_cross_asset_window(lookback).tolist(),
                x_event_context=[float(x) for x in event_vector],
                x_static=build_static_features(
                    current_price=current_price,
                    recent_realized_volatility=recent_vol,
                    lookback=lookback,
                    horizon=horizon,
                ).tolist(),
                y_vol_scaled_cum_return=scaled_target.tolist(),
                y_direction=(future_log_path > 0.0).astype(np.int64).tolist(),
                y_future_volatility=_future_volatility(future_returns, 0, horizon).tolist(),
                current_price=current_price,
                recent_realized_volatility=recent_vol,
                feature_version=DEEP_FEATURE_VERSION,
                data_status={"status": "real", "source": "dataframe"},
            )
        )

    if config.max_samples is not None and len(samples) > config.max_samples:
        samples = samples[-int(config.max_samples) :]

    train_idx, val_idx, test_idx = _time_split_indices(len(samples), config.validation_ratio, config.test_ratio)
    return DeepDataset(samples=samples, train_indices=train_idx, validation_indices=val_idx, test_indices=test_idx)


def synthetic_ohlcv(rows: int = 420, *, seed: int = 42, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.arange(rows, dtype=np.float64)
    noise = rng.normal(0.0, 0.004, size=rows)
    returns = 0.0003 + 0.006 * np.sin(x / 18.0) + noise
    close = 75.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(np.abs(close - open_) * 0.6, close * 0.006)
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=rows, freq="D", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": 10_000 + 500 * np.sin(x / 7.0) + rng.normal(0.0, 50.0, size=rows),
        }
    )


def build_synthetic_deep_dataset(config: DeepDatasetConfig) -> DeepDataset:
    frames = [synthetic_ohlcv(max(config.min_history + config.horizon + 80, 260), seed=config.seed + idx) for idx, _ in enumerate(config.symbols)]
    all_samples: list[DeepLearningSample] = []
    for symbol, frame in zip(config.symbols, frames):
        ds = build_deep_dataset_from_frame(symbol=symbol, interval=config.interval, candles=frame, config=config, event_provider=NullEventProvider())
        all_samples.extend(ds.samples)
    if config.max_samples is not None and len(all_samples) > config.max_samples:
        all_samples = all_samples[-int(config.max_samples) :]
    train_idx, val_idx, test_idx = _time_split_indices(len(all_samples), config.validation_ratio, config.test_ratio)
    return DeepDataset(samples=all_samples, train_indices=train_idx, validation_indices=val_idx, test_indices=test_idx)
