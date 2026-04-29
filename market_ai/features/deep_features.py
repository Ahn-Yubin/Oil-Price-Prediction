from __future__ import annotations

import numpy as np
import pandas as pd

from market_ai.features.price_features import build_price_features


DEEP_FEATURE_VERSION = "deep_price_v1"

PRICE_FEATURE_COLUMNS: tuple[str, ...] = (
    "log_return",
    "vol_scaled_return",
    "high_low_range",
    "close_to_open_return",
    "true_range",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "drawdown_20",
    "drawdown_60",
    "volume_zscore_20",
    "range_zscore_20",
    "autocorr_20",
    "trend_strength",
    "mean_reversion_score",
    "realized_skew",
    "realized_kurtosis",
    "cycle_strength",
    "cycle_phase_sin",
    "cycle_phase_cos",
)

CROSS_ASSET_FEATURE_COLUMNS: tuple[str, ...] = (
    "related_returns",
    "related_rolling_corr",
    "spread",
    "relative_strength",
    "risk_on_off_proxy",
    "missing_indicator",
)


def _safe_std(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    out = float(np.std(values))
    return out if np.isfinite(out) else 0.0


def _cycle_features(log_return: pd.Series, window: int = 60) -> pd.DataFrame:
    values = pd.to_numeric(log_return, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    strength = np.zeros(len(values), dtype=np.float64)
    phase_sin = np.zeros(len(values), dtype=np.float64)
    phase_cos = np.ones(len(values), dtype=np.float64)
    for end in range(len(values)):
        start = max(0, end - window + 1)
        hist = values[start : end + 1]
        if len(hist) < 12:
            continue
        demeaned = hist - float(np.mean(hist))
        denom = max(_safe_std(demeaned), 1e-8)
        spectrum = np.fft.rfft(demeaned)
        if len(spectrum) <= 2:
            continue
        idx = int(np.argmax(np.abs(spectrum[1:])) + 1)
        amplitude = float(np.abs(spectrum[idx]) / len(hist)) / denom
        phase = float(np.angle(spectrum[idx]))
        strength[end] = min(amplitude, 5.0)
        phase_sin[end] = float(np.sin(phase))
        phase_cos[end] = float(np.cos(phase))
    return pd.DataFrame(
        {
            "cycle_strength": strength,
            "cycle_phase_sin": phase_sin,
            "cycle_phase_cos": phase_cos,
        },
        index=log_return.index,
    )


def build_deep_price_features(candles: pd.DataFrame) -> pd.DataFrame:
    features = build_price_features(candles)
    rolling_vol = features["rolling_vol_20"].replace(0.0, np.nan).abs()
    features["vol_scaled_return"] = features["log_return"] / rolling_vol
    features = pd.concat([features, _cycle_features(features["log_return"])], axis=1)
    if "date" in features.columns:
        out = features[["date", *PRICE_FEATURE_COLUMNS]].copy()
    else:
        out = features[list(PRICE_FEATURE_COLUMNS)].copy()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def empty_cross_asset_window(lookback: int) -> np.ndarray:
    arr = np.zeros((lookback, len(CROSS_ASSET_FEATURE_COLUMNS)), dtype=np.float32)
    arr[:, CROSS_ASSET_FEATURE_COLUMNS.index("missing_indicator")] = 1.0
    return arr


def build_static_features(*, current_price: float, recent_realized_volatility: float, lookback: int, horizon: int) -> np.ndarray:
    return np.asarray(
        [
            np.log(max(float(current_price), 1e-8)),
            float(recent_realized_volatility),
            float(lookback),
            float(horizon),
        ],
        dtype=np.float32,
    )
