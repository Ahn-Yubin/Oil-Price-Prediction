from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_ai.schemas.market import RegimeProbabilities


@dataclass(frozen=True)
class RegimeDetectionResult:
    label: str
    probabilities: RegimeProbabilities
    confidence: float


def _returns(close) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(close), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    return np.diff(np.log(arr)) if len(arr) >= 2 else np.asarray([], dtype=np.float64)


def detect_regime(close) -> RegimeDetectionResult:
    returns = _returns(close)
    if len(returns) < 12:
        probs = RegimeProbabilities(confidence=0.35).normalized()
        return RegimeDetectionResult(label="range", probabilities=probs, confidence=probs.confidence)

    recent = returns[-min(len(returns), 20) :]
    longer = returns[-min(len(returns), 80) :]
    trend = float(np.sum(recent))
    short_vol = max(float(np.std(recent)), 1e-8)
    long_vol = max(float(np.std(longer)), 1e-8)
    shock = abs(float(recent[-1])) > short_vol * 3.0
    trend_strength = min(abs(trend) / (short_vol * np.sqrt(len(recent))), 1.0)
    vol_ratio = short_vol / long_vol

    trend_up = 0.12 + (0.45 * trend_strength if trend > 0 else 0.0)
    trend_down = 0.12 + (0.45 * trend_strength if trend < 0 else 0.0)
    high_volatility = 0.12 + min(max(vol_ratio - 1.0, 0.0) * 0.35, 0.45)
    range_prob = 0.35 if trend_strength < 0.35 else 0.16
    event_driven = 0.08 + (0.25 if shock else 0.0)
    probs = RegimeProbabilities(
        trend_up=trend_up,
        trend_down=trend_down,
        range=range_prob,
        high_volatility=high_volatility,
        event_driven=event_driven,
        confidence=float(np.clip(0.45 + trend_strength * 0.25 + abs(vol_ratio - 1.0) * 0.1, 0.35, 0.85)),
    ).normalized()
    values = {
        "trend_up": probs.trend_up,
        "trend_down": probs.trend_down,
        "range": probs.range,
        "high_volatility": probs.high_volatility,
        "event_driven": probs.event_driven,
    }
    label = max(values, key=values.get)
    if shock:
        label = "shock"
    elif vol_ratio < 0.7 and label == "range":
        label = "low_volatility"
    return RegimeDetectionResult(label=label, probabilities=probs, confidence=probs.confidence)
