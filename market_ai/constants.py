from __future__ import annotations

from datetime import timedelta


CONFIDENCE_Z = 1.96
FALLBACK_INTERVAL = "1d"

SUPPORTED_FORECAST_INTERVALS = {"1d", "1h"}
SUPPORTED_DISPLAY_HORIZONS = (7, 14, 30)
MAX_MODEL_HORIZON = 30
INTERVAL_TO_MODEL_HORIZONS = {
    "1d": (8, 14, 30),
    "1h": (30,),
    "30m": (30,),
    "15m": (30,),
}

INTERVAL_TO_PERIOD = {
    "1d": "2y",
    "1h": "180d",
    "30m": "60d",
    "15m": "30d",
}

INTERVAL_TO_PERIOD_CANDIDATES = {
    "1d": ["2y", "1y", "6mo"],
    "1h": ["180d", "120d", "90d"],
    "30m": ["60d", "30d", "14d"],
    "15m": ["60d", "30d", "14d"],
}

INTERVAL_TO_HORIZON = {
    "1d": MAX_MODEL_HORIZON,
    "1h": MAX_MODEL_HORIZON,
    "30m": MAX_MODEL_HORIZON,
    "15m": MAX_MODEL_HORIZON,
}


def select_model_horizon(interval: str, requested_horizon: int | None) -> tuple[int, int]:
    """Return artifact horizon and display horizon for a user request."""
    default_horizon = INTERVAL_TO_HORIZON.get(interval, INTERVAL_TO_HORIZON[FALLBACK_INTERVAL])
    if requested_horizon is None or int(requested_horizon) <= 0:
        return default_horizon, default_horizon
    display_horizon = max(1, min(int(requested_horizon), default_horizon))
    candidates = INTERVAL_TO_MODEL_HORIZONS.get(interval, (default_horizon,))
    for candidate in sorted(int(value) for value in candidates):
        if candidate >= display_horizon:
            return candidate, display_horizon
    return max(candidates), display_horizon

INTERVAL_TO_RETURN_CLIP = {
    "1d": 0.08,
    "1h": 0.03,
    "30m": 0.02,
    "15m": 0.015,
}

INTERVAL_TO_MAX_LOG_BAND = {
    "1d": 0.22,
    "1h": 0.07,
    "30m": 0.075,
    "15m": 0.065,
}

INTERVAL_TO_DELTA = {
    "1d": timedelta(days=1),
    "1h": timedelta(hours=1),
    "30m": timedelta(minutes=30),
    "15m": timedelta(minutes=15),
}
