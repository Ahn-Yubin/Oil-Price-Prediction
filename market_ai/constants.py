from __future__ import annotations

from datetime import timedelta


CONFIDENCE_Z = 1.96
FALLBACK_INTERVAL = "1d"

SUPPORTED_FORECAST_INTERVALS = {"1d", "1h", "30m", "15m"}

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
    "1d": 45,
    "1h": 72,
    "30m": 120,
    "15m": 192,
}

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
