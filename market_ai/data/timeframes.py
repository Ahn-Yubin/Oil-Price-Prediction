from __future__ import annotations

from market_ai.config import Settings
from market_ai.constants import FALLBACK_INTERVAL, INTERVAL_TO_PERIOD, SUPPORTED_FORECAST_INTERVALS
from market_ai.schemas.market import Timeframe


SECONDS_BY_INTERVAL = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1_800,
    "1h": 3_600,
    "4h": 14_400,
    "1d": 86_400,
    "1w": 604_800,
}

YFINANCE_INTERVALS = {
    "1m": ("1m", "7d"),
    "5m": ("5m", "60d"),
    "15m": ("15m", "30d"),
    "30m": ("30m", "60d"),
    "1h": ("1h", "180d"),
    "4h": ("1h", "180d"),
    "1d": ("1d", "10y"),
    "1w": ("1wk", "10y"),
}


def normalize_timeframe(
    raw_interval: str,
    settings: Settings | None = None,
    *,
    fallback_to_supported: bool = True,
) -> Timeframe:
    default_interval = settings.default_interval if settings else FALLBACK_INTERVAL
    requested = (raw_interval or default_interval).strip().lower() or default_interval
    normalized = requested
    warning = None

    if requested not in SECONDS_BY_INTERVAL:
        normalized = default_interval if default_interval in SECONDS_BY_INTERVAL else FALLBACK_INTERVAL
        warning = f"Unsupported interval '{requested}', using '{normalized}'."

    if fallback_to_supported and normalized not in SUPPORTED_FORECAST_INTERVALS:
        fallback = default_interval if default_interval in SUPPORTED_FORECAST_INTERVALS else FALLBACK_INTERVAL
        warning = f"Forecast interval '{normalized}' is not supported yet, using '{fallback}'."
        normalized = fallback

    provider_interval, provider_period = YFINANCE_INTERVALS.get(
        normalized,
        (normalized, INTERVAL_TO_PERIOD.get(normalized, "2y")),
    )
    return Timeframe(
        requested=requested,
        normalized=normalized,
        provider_interval=provider_interval,
        provider_period=provider_period,
        seconds=SECONDS_BY_INTERVAL.get(normalized, SECONDS_BY_INTERVAL[FALLBACK_INTERVAL]),
        is_supported=normalized in SUPPORTED_FORECAST_INTERVALS,
        warning=warning,
    )
