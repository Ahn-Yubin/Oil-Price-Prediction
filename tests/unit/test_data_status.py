from datetime import datetime, timezone

import pandas as pd
import pytest

from market_ai.config import Settings
from market_ai.data.providers import yfinance_provider as market_data
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable, load_market_data_window


def _frame() -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        {
            "date": [now],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
        }
    )


def test_real_data_status(monkeypatch):
    monkeypatch.setattr(market_data, "_download_ohlc", lambda provider_symbol, timeframe: _frame())
    settings = Settings(app_env="production", allow_mock_data=False)
    window = load_market_data_window("CL=F", "1d", settings=settings)
    assert window.data_status.status == "real"
    assert window.data_status.source == "yfinance"
    assert window.data_status.symbol_resolved == "CL=F"


def test_development_mock_fallback_allowed(monkeypatch, tmp_path):
    def fail(_provider_symbol, _timeframe):
        raise ValueError("network unavailable")

    monkeypatch.setattr(market_data, "_download_ohlc", fail)
    settings = Settings(
        app_env="development",
        allow_mock_data=False,
        baseline_ohlc_path=tmp_path / "missing_ohlc.csv",
        baseline_predictions_path=tmp_path / "missing_predictions.csv",
    )
    window = load_market_data_window("BAD", "1d", settings=settings)
    assert window.data_status.status == "mock"
    assert window.data_status.source == "mock"
    assert window.candles


def test_production_mock_fallback_forbidden(monkeypatch, tmp_path):
    def fail(_provider_symbol, _timeframe):
        raise ValueError("network unavailable")

    monkeypatch.setattr(market_data, "_download_ohlc", fail)
    settings = Settings(
        app_env="production",
        allow_mock_data=False,
        baseline_ohlc_path=tmp_path / "missing_ohlc.csv",
        baseline_predictions_path=tmp_path / "missing_predictions.csv",
    )
    with pytest.raises(MarketDataUnavailable):
        load_market_data_window("BAD", "1d", settings=settings)
