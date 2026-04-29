import numpy as np
import pandas as pd

from market_ai.features.calendar_features import build_calendar_features
from market_ai.features.price_features import FEATURE_SET_VERSION, build_price_features
from market_ai.features.target_transforms import (
    cumulative_future_returns,
    reconstruct_price_path,
    to_log_returns,
    volatility_scaled_cumulative_returns,
)


def _candles(rows: int = 90) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    close = 100.0 + x * 0.1 + np.sin(x / 5.0)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) + 0.5,
            "low": np.minimum(open_, close) - 0.5,
            "close": close,
            "volume": 1000 + x,
        }
    )


def test_price_feature_shape_and_version():
    features = build_price_features(_candles())
    assert FEATURE_SET_VERSION == "price_v1"
    assert len(features) == 90
    for col in ["log_return", "rolling_vol_20", "momentum_20", "drawdown_20", "autocorr_20"]:
        assert col in features.columns
    assert np.isfinite(features.select_dtypes("number").to_numpy()).all()


def test_price_features_do_not_use_future_values_for_past_rows():
    candles = _candles()
    baseline = build_price_features(candles).iloc[:40].reset_index(drop=True)
    modified = candles.copy()
    modified.loc[70:, "close"] *= 10.0
    changed = build_price_features(modified).iloc[:40].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline, changed)


def test_target_transform_roundtrip():
    close = _candles()["close"].to_numpy()
    returns = to_log_returns(close)
    cumulative = cumulative_future_returns(returns, horizon=3)
    assert cumulative.shape[1] == 3
    scaled, scales = volatility_scaled_cumulative_returns(returns, horizon=3, vol_window=10)
    assert scaled.shape[0] == len(scales)
    reconstructed = reconstruct_price_path(close[0], np.log(close[1:4] / close[0]))
    assert np.allclose(reconstructed, close[1:4])


def test_calendar_features_basic_fields():
    out = build_calendar_features(_candles()["date"])
    assert {"day_of_week", "month", "quarter", "session"}.issubset(out.columns)
