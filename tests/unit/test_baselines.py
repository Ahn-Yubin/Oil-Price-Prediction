import numpy as np

from market_ai.modeling.forecasters.baselines import BASELINE_FORECASTERS, ForecastContext
from market_ai.features.target_transforms import reconstruct_price_path


def _close() -> np.ndarray:
    x = np.arange(180, dtype=float)
    return 100.0 + 0.08 * x + 2.0 * np.sin(x / 7.0)


def test_required_baselines_emit_quantile_shapes():
    context = ForecastContext(close=_close(), interval="1d", horizon=12)
    for name in ["random_walk", "seasonal_naive", "volatility_scaled_naive", "simple_moving_average_path"]:
        result = BASELINE_FORECASTERS[name](context)
        assert result.cum_log_path.shape == (12,)
        assert result.prob_up.shape == (12,)
        assert result.expected_volatility.shape == (12,)
        assert result.confidence.shape == (12,)
        for key in ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]:
            assert result.quantiles[key].shape == (12,)
            assert np.all(np.isfinite(result.quantiles[key]))


def test_baseline_quantiles_are_monotonic_and_finite():
    context = ForecastContext(close=_close(), interval="1d", horizon=8)
    result = BASELINE_FORECASTERS["seasonal_naive"](context)
    stacked = np.vstack([result.quantiles[key] for key in ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]])
    assert np.all(np.diff(stacked, axis=0) >= -1e-12)
    assert np.all(np.isfinite(stacked))


def test_price_reconstruction_uses_cumulative_log_return():
    current = 100.0
    path = np.array([0.0, 0.01, -0.02])
    prices = reconstruct_price_path(current, path)
    assert np.allclose(prices, current * np.exp(path))
