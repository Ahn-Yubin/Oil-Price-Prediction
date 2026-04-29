import numpy as np

from market_ai.modeling.forecasters.baselines import ForecastContext
from market_ai.modeling.regimes.moe import regime_ensemble_forecast
from market_ai.modeling.regimes.detector import detect_regime


def _close(rows: int = 160, high_vol: bool = False) -> np.ndarray:
    x = np.arange(rows, dtype=float)
    amp = 5.0 if high_vol else 1.0
    return 100.0 + 0.08 * x + amp * np.sin(x / 3.0)


def test_regime_probabilities_sum_to_one():
    result = detect_regime(_close())
    total = (
        result.probabilities.trend_up
        + result.probabilities.trend_down
        + result.probabilities.range
        + result.probabilities.high_volatility
        + result.probabilities.event_driven
    )
    assert abs(total - 1.0) < 1e-9
    assert 0.0 <= result.confidence <= 1.0


def test_moe_output_quantiles_are_monotonic():
    context = ForecastContext(close=_close(), interval="1d", horizon=10)
    result = regime_ensemble_forecast(context)
    stacked = np.vstack([result.quantiles[key] for key in ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]])
    assert np.all(np.diff(stacked, axis=0) >= -1e-12)
    assert result.cum_log_path.shape == (10,)


def test_high_volatility_widens_moe_bands():
    calm = regime_ensemble_forecast(ForecastContext(close=_close(high_vol=False), interval="1d", horizon=10))
    volatile = regime_ensemble_forecast(ForecastContext(close=_close(high_vol=True), interval="1d", horizon=10))
    calm_width = np.mean(calm.quantiles["p90"] - calm.quantiles["p10"])
    volatile_width = np.mean(volatile.quantiles["p90"] - volatile.quantiles["p10"])
    assert volatile_width >= calm_width
