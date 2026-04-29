import pytest

from market_ai.backtesting.runner import run_rolling_backtest
from tests.unit.test_backtest_overhaul import _close


def test_backtest_removed_model_request_fails_before_running():
    with pytest.raises(ValueError, match="Removed/deprecated"):
        run_rolling_backtest(
            _close(120),
            "1d",
            ["ensemble"],
            lookback=30,
            horizon=5,
            step=10,
            max_origins=2,
            rolling=True,
            expanding=False,
            include_regime_breakdown=False,
        )
