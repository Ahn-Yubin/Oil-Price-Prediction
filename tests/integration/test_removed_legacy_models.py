import pytest

from market_ai.backtesting.runner import FORECASTERS, run_rolling_backtest


def test_removed_legacy_models_not_in_backtest_registry():
    assert {"cycle", "lstm", "tcn", "ensemble"}.isdisjoint(FORECASTERS)


def test_removed_legacy_backtest_request_is_clear():
    with pytest.raises(ValueError, match="Removed/deprecated"):
        run_rolling_backtest(
            close=[80 + idx * 0.1 for idx in range(80)],
            interval="1d",
            model_names=["cycle"],
            lookback=20,
            horizon=3,
            step=5,
            max_origins=2,
            rolling=True,
            expanding=False,
            include_regime_breakdown=False,
        )
