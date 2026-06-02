from market_ai.backtesting.runner import run_rolling_backtest
from tests.unit.test_backtest_overhaul import _close


def test_backtest_deep_models_missing_artifact_is_unavailable_not_crash():
    outputs = run_rolling_backtest(
        _close(120),
        "1d",
        ["random_walk", "oil_context_fusion"],
        lookback=30,
        horizon=5,
        step=10,
        max_origins=2,
        rolling=True,
        expanding=False,
        include_regime_breakdown=False,
    )
    availability = outputs["model_availability"].set_index("model")
    assert availability.loc["random_walk", "status"] == "available"
    assert availability.loc["oil_context_fusion", "status"] in {"available", "unavailable"}
