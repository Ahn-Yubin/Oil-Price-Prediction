import numpy as np

from market_ai.backtesting.runner import (
    point_metrics,
    probabilistic_metrics,
    rolling_origin_indices,
    run_rolling_backtest,
    write_backtest_outputs,
    _quantile_paths_from_point,
)


def _close(rows: int = 140) -> np.ndarray:
    x = np.arange(rows, dtype=float)
    return 80.0 + 0.05 * x + 1.5 * np.sin(x / 6.0)


def test_metric_functions_and_probabilistic_metrics():
    actual = np.array([100.0, 101.0, 102.0])
    pred = np.array([100.5, 100.8, 103.0])
    metrics = point_metrics(pred, actual, np.array([98.0, 99.0, 100.0]))
    assert metrics["mae"] > 0
    paths = _quantile_paths_from_point(np.array([0.0, 0.01, 0.02]), _close())
    prob = probabilistic_metrics(paths, np.array([0.0, 0.015, 0.018]))
    assert prob["pinball_loss"] >= 0
    assert 0 <= prob["coverage_80"] <= 1


def test_rolling_origin_split_respects_bounds():
    origins = rolling_origin_indices(120, 5, lookback=30, step=10, max_origins=4)
    assert len(origins) == 4
    assert origins == sorted(origins)
    assert origins[-1] <= 114


def test_small_synthetic_backtest_and_output_files(tmp_path):
    outputs = run_rolling_backtest(
        _close(),
        "1d",
        ["random_walk", "drift", "flat"],
        lookback=40,
        horizon=5,
        step=10,
        max_origins=3,
        rolling=True,
        expanding=False,
        include_regime_breakdown=True,
    )
    assert not outputs["summary"].empty
    assert not outputs["details"].empty
    assert not outputs["horizon_metrics"].empty
    assert not outputs["probabilistic_metrics"].empty
    assert not outputs["regime_metrics"].empty
    written = write_backtest_outputs(outputs, tmp_path, "SYN_1d")
    assert all(path.exists() for path in written)
