import numpy as np
import pandas as pd

from market_ai.backtesting.runner import (
    point_metrics,
    probabilistic_metrics,
    rolling_origin_indices,
    run_rolling_backtest,
    write_backtest_outputs,
    _quantile_paths_from_point,
    _online_residual_correction,
)


def _close(rows: int = 140) -> np.ndarray:
    x = np.arange(rows, dtype=float)
    return 80.0 + 0.05 * x + 1.5 * np.sin(x / 6.0)


def test_metric_functions_and_probabilistic_metrics():
    actual = np.array([100.0, 101.0, 102.0])
    pred = np.array([100.5, 100.8, 103.0])
    metrics = point_metrics(pred, actual, np.array([98.0, 99.0, 100.0]))
    assert metrics["mae"] > 0
    assert metrics["mape"] > 0
    assert "shape_score" in metrics
    assert metrics["range_ratio"] > 0
    paths = _quantile_paths_from_point(np.array([0.0, 0.01, 0.02]), _close())
    prob = probabilistic_metrics(paths, np.array([0.0, 0.015, 0.018]))
    assert prob["pinball_loss"] >= 0
    assert 0 <= prob["coverage_80"] <= 1


def test_rolling_origin_split_respects_bounds():
    origins = rolling_origin_indices(120, 5, lookback=30, step=10, max_origins=4)
    assert len(origins) == 4
    assert origins == sorted(origins)
    assert origins[-1] <= 114


def test_online_residual_correction_uses_only_available_origins():
    history = [(idx, np.array([0.01, 0.02, 0.03])) for idx in range(1, 9)]
    history.append((20, np.array([0.20, 0.20, 0.20])))
    correction, samples = _online_residual_correction(history, origin=12, horizon=3)
    assert samples == 8
    assert correction[0] > 0
    assert correction[0] < 0.18

    later_correction, later_samples = _online_residual_correction(history, origin=21, horizon=3)
    assert later_samples == 8
    assert later_correction[0] <= 0.18

    no_correction, no_samples = _online_residual_correction(history[:7], origin=12, horizon=3)
    assert no_samples == 0
    assert np.allclose(no_correction, 0.0)


def test_small_synthetic_backtest_and_output_files(tmp_path):
    dates = pd.date_range("2024-01-01", periods=140, freq="D", tz="UTC")
    close = _close()
    candles = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
        }
    )
    outputs = run_rolling_backtest(
        close,
        "1d",
        ["random_walk", "drift", "flat"],
        symbol="CL=F",
        candles=candles,
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
    assert "origin_start" in outputs["summary"].columns
    assert "mape" in outputs["summary"].columns
    assert "shape_score" in outputs["summary"].columns
    assert "leakage_audit_status" in outputs["details"].columns
    written = write_backtest_outputs(outputs, tmp_path, "SYN_1d")
    assert all(path.exists() for path in written)
