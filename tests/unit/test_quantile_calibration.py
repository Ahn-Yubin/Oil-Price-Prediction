from __future__ import annotations

import pandas as pd

from market_ai.modeling.calibration.conformal import apply_calibration_to_points, compute_conformal_adjustment
from market_ai.schemas.market import ForecastPoint


def test_conformal_adjustment_shape_and_status():
    rows = []
    for origin in range(25):
        for step in range(1, 4):
            rows.append(
                {
                    "model": "motif",
                    "origin": origin,
                    "step": step,
                    "actual_log_return": 0.02 * step,
                    "p10_log_return": -0.01 * step,
                    "p90_log_return": 0.01 * step,
                    "p05_log_return": -0.015 * step,
                    "p95_log_return": 0.015 * step,
                }
            )
    artifact = compute_conformal_adjustment(pd.DataFrame(rows), model="motif", symbol="CL=F", interval="1d")
    assert artifact.calibration_status == "calibrated"
    assert len(artifact.adjustment_80) == 3
    assert artifact.n_origins == 25


def test_apply_calibration_widens_bands():
    point = ForecastPoint(
        time=1,
        p05=90,
        p10=95,
        p25=98,
        p50=100,
        p75=102,
        p90=105,
        p95=110,
        expected_return=0.0,
        expected_volatility=0.1,
        prob_up=0.5,
        confidence=0.5,
    )
    artifact = compute_conformal_adjustment(
        pd.DataFrame(
            [
                {
                    "model": "motif",
                    "origin": origin,
                    "step": 1,
                    "actual_log_return": 0.2,
                    "p10_log_return": -0.01,
                    "p90_log_return": 0.01,
                    "p05_log_return": -0.02,
                    "p95_log_return": 0.02,
                }
                for origin in range(25)
            ]
        ),
        model="motif",
        symbol="CL=F",
        interval="1d",
    )
    calibrated = apply_calibration_to_points([point], current_price=100, artifact=artifact)[0]
    assert calibrated.p90 > point.p90
    assert calibrated.p10 < point.p10
