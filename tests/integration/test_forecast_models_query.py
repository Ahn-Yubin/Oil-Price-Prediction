from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from market_ai.config import get_settings
from market_ai.forecasting import service
from market_ai.modeling.deep.availability import DeepArtifactAvailability
from market_ai.modeling.forecasters.deep_fusion import DeepModelUnavailable
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError
from market_ai.modeling.model_catalog import DEEP_MODELS
from market_ai.schemas.market import AssetClass, Candle, DataStatus, MarketDataWindow, MarketSymbol, Timeframe


def _market_window(rows: int = 90) -> MarketDataWindow:
    candles = []
    base_time = 1_700_000_000
    for idx in range(rows):
        close = 80.0 + idx * 0.05 + np.sin(idx / 5.0)
        candles.append(Candle(time=base_time + idx * 86_400, open=close - 0.1, high=close + 0.5, low=close - 0.5, close=close, volume=1.0))
    return MarketDataWindow(
        symbol=MarketSymbol(requested="CL=F", normalized="CL=F", provider_symbol="CL=F", asset_class=AssetClass.futures),
        timeframe=Timeframe(requested="1d", normalized="1d", provider_interval="1d", provider_period="10y", seconds=86_400),
        candles=candles,
        data_status=DataStatus(
            status="real",
            source="unit",
            symbol_requested="CL=F",
            symbol_resolved="CL=F",
            interval_requested="1d",
            interval_resolved="1d",
            updated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )


def _comparison(close, interval, horizon, z_value, return_clip, max_log_band):
    base = float(close[-1])
    path = base * np.exp(np.linspace(0.0, 0.01, horizon))
    return (
        [
            {"id": "motif", "label": "Motif", "description": "test", "color": "#fff", "values": path},
            {"id": "pattern_mlp", "label": "Pattern MLP", "description": "test", "color": "#fff", "values": path},
        ],
        {"feature_version": "test", "trained_at": "2025-01-01", "_ci_band_values": np.repeat(0.02, horizon)},
    )


def _missing_deep_availability(settings, interval, horizon):
    return {
        name: DeepArtifactAvailability(
            model_name=name,
            interval=interval,
            horizon=horizon,
            status="artifact_missing",
            expected_artifact_file=f"{name}_{interval}_h{horizon}.pt",
            expected_metadata_file=f"{name}_{interval}_h{horizon}.json",
            artifact_path=Path("artifacts/models") / f"{name}_{interval}_h{horizon}.pt",
            metadata_path=Path("artifacts/metadata") / f"{name}_{interval}_h{horizon}.json",
            training_command=f"python scripts/train/train_deep_fusion_models.py --model {name} --interval {interval}",
            metadata={},
            reason="missing test artifact",
        )
        for name in DEEP_MODELS
    }


def _available_deep_availability(settings, interval, horizon):
    availability = _missing_deep_availability(settings, interval, horizon)
    return {
        name: item.__class__(
            **{
                **item.__dict__,
                "status": "available",
                "reason": None,
            }
        )
        for name, item in availability.items()
    }


def _oil_model(**kwargs):
    horizon = int(kwargs["horizon"])
    base = float(kwargs["close"][-1])
    values = base * np.exp(np.linspace(0.0, 0.01, horizon))
    return {
        "id": "oil_context_fusion",
        "label": "Oil Context Fusion",
        "description": "test",
        "color": "#fff",
        "values": values,
    }


def _turn_count(values: np.ndarray) -> int:
    steps = np.diff(np.asarray(values, dtype=np.float64))
    signs = np.sign(steps[np.abs(steps) > 1e-8])
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def test_deep_quantile_bands_are_recentered_around_display_path():
    future_times = [datetime(2025, 1, 2, tzinfo=timezone.utc), datetime(2025, 1, 3, tzinfo=timezone.utc)]
    points = service._forecast_points_from_model(
        model={
            "values": np.array([118.0, 130.0]),
            "quantile_prices": {
                "p05": np.array([90.0, 91.0]),
                "p10": np.array([94.0, 95.0]),
                "p25": np.array([98.0, 99.0]),
                "p50": np.array([100.0, 101.0]),
                "p75": np.array([103.0, 104.0]),
                "p90": np.array([106.0, 107.0]),
                "p95": np.array([110.0, 111.0]),
            },
            "prob_up": np.array([0.9, 0.95]),
            "expected_volatility": np.array([0.02, 0.03]),
            "confidence": np.array([0.1, 0.1]),
        },
        current_price=100.0,
        future_times=future_times,
        fallback_band=np.array([0.02, 0.02]),
        data_status="real",
    )

    assert [point.p50 for point in points] == [118.0, 130.0]
    assert all(point.p05 < point.p10 < point.p50 < point.p90 < point.p95 for point in points)
    for point in points:
        assert np.isclose(np.log(point.p90 / point.p50), np.log(point.p50 / point.p10))
        assert np.isclose(np.log(point.p95 / point.p50), np.log(point.p50 / point.p05))


def test_event_regime_adapter_handles_bullish_breakout_without_future_data():
    returns = np.tile(np.array([0.010, -0.006], dtype=np.float64), 35)
    close = 80.0 * np.exp(np.cumsum(returns))
    deep_values = np.repeat(float(close[-1]), 30)
    event_frame = pd.DataFrame(
        [
            {
                "feature_available_at": "2026-02-24T00:00:00Z",
                "symbol": "CL=F",
                "directional_bias_score": 1.0,
                "raw_net_pressure": 0.78,
                "raw_geopolitical_pressure": 0.74,
            }
        ]
    )

    adapted, info = service._event_regime_path_adapter(
        close=close,
        interval="1d",
        as_of_time=int(pd.Timestamp("2026-02-24T09:00:00+09:00").timestamp()),
        symbol="CL=F",
        deep_values=deep_values,
        pattern_values=deep_values,
        event_context_frame=event_frame,
        settings=get_settings(),
    )

    assert info["adapter"] == "bullish_event_breakout"
    assert adapted[-1] > deep_values[-1] * 1.12
    assert adapted[6] > adapted[1]


def test_event_regime_adapter_turns_geopolitical_supply_shock_into_upside_path():
    returns = np.tile(np.array([0.004, -0.003, 0.002], dtype=np.float64), 30)
    close = 85.0 * np.exp(np.cumsum(returns))
    deep_values = np.repeat(float(close[-1]), 30)
    event_frame = pd.DataFrame(
        [
            {
                "feature_available_at": "2026-06-05T00:00:00Z",
                "symbol": "CL=F",
                "directional_bias_score": 0.05,
                "impact_score": 0.82,
                "bullish_event_score": 0.62,
                "bearish_event_score": 0.20,
                "geopolitical_event_score": 0.84,
                "raw_bullish_pressure": 0.55,
                "raw_bearish_pressure": 0.34,
                "raw_net_pressure": 0.21,
                "raw_energy_pressure": 0.78,
                "raw_geopolitical_pressure": 0.90,
                "raw_supply_pressure": 0.82,
                "source_diversity_score": 0.65,
            }
        ]
    )

    adapted, info = service._event_regime_path_adapter(
        close=close,
        interval="1d",
        as_of_time=int(pd.Timestamp("2026-06-05T09:00:00+09:00").timestamp()),
        symbol="CL=F",
        deep_values=deep_values,
        pattern_values=deep_values,
        event_context_frame=event_frame,
        settings=get_settings(),
    )

    assert info["adapter"] == "geopolitical_supply_shock"
    assert info["geopolitical_supply_shock_score"] >= 0.55
    assert adapted[-1] > deep_values[-1] * 1.08
    assert adapted[4] > adapted[0]


def test_event_regime_adapter_uses_raw_news_pressure_when_llm_bias_flips():
    returns = np.tile(np.array([0.004, -0.002, 0.003], dtype=np.float64), 30)
    close = 85.0 * np.exp(np.cumsum(returns))
    deep_values = np.repeat(float(close[-1]), 30)
    event_frame = pd.DataFrame(
        [
            {
                "feature_available_at": "2026-02-26T00:00:00Z",
                "symbol": "CL=F",
                "directional_bias_score": -1.0,
                "impact_score": 0.50,
                "bullish_event_score": 0.0,
                "bearish_event_score": 1.0,
                "geopolitical_event_score": 0.0,
                "raw_bullish_pressure": 0.71,
                "raw_bearish_pressure": 0.05,
                "raw_net_pressure": 0.66,
                "raw_energy_pressure": 0.99,
                "raw_geopolitical_pressure": 0.69,
                "raw_supply_pressure": 0.30,
                "source_diversity_score": 0.60,
            }
        ]
    )

    adapted, info = service._event_regime_path_adapter(
        close=close,
        interval="1d",
        as_of_time=int(pd.Timestamp("2026-02-26T09:00:00+09:00").timestamp()),
        symbol="CL=F",
        deep_values=deep_values,
        pattern_values=deep_values,
        event_context_frame=event_frame,
        settings=get_settings(),
    )

    assert info["adapter"] == "event_risk_premium"
    assert info["event_upside_pressure_score"] > 0.60
    assert adapted[-1] > deep_values[-1] * 1.20


def test_event_risk_premium_keeps_path_shape_instead_of_straight_line():
    returns = np.tile(np.array([0.010, -0.007, 0.006, -0.004, 0.005], dtype=np.float64), 20)
    close = 85.0 * np.exp(np.cumsum(returns))
    current = float(close[-1])
    horizon = 30
    ramp = np.linspace(1.0 / horizon, 1.0, horizon)
    residual = 0.035 * np.sin(np.linspace(0.0, 5.0 * np.pi, horizon))
    deep_values = current * np.exp(0.02 * ramp)
    pattern_values = current * np.exp(0.02 * ramp + residual)
    event_frame = pd.DataFrame(
        [
            {
                "feature_available_at": "2026-02-27T00:00:00Z",
                "symbol": "CL=F",
                "directional_bias_score": -1.0,
                "impact_score": 0.50,
                "bullish_event_score": 0.0,
                "bearish_event_score": 1.0,
                "geopolitical_event_score": 0.0,
                "raw_bullish_pressure": 0.71,
                "raw_bearish_pressure": 0.05,
                "raw_net_pressure": 0.66,
                "raw_energy_pressure": 0.99,
                "raw_geopolitical_pressure": 0.69,
                "raw_supply_pressure": 0.30,
                "source_diversity_score": 0.60,
            }
        ]
    )

    adapted, info = service._event_regime_path_adapter(
        close=close,
        interval="1d",
        as_of_time=int(pd.Timestamp("2026-02-27T09:00:00+09:00").timestamp()),
        symbol="CL=F",
        deep_values=deep_values,
        pattern_values=pattern_values,
        event_context_frame=event_frame,
        settings=get_settings(),
    )

    assert info["adapter"] == "event_risk_premium"
    assert adapted[-1] > current * 1.10
    assert _turn_count(adapted) >= 3


def test_event_regime_adapter_handles_overextended_mean_reversion():
    returns = np.repeat(0.018, 70)
    close = 80.0 * np.exp(np.cumsum(returns))
    deep_values = np.repeat(float(close[-1]), 30)

    adapted, info = service._event_regime_path_adapter(
        close=close,
        interval="1d",
        as_of_time=int(pd.Timestamp("2026-04-06T09:00:00+09:00").timestamp()),
        symbol="CL=F",
        deep_values=deep_values,
        pattern_values=deep_values,
        event_context_frame=pd.DataFrame(),
        settings=get_settings(),
    )

    assert info["adapter"] == "overextended_mean_reversion"
    assert np.min(adapted) < deep_values[-1] * 0.90
    assert adapted[-1] < deep_values[-1]


def test_event_regime_adapter_detemplates_normal_path_with_pattern_residual():
    close = 95.0 + np.sin(np.arange(80, dtype=np.float64) / 4.0)
    current = float(close[-1])
    horizon = 30
    ramp = np.linspace(1.0 / horizon, 1.0, horizon)
    deep_values = current * np.exp(0.02 * ramp)
    pattern_values = current * np.exp(0.02 * ramp + 0.03 * np.sin(np.linspace(0.0, np.pi * 2.0, horizon)))

    adapted, info = service._event_regime_path_adapter(
        close=close,
        interval="1d",
        as_of_time=int(pd.Timestamp("2026-01-01T00:00:00Z").timestamp()),
        symbol="CL=F",
        deep_values=deep_values,
        pattern_values=pattern_values,
        event_context_frame=pd.DataFrame(),
        settings=get_settings(),
    )

    assert info["adapter"] == "pattern_residual_detemplate"
    assert np.isclose(adapted[-1], deep_values[-1])
    assert not np.allclose(adapted, deep_values)


def test_forecast_models_query_selects_requested_model(monkeypatch):
    monkeypatch.setattr(service, "load_market_data_window", lambda *args, **kwargs: _market_window())
    monkeypatch.setattr(service, "forecast_model_comparison", _comparison)
    monkeypatch.setattr(service, "_deep_availability_by_model", _available_deep_availability)
    monkeypatch.setattr(service, "forecast_with_deep_model", _oil_model)
    bundle = service.build_forecast(symbol="CL=F", interval="1d", models="oil_context_fusion", include_scenarios=False)
    assert bundle.response.selected_models == ["oil_context_fusion"]
    assert bundle.response.primary_model == "oil_context_fusion"
    assert [model["id"] for model in bundle.forecast_models] == ["oil_context_fusion"]


def test_forecast_deep_request_falls_back_when_artifact_missing(monkeypatch):
    monkeypatch.setattr(service, "load_market_data_window", lambda *args, **kwargs: _market_window())
    monkeypatch.setattr(service, "forecast_model_comparison", _comparison)
    monkeypatch.setattr(service, "_deep_availability_by_model", _missing_deep_availability)
    monkeypatch.setattr(
        service,
        "forecast_with_deep_model",
        lambda **kwargs: (_ for _ in ()).throw(DeepModelUnavailable("missing test artifact")),
    )
    bundle = service.build_forecast(symbol="CL=F", interval="1d", models="oil_context_fusion", include_scenarios=False)
    assert "oil_context_fusion" in bundle.response.selected_models
    assert bundle.response.artifact_status["oil_context_fusion"] == "artifact_missing"
    assert bundle.response.primary_model == "motif"
    assert any(item.code == "deep_artifact_unavailable" for item in bundle.response.warning_objects)


def test_forecast_default_uses_unified_oil_model_and_warns_when_missing(monkeypatch):
    monkeypatch.setattr(service, "load_market_data_window", lambda *args, **kwargs: _market_window())
    monkeypatch.setattr(service, "forecast_model_comparison", _comparison)
    monkeypatch.setattr(service, "_deep_availability_by_model", _missing_deep_availability)
    bundle = service.build_forecast(symbol="CL=F", interval="1d", include_scenarios=False)
    assert bundle.response.selected_models == ["oil_context_fusion"]
    assert bundle.response.artifact_status["oil_context_fusion"] == "artifact_missing"
    assert any(item.code == "deep_artifact_unavailable" for item in bundle.response.warning_objects)
    assert bundle.response.calibration_status["calibration_status"] == "calibrated"
    assert not any(item.code == "quantile_bands_uncalibrated" for item in bundle.response.warning_objects)


def test_forecast_serves_baseline_when_pattern_artifact_for_horizon_is_missing(monkeypatch):
    monkeypatch.setattr(service, "load_market_data_window", lambda *args, **kwargs: _market_window())
    monkeypatch.setattr(service, "_deep_availability_by_model", _missing_deep_availability)

    def missing_pattern(*args, **kwargs):
        raise PretrainedModelNotFoundError("missing h8 artifact")

    monkeypatch.setattr(service, "forecast_model_comparison", missing_pattern)
    bundle = service.build_forecast(symbol="CL=F", interval="1d", horizon=8, include_scenarios=False)

    assert bundle.horizon == 8
    assert bundle.response.primary_model == "random_walk"
    assert [model["id"] for model in bundle.forecast_models] == ["random_walk"]
    assert any(item.code == "pretrained_pattern_artifact_unavailable" for item in bundle.response.warning_objects)
