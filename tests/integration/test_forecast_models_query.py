from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from market_ai.config import get_settings
from market_ai.forecasting import scenarios as scenario_service
from market_ai.forecasting import service
from market_ai.modeling.deep.availability import DeepArtifactAvailability
from market_ai.modeling.forecasters.deep_fusion import DeepModelUnavailable
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError
from market_ai.modeling.model_catalog import DEEP_MODELS
from market_ai.schemas.llm_context import LLMContextOutput, StructuredEvent
from market_ai.schemas.market import (
    AssetClass,
    AssetMetadata,
    Candle,
    DataStatus,
    ForecastPoint,
    ForecastResponse,
    MarketDataWindow,
    MarketSymbol,
    ScenarioEventInput,
    Timeframe,
)


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


def _oil_model_with_metadata(**kwargs):
    model = _oil_model(**kwargs)
    model["metadata"] = {
        "model_name": "oil_context_fusion",
        "feature_set": "deep_price_v1",
        "target": "volatility_scaled_cumulative_log_return_distribution",
        "training_cutoff": "2025-12-31T00:00:00+00:00",
        "metrics": {
            "validation_mae": 4.2,
            "validation_rmse": 5.3,
            "validation_mape": 6.4,
        },
    }
    return model


def test_scenario_forecast_builds_context_override_from_llm_event(monkeypatch):
    market = _market_window()
    captured = {"event_context_frames": [], "adapter_flags": []}

    class FakeScenarioEncoder:
        def encode_events(self, context):
            captured["context"] = context
            return LLMContextOutput(
                events=[
                    StructuredEvent(
                        event_type="geopolitical_supply_shock",
                        affected_assets=["CL=F"],
                        directional_bias="bullish",
                        impact_strength=0.9,
                        uncertainty=0.35,
                        time_decay=0.85,
                        summary="Iran blockade creates oil supply disruption risk",
                        risk_factors=["Hormuz shipping disruption"],
                    )
                ],
                overall_bias="bullish",
                impact_score=0.9,
                uncertainty=0.35,
                event_embedding=[1.0, 0.9, 0.35, 0.85, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.8],
                explanation="Structured scenario context only.",
                warnings=[],
            )

    def fake_build_forecast(**kwargs):
        captured["event_context_frames"].append(kwargs["event_context_frame_override"])
        captured["adapter_flags"].append(kwargs.get("apply_event_path_adapter"))
        horizon = int(kwargs.get("horizon") or 3)
        forecast = [
            ForecastPoint(
                time=market.candles[-1].time + (idx + 1) * 86_400,
                p05=82.0 + idx,
                p10=83.0 + idx,
                p25=84.0 + idx,
                p50=88.0 + idx,
                p75=89.0 + idx,
                p90=90.0 + idx,
                p95=91.0 + idx,
                expected_return=0.03,
                expected_volatility=0.04,
                prob_up=0.7,
                confidence=0.6,
            )
            for idx in range(horizon)
        ]
        response = ForecastResponse(
            symbol="CL=F",
            asset_metadata=AssetMetadata(symbol="CL=F", provider_symbol="CL=F", asset_class=AssetClass.futures),
            interval="1d",
            generated_at=datetime.now(timezone.utc),
            current_price=85.0,
            data_status=market.data_status,
            forecast=forecast,
            primary_model="oil_context_fusion",
        )
        return service.ForecastBundle(
            response=response,
            market_data=market,
            forecast_models=[
                {
                    "id": "oil_context_fusion",
                    "points": [{"time": market.candles[-1].time, "value": 85.0}]
                    + [{"time": point.time, "value": point.p50} for point in forecast],
                }
            ],
            metrics={},
            model_info={},
            horizon=horizon,
        )

    monkeypatch.setattr(scenario_service, "load_market_data_window", lambda *args, **kwargs: market)
    monkeypatch.setattr(scenario_service, "encoder_for_mode", lambda *args, **kwargs: FakeScenarioEncoder())
    monkeypatch.setattr(scenario_service, "build_forecast", fake_build_forecast)

    origin = datetime.fromtimestamp(market.candles[-1].time, tz=timezone.utc)
    response = scenario_service.build_scenario_forecast(
        title="이란 침공",
        content="모레 트럼프가 이란을 다시 침공하면 유가 동향은?",
        event_time=origin + pd.Timedelta(days=2),
        symbol="CL=F",
        interval="1d",
        horizon=3,
        settings=get_settings().model_copy(
            update={
                "enable_llm_context": True,
                "enable_external_llm_calls": True,
                "llm_api_key": "unit-key",
                "llm_context_mode": "google_generative",
            }
        ),
    )

    rows = [frame.iloc[0] for frame in captured["event_context_frames"]]
    assert response.points[-1].value == 90.0
    assert any(row.get("raw_geopolitical_pressure", 0.0) >= 0.5 for row in rows)
    assert any(row.get("raw_supply_pressure", 0.0) >= 0.5 for row in rows)
    assert all(flag is False for flag in captured["adapter_flags"])
    assert all(row["feature_available_at"] == pd.Timestamp(market.candles[-1].time, unit="s", tz="UTC").isoformat() for row in rows)
    assert "Do not output oil price targets" in captured["context"].news[0].text
    assert response.llm_context_summary["model_context_schedule"]["output_postprocessing"] is False
    assert "target_price" not in str(response.model_dump()).lower()


def test_scenario_forecast_uses_horizon_event_context_schedule(monkeypatch):
    market = _market_window()
    captured = {"frames": [], "adapter_flags": []}

    class NeutralScenarioEncoder:
        def encode_events(self, context):
            return LLMContextOutput(
                events=[],
                overall_bias="unknown",
                impact_score=0.1,
                uncertainty=0.3,
                event_embedding=[0.0] * 13,
                explanation="Structured context only.",
                warnings=[],
            )

    def fake_build_forecast(**kwargs):
        frame = kwargs["event_context_frame_override"]
        captured["frames"].append(frame.copy())
        captured["adapter_flags"].append(kwargs.get("apply_event_path_adapter"))
        row = frame.iloc[0]
        net = float(row.get("raw_net_pressure", 0.0) or 0.0)
        bearish = float(row.get("raw_bearish_pressure", 0.0) or 0.0)
        bullish = float(row.get("raw_bullish_pressure", 0.0) or 0.0)
        current = 85.0
        horizon = 30
        if bearish > 0.25 and bullish > 0.25:
            multiplier = 0.99
        elif net > 0.25:
            multiplier = 1.08
        elif net < -0.25:
            multiplier = 0.94
        else:
            multiplier = 1.0
        forecast = [
            ForecastPoint(
                time=market.candles[-1].time + (idx + 1) * 86_400,
                p05=current * multiplier * 0.95,
                p10=current * multiplier * 0.96,
                p25=current * multiplier * 0.98,
                p50=current * multiplier,
                p75=current * multiplier * 1.02,
                p90=current * multiplier * 1.04,
                p95=current * multiplier * 1.05,
                expected_return=float(np.log(multiplier)),
                expected_volatility=0.02,
                prob_up=0.7 if multiplier > 1.0 else 0.3 if multiplier < 1.0 else 0.5,
                confidence=0.5,
            )
            for idx in range(horizon)
        ]
        response = ForecastResponse(
            symbol="CL=F",
            asset_metadata=AssetMetadata(symbol="CL=F", provider_symbol="CL=F", asset_class=AssetClass.futures),
            interval="1d",
            generated_at=datetime.now(timezone.utc),
            current_price=current,
            data_status=market.data_status,
            forecast=forecast,
            primary_model="oil_context_fusion",
        )
        return service.ForecastBundle(
            response=response,
            market_data=market,
            forecast_models=[
                {
                    "id": "oil_context_fusion",
                    "points": [{"time": market.candles[-1].time, "value": current}]
                    + [{"time": point.time, "value": current} for point in forecast],
                }
            ],
            metrics={},
            model_info={},
            horizon=horizon,
        )

    monkeypatch.setattr(scenario_service, "load_market_data_window", lambda *args, **kwargs: market)
    monkeypatch.setattr(scenario_service, "encoder_for_mode", lambda *args, **kwargs: NeutralScenarioEncoder())
    monkeypatch.setattr(scenario_service, "build_forecast", fake_build_forecast)

    origin = datetime.fromtimestamp(market.candles[-1].time, tz=timezone.utc)
    response = scenario_service.build_scenario_forecast(
        title="혼합 시나리오",
        content="상방 공급 차질 이후 증산 뉴스가 나온다.",
        events=[
            ScenarioEventInput(
                title="호르무즈 해협 재봉쇄",
                content="미국의 이란 제재와 호르무즈 해협 봉쇄로 공급 차질 우려가 커진다.",
                event_time=origin + pd.Timedelta(days=9),
            ),
            ScenarioEventInput(
                title="OPEC 증산",
                content="OPEC이 일일 석유 생산량을 크게 증산한다.",
                event_time=origin + pd.Timedelta(days=20),
            ),
        ],
        symbol="CL=F",
        interval="1d",
        settings=get_settings().model_copy(
            update={
                "enable_llm_context": True,
                "enable_external_llm_calls": True,
                "llm_api_key": "unit-key",
                "llm_context_mode": "google_generative",
            }
        ),
    )

    values = np.asarray([point.value for point in response.points], dtype=np.float64)
    assert response.llm_context_summary["overall_bias"] == "mixed"
    schedule = response.llm_context_summary["model_context_schedule"]
    assert schedule["mode"] == "horizon_event_context_schedule"
    assert schedule["output_postprocessing"] is False
    assert schedule["model_calls"] >= 3
    assert "scheduled_event_adjustment" not in response.llm_context_summary
    assert all(flag is False for flag in captured["adapter_flags"])
    assert any(float(frame.iloc[0].get("raw_net_pressure", 0.0) or 0.0) > 0.25 for frame in captured["frames"])
    assert any(float(frame.iloc[0].get("raw_bearish_pressure", 0.0) or 0.0) > 0.25 for frame in captured["frames"])
    assert values[12] > values[0] * 1.05
    assert values[24] < np.max(values[10:19]) * 0.98
    assert "target_price" not in str(response.model_dump()).lower()


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


def test_selected_deep_model_metrics_do_not_use_internal_pattern_label(monkeypatch):
    monkeypatch.setattr(service, "load_market_data_window", lambda *args, **kwargs: _market_window())
    monkeypatch.setattr(service, "forecast_model_comparison", _comparison)
    monkeypatch.setattr(service, "_deep_availability_by_model", _available_deep_availability)
    monkeypatch.setattr(service, "forecast_with_deep_model", _oil_model_with_metadata)
    bundle = service.build_forecast(symbol="CL=F", interval="1d", models="oil_context_fusion", include_scenarios=False)

    assert bundle.response.primary_model == "oil_context_fusion"
    assert bundle.metrics["model"] == "Oil Context Fusion"
    assert bundle.metrics["model_id"] == "oil_context_fusion"
    assert bundle.metrics["mape"] == 6.4
    assert bundle.metrics["target_mode"] == "volatility_scaled_cumulative_log_return_distribution"


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
