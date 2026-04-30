from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from market_ai.forecasting import service
from market_ai.modeling.deep.availability import DeepArtifactAvailability
from market_ai.modeling.forecasters.deep_fusion import DeepModelUnavailable
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
        timeframe=Timeframe(requested="1d", normalized="1d", provider_interval="1d", provider_period="2y", seconds=86_400),
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


def test_forecast_models_query_selects_requested_model(monkeypatch):
    monkeypatch.setattr(service, "load_market_data_window", lambda *args, **kwargs: _market_window())
    monkeypatch.setattr(service, "forecast_model_comparison", _comparison)
    bundle = service.build_forecast(symbol="CL=F", interval="1d", models="random_walk", include_scenarios=False)
    assert bundle.response.selected_models == ["random_walk"]
    assert bundle.response.primary_model == "random_walk"
    assert [model["id"] for model in bundle.forecast_models] == ["random_walk"]


def test_forecast_deep_request_falls_back_when_artifact_missing(monkeypatch):
    monkeypatch.setattr(service, "load_market_data_window", lambda *args, **kwargs: _market_window())
    monkeypatch.setattr(service, "forecast_model_comparison", _comparison)
    monkeypatch.setattr(service, "_deep_availability_by_model", _missing_deep_availability)
    monkeypatch.setattr(
        service,
        "forecast_with_deep_model",
        lambda **kwargs: (_ for _ in ()).throw(DeepModelUnavailable("missing test artifact")),
    )
    bundle = service.build_forecast(symbol="CL=F", interval="1d", models="deep_lstm_tcn_fusion", include_scenarios=False)
    assert "deep_lstm_tcn_fusion" in bundle.response.selected_models
    assert bundle.response.artifact_status["deep_lstm_tcn_fusion"] == "artifact_missing"
    assert bundle.response.primary_model == "motif"
    assert any(item.code == "deep_artifact_unavailable" for item in bundle.response.warning_objects)


def test_forecast_default_skips_missing_deep_artifacts_without_warning(monkeypatch):
    monkeypatch.setattr(service, "load_market_data_window", lambda *args, **kwargs: _market_window())
    monkeypatch.setattr(service, "forecast_model_comparison", _comparison)
    monkeypatch.setattr(service, "_deep_availability_by_model", _missing_deep_availability)
    bundle = service.build_forecast(symbol="CL=F", interval="1d", include_scenarios=False)
    assert "deep_lstm_tcn_fusion" not in bundle.response.selected_models
    assert "llm_context_seq_moe" not in bundle.response.selected_models
    assert bundle.response.artifact_status["deep_lstm_tcn_fusion"] == "artifact_missing"
    assert not any(item.code == "deep_artifact_unavailable" for item in bundle.response.warning_objects)
    assert any(item.code == "quantile_bands_uncalibrated" and item.severity == "info" for item in bundle.response.warning_objects)
