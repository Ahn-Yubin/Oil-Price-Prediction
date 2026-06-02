import numpy as np

from market_ai.config import Settings
from market_ai.modeling.deep.artifacts import save_deep_artifact, write_deep_metadata
from market_ai.modeling.deep.lstm_tcn_fusion import DeepLstmTcnFusion
from market_ai.modeling.deep.oil_context_fusion import OilContextFusion
from market_ai.modeling.forecasters import deep_fusion
from market_ai.modeling.forecasters.deep_fusion import forecast_with_deep_model


def test_deep_forecaster_adapter_loads_artifact(tmp_path):
    model = DeepLstmTcnFusion(price_feature_dim=23, cross_asset_dim=6, event_context_dim=13, static_dim=4, horizon=3)
    artifact_dir = tmp_path / "models"
    metadata_dir = tmp_path / "metadata"
    artifact = artifact_dir / "deep_lstm_tcn_fusion_1d_h3.pt"
    metadata = {
        "lookback": 16,
        "horizon": 3,
        "interval": "1d",
        "feature_set": "test",
        "deep_config": model.config_dict(),
        "supported_intervals": ["1d"],
    }
    save_deep_artifact(model, artifact, model_name="deep_lstm_tcn_fusion", metadata=metadata)
    write_deep_metadata(metadata_dir / "deep_lstm_tcn_fusion_1d_h3.json", model_name="deep_lstm_tcn_fusion", artifact_path=artifact, metadata=metadata)
    close = 80 + np.arange(80, dtype=float) * 0.1
    out = forecast_with_deep_model(
        model_name="deep_lstm_tcn_fusion",
        close=close,
        interval="1d",
        horizon=3,
        settings=Settings(model_dir=artifact_dir, metadata_dir=metadata_dir),
        symbol="CL=F",
    )
    assert out["id"] == "deep_lstm_tcn_fusion"
    assert len(out["values"]) == 3


def test_deep_forecaster_adapter_loads_oil_context_fusion_extra_metadata(tmp_path):
    model = OilContextFusion(price_feature_dim=23, cross_asset_dim=6, event_context_dim=13, static_dim=4, horizon=3)
    artifact_dir = tmp_path / "models"
    metadata_dir = tmp_path / "metadata"
    artifact = artifact_dir / "oil_context_fusion_1d_h3.pt"
    metadata = {
        "lookback": 16,
        "horizon": 3,
        "interval": "1d",
        "feature_set": "test",
        "deep_config": model.config_dict(),
        "supported_intervals": ["1d"],
        "event_context_enabled": False,
    }
    save_deep_artifact(model, artifact, model_name="oil_context_fusion", metadata=metadata)
    write_deep_metadata(metadata_dir / "oil_context_fusion_1d_h3.json", model_name="oil_context_fusion", artifact_path=artifact, metadata=metadata)
    close = 80 + np.arange(80, dtype=float) * 0.1
    out = forecast_with_deep_model(
        model_name="oil_context_fusion",
        close=close,
        interval="1d",
        horizon=3,
        settings=Settings(model_dir=artifact_dir, metadata_dir=metadata_dir, data_dir=tmp_path),
        symbol="CL=F",
    )
    assert out["id"] == "oil_context_fusion"
    assert len(out["values"]) == 3
    assert "motif" in out["metadata"]["deep_config"]["expert_names"]


def test_deep_forecaster_respects_artifact_event_context_flag(tmp_path, monkeypatch):
    model = DeepLstmTcnFusion(price_feature_dim=23, cross_asset_dim=6, event_context_dim=13, static_dim=4, horizon=3)
    artifact_dir = tmp_path / "models"
    metadata_dir = tmp_path / "metadata"
    artifact = artifact_dir / "deep_lstm_tcn_fusion_1d_h3.pt"
    metadata = {
        "lookback": 16,
        "horizon": 3,
        "interval": "1d",
        "feature_set": "test",
        "deep_config": model.config_dict(),
        "supported_intervals": ["1d"],
        "event_context_enabled": False,
    }
    save_deep_artifact(model, artifact, model_name="deep_lstm_tcn_fusion", metadata=metadata)
    write_deep_metadata(metadata_dir / "deep_lstm_tcn_fusion_1d_h3.json", model_name="deep_lstm_tcn_fusion", artifact_path=artifact, metadata=metadata)
    close = 80 + np.arange(80, dtype=float) * 0.1
    captured = {}

    def fake_predict(**kwargs):
        captured["event_provider"] = kwargs.get("event_provider")
        captured["event_context_frame"] = kwargs.get("event_context_frame")
        return {
            "quantile_prices": {
                "p05": [79.0, 79.1, 79.2],
                "p10": [79.0, 79.1, 79.2],
                "p25": [80.0, 80.1, 80.2],
                "p50": [81.0, 81.1, 81.2],
                "p75": [82.0, 82.1, 82.2],
                "p90": [83.0, 83.1, 83.2],
                "p95": [83.0, 83.1, 83.2],
            },
            "prob_up": np.repeat(0.5, 3),
            "expected_volatility": np.repeat(0.01, 3),
            "confidence": np.repeat(0.5, 3),
            "metadata": metadata,
        }

    monkeypatch.setattr(deep_fusion, "predict_deep_quantiles", fake_predict)
    forecast_with_deep_model(
        model_name="deep_lstm_tcn_fusion",
        close=close,
        interval="1d",
        horizon=3,
        settings=Settings(model_dir=artifact_dir, metadata_dir=metadata_dir, data_dir=tmp_path, enable_llm_context=True),
        symbol="CL=F",
    )

    assert captured["event_provider"] is None
    assert captured["event_context_frame"] is None
