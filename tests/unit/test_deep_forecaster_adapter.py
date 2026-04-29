import numpy as np

from market_ai.config import Settings
from market_ai.modeling.deep.artifacts import save_deep_artifact, write_deep_metadata
from market_ai.modeling.deep.lstm_tcn_fusion import DeepLstmTcnFusion
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
