from pathlib import Path

import torch

from market_ai.modeling.deep.artifacts import load_deep_artifact, save_deep_artifact, write_deep_metadata
from market_ai.modeling.deep.lstm_tcn_fusion import DeepLstmTcnFusion


def test_deep_artifact_save_load_roundtrip(tmp_path: Path):
    model = DeepLstmTcnFusion(price_feature_dim=5, cross_asset_dim=2, event_context_dim=13, static_dim=4, horizon=3)
    artifact = tmp_path / "deep_lstm_tcn_fusion_1d_h3.pt"
    metadata = {
        "lookback": 16,
        "horizon": 3,
        "interval": "1d",
        "feature_set": "test",
        "deep_config": model.config_dict(),
    }
    save_deep_artifact(model, artifact, model_name="deep_lstm_tcn_fusion", metadata=metadata)
    write_deep_metadata(tmp_path / "deep_lstm_tcn_fusion_1d_h3.json", model_name="deep_lstm_tcn_fusion", artifact_path=artifact, metadata=metadata)
    loaded, loaded_meta = load_deep_artifact(artifact)
    out = loaded(torch.randn(1, 16, 5), torch.randn(1, 16, 2), torch.zeros(1, 13), torch.zeros(1, 4))
    assert out["quantiles"].shape == (1, 3, 7)
    assert loaded_meta["horizon"] == 3
