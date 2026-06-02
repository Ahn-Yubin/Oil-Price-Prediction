from pathlib import Path

import torch

from market_ai.modeling.deep.artifacts import load_deep_artifact, save_deep_artifact, write_deep_metadata
from market_ai.modeling.deep.lstm_tcn_fusion import DeepLstmTcnFusion
from market_ai.modeling.deep.oil_context_fusion import OilContextFusion


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


def test_oil_context_fusion_artifact_load_accepts_expert_names(tmp_path: Path):
    model = OilContextFusion(price_feature_dim=5, cross_asset_dim=2, event_context_dim=13, static_dim=4, horizon=3)
    artifact = tmp_path / "oil_context_fusion_1d_h3.pt"
    metadata = {
        "lookback": 16,
        "horizon": 3,
        "interval": "1d",
        "feature_set": "test",
        "deep_config": model.config_dict(),
    }
    save_deep_artifact(model, artifact, model_name="oil_context_fusion", metadata=metadata)
    loaded, loaded_meta = load_deep_artifact(artifact)
    out = loaded(torch.randn(1, 16, 5), torch.randn(1, 16, 2), torch.zeros(1, 13), torch.zeros(1, 4))
    assert out["quantiles"].shape == (1, 3, 7)
    assert out["expert_weights"].shape == (1, 6)
    assert out["expert_names"] == ("lstm", "tcn", "attention", "context", "pattern", "motif")
    assert loaded_meta["deep_config"]["expert_names"] == ["lstm", "tcn", "attention", "context", "pattern", "motif"]


def test_deep_metadata_writes_operational_fields(tmp_path: Path):
    artifact = tmp_path / "deep_lstm_tcn_fusion_1d_h3.pt"
    metadata_path = tmp_path / "deep_lstm_tcn_fusion_1d_h3.json"
    write_deep_metadata(
        metadata_path,
        model_name="deep_lstm_tcn_fusion",
        artifact_path=artifact,
        metadata={
            "interval": "1d",
            "horizon": 3,
            "lookback": 16,
            "feature_set": "test",
            "train_start": "2020-01-01T00:00:00+00:00",
            "train_end": "2020-02-01T00:00:00+00:00",
            "training_cutoff": "2020-02-01T00:00:00+00:00",
            "n_train": 10,
            "n_val": 2,
            "n_test": 2,
            "data_source": "yfinance",
            "synthetic_used": False,
            "event_context_enabled": True,
            "events_path": ["events.csv"],
            "related_assets_enabled": True,
            "git_commit": "abc123",
            "status": "available",
        },
    )
    raw = metadata_path.read_text(encoding="utf-8")
    for key in [
        '"artifact_file"',
        '"interval"',
        '"target"',
        '"n_train"',
        '"data_source"',
        '"synthetic_used"',
        '"event_context_enabled"',
        '"events_path"',
        '"related_assets_enabled"',
        '"git_commit"',
        '"status"',
    ]:
        assert key in raw
