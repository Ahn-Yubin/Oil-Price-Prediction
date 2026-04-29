import json
from pathlib import Path

import numpy as np
import pytest

from market_ai.config import Settings
from market_ai.modeling.registry import ModelArtifactNotFound, ModelRegistry, metadata_for_artifact, metadata_sidecar_path


def test_legacy_npz_without_metadata(tmp_path: Path):
    path = tmp_path / "global_dl_1d_h45.npz"
    np.savez_compressed(path, w1=np.zeros((1, 1)))
    metadata = metadata_for_artifact(path)
    assert metadata.status == "legacy"
    assert metadata.supported_intervals == ["1d"]
    assert metadata.horizon == 45


def test_sidecar_metadata_preferred(tmp_path: Path):
    path = tmp_path / "global_dl_1h_h72.npz"
    np.savez_compressed(path, w1=np.zeros((1, 1)))
    sidecar = metadata_sidecar_path(path)
    sidecar.write_text(
        json.dumps(
            {
                "model_name": "pattern_mlp",
                "model_type": "global_dl_mlp",
                "version": "test",
                "artifact_file": path.name,
                "supported_intervals": ["1h"],
                "supported_asset_classes": ["futures"],
                "horizon": 72,
                "status": "available",
            }
        ),
        encoding="utf-8",
    )
    metadata = metadata_for_artifact(path)
    assert metadata.version == "test"
    assert metadata.supported_asset_classes == ["futures"]


def test_registry_resolve_and_missing(tmp_path: Path):
    path = tmp_path / "global_dl_30m_h120.npz"
    np.savez_compressed(path, meta=np.array(json.dumps({"interval": "30m", "horizon": 120, "window": 64})))
    registry = ModelRegistry(Settings(model_dir=tmp_path))
    resolved = registry.resolve(model_name="pattern_mlp", interval="30m", asset_class="unknown")
    assert resolved.artifact_file == path.name
    with pytest.raises(ModelArtifactNotFound):
        registry.resolve(model_name="missing", interval="1d")
