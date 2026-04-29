from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from market_ai.config import PROJECT_DIR
from market_ai.modeling.deep.lstm_tcn_fusion import DeepLstmTcnFusion
from market_ai.modeling.deep.llm_seq_moe import LLMContextSeqMoE


DEEP_MODEL_CLASSES = {
    "deep_lstm_tcn_fusion": DeepLstmTcnFusion,
    "llm_context_seq_moe": LLMContextSeqMoE,
}


def deep_artifact_name(model_name: str, interval: str, horizon: int) -> str:
    return f"{model_name}_{interval}_h{int(horizon)}.pt"


def deep_metadata_name(model_name: str, interval: str, horizon: int) -> str:
    return f"{model_name}_{interval}_h{int(horizon)}.json"


def save_deep_artifact(
    model: torch.nn.Module,
    path: Path,
    *,
    model_name: str,
    metadata: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = model.config_dict() if hasattr(model, "config_dict") else {}
    payload = {
        "model_name": model_name,
        "model_type": "deep_sequence",
        "config": config,
        "metadata": metadata,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    return path


def write_deep_metadata(
    metadata_path: Path,
    *,
    model_name: str,
    artifact_path: Path,
    metadata: dict[str, Any],
) -> Path:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "model_name": model_name,
        "model_type": metadata.get("model_type", "deep_sequence"),
        "version": metadata.get("version", "deep_v1"),
        "artifact_file": artifact_path.name,
        "created_at": metadata.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "training_cutoff": metadata.get("training_cutoff"),
        "asset_universe": metadata.get("asset_universe", []),
        "supported_asset_classes": metadata.get("supported_asset_classes", ["unknown"]),
        "supported_intervals": metadata.get("supported_intervals", [metadata.get("interval")]),
        "lookback": metadata.get("lookback"),
        "horizon": metadata.get("horizon"),
        "target": "volatility_scaled_cumulative_log_return_distribution",
        "feature_set": metadata.get("feature_set"),
        "scaler": "recent_realized_volatility",
        "metrics": metadata.get("metrics", {}),
        "notes": metadata.get("notes"),
        "status": metadata.get("status", "available"),
        "deep_config": metadata.get("deep_config", {}),
    }
    raw["supported_intervals"] = [item for item in raw["supported_intervals"] if item]
    metadata_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def load_deep_artifact(path: Path, *, map_location: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    model_name = payload.get("model_name")
    if model_name not in DEEP_MODEL_CLASSES:
        raise ValueError(f"Unsupported deep model artifact: {model_name}")
    config = dict(payload.get("config") or payload.get("metadata", {}).get("deep_config") or {})
    model = DEEP_MODEL_CLASSES[model_name](**config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("model_name", model_name)
    metadata.setdefault("artifact_file", path.name)
    return model, metadata


def default_artifact_paths(model_name: str, interval: str, horizon: int) -> tuple[Path, Path]:
    artifact = PROJECT_DIR / "artifacts" / "models" / deep_artifact_name(model_name, interval, horizon)
    metadata = PROJECT_DIR / "artifacts" / "metadata" / deep_metadata_name(model_name, interval, horizon)
    return artifact, metadata
