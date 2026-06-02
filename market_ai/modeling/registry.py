from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from market_ai.config import PROJECT_DIR, Settings, get_settings
from market_ai.schemas.market import AssetClass, ModelInfo, ModelMetadata


MODEL_FILENAME_RE = re.compile(r"global_dl_(?P<interval>.+)_h(?P<horizon>\d+)\.npz$")
DEEP_MODEL_FILENAME_RE = re.compile(
    r"(?P<model_name>oil_context_fusion|deep_lstm_tcn_fusion|llm_context_seq_moe)_(?P<interval>.+)_h(?P<horizon>\d+)\.pt$"
)


class ModelArtifactNotFound(RuntimeError):
    pass


def metadata_sidecar_path(artifact_path: Path, metadata_dir: Path | None = None) -> Path:
    if metadata_dir is not None:
        return metadata_dir / artifact_path.with_suffix(".json").name
    if artifact_path.parent == PROJECT_DIR / "artifacts" / "models":
        return PROJECT_DIR / "artifacts" / "metadata" / artifact_path.with_suffix(".json").name
    return artifact_path.with_suffix(".json")


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_npz_embedded_meta(path: Path) -> dict[str, Any] | None:
    try:
        with np.load(path, allow_pickle=False) as npz:
            if "meta" not in npz:
                return None
            return json.loads(str(npz["meta"].item()))
    except Exception:
        return None


def _filename_interval_horizon(path: Path) -> tuple[str | None, int | None]:
    match = MODEL_FILENAME_RE.match(path.name)
    if not match:
        match = DEEP_MODEL_FILENAME_RE.match(path.name)
    if not match:
        return None, None
    return match.group("interval"), int(match.group("horizon"))


def _filename_model_name(path: Path) -> str | None:
    match = DEEP_MODEL_FILENAME_RE.match(path.name)
    return match.group("model_name") if match else None


def _metadata_from_raw(path: Path, raw: dict[str, Any], *, status: str = "available") -> ModelMetadata:
    interval, horizon = _filename_interval_horizon(path)
    interval = str(raw.get("interval") or interval or "")
    horizon = int(raw.get("horizon") or horizon or 0) or None
    symbols = list(raw.get("symbols") or raw.get("asset_universe") or [])
    metrics = {
        key: raw.get(key)
        for key in ["val_mae_ret", "val_rmse_ret", "val_mape_pct", "n_train", "n_val"]
        if key in raw
    }
    metrics.update(raw.get("metrics") or {})
    return ModelMetadata(
        model_name=str(raw.get("model_name") or _filename_model_name(path) or "pattern_mlp"),
        model_type=str(raw.get("model_type") or ("deep_sequence" if path.suffix == ".pt" else "global_dl_mlp")),
        version=str(raw.get("version") or ("deep_v1" if path.suffix == ".pt" else "legacy_npz_v1")),
        artifact_file=path.name,
        interval=interval,
        created_at=raw.get("created_at") or raw.get("trained_at"),
        train_start=raw.get("train_start"),
        train_end=raw.get("train_end"),
        training_cutoff=raw.get("training_cutoff") or raw.get("trained_at"),
        asset_universe=symbols,
        supported_asset_classes=list(raw.get("supported_asset_classes") or [AssetClass.unknown.value]),
        supported_intervals=list(raw.get("supported_intervals") or ([interval] if interval else [])),
        lookback=raw.get("lookback") or raw.get("window"),
        horizon=horizon,
        target=raw.get("target") or raw.get("target_mode"),
        feature_set=raw.get("feature_set") or raw.get("feature_version"),
        scaler=raw.get("scaler") or "recent_realized_volatility",
        data_hash=raw.get("data_hash"),
        git_commit=raw.get("git_commit"),
        n_train=raw.get("n_train") or metrics.get("n_train"),
        n_val=raw.get("n_val") or metrics.get("n_val"),
        n_test=raw.get("n_test") or metrics.get("n_test"),
        data_source=raw.get("data_source") or (raw.get("data_report") or {}).get("source"),
        synthetic_used=raw.get("synthetic_used"),
        event_context_enabled=raw.get("event_context_enabled"),
        events_path=list(raw.get("events_path") or []),
        related_assets_enabled=raw.get("related_assets_enabled"),
        metrics=metrics,
        notes=raw.get("notes"),
        status=status,
        deep_config=dict(raw.get("deep_config") or {}),
    )


def legacy_metadata_for_artifact(path: Path) -> ModelMetadata:
    interval, horizon = _filename_interval_horizon(path)
    filename_model = _filename_model_name(path)
    return ModelMetadata(
        model_name=filename_model or "pattern_mlp",
        model_type="deep_sequence" if path.suffix == ".pt" else "global_dl_mlp",
        version="deep_without_metadata" if path.suffix == ".pt" else "legacy_npz_without_metadata",
        artifact_file=path.name,
        interval=interval,
        created_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat() if path.exists() else None,
        training_cutoff=None,
        asset_universe=[],
        supported_asset_classes=[AssetClass.unknown.value],
        supported_intervals=[interval] if interval else [],
        lookback=None,
        horizon=horizon,
        target="volatility_scaled_cumulative_returns",
        feature_set=None,
        scaler="recent_realized_volatility",
        metrics={},
        notes="Legacy artifact without embedded or sidecar metadata.",
        status="metadata_missing" if path.suffix == ".pt" else "legacy",
    )


def metadata_for_artifact(path: Path, metadata_dir: Path | None = None) -> ModelMetadata:
    sidecar = metadata_sidecar_path(path, metadata_dir=metadata_dir)
    if sidecar.exists():
        raw = _safe_json_load(sidecar)
        if raw is not None:
            try:
                return ModelMetadata(**raw)
            except Exception:
                return _metadata_from_raw(path, raw, status="metadata_invalid_fallback")

    embedded = _read_npz_embedded_meta(path)
    if embedded is not None:
        return _metadata_from_raw(path, embedded, status="legacy")

    return legacy_metadata_for_artifact(path)


def build_sidecar_metadata(path: Path, embedded_meta: dict[str, Any]) -> ModelMetadata:
    raw = {
        **embedded_meta,
        "model_name": embedded_meta.get("model_name", "pattern_mlp"),
        "model_type": embedded_meta.get("model_type", "global_dl_mlp"),
        "version": embedded_meta.get("version", "legacy_npz_v1"),
        "artifact_file": path.name,
        "supported_asset_classes": embedded_meta.get("supported_asset_classes", [AssetClass.unknown.value]),
        "supported_intervals": embedded_meta.get("supported_intervals", [embedded_meta.get("interval")]),
        "target": embedded_meta.get("target", embedded_meta.get("target_mode")),
        "feature_set": embedded_meta.get("feature_set", embedded_meta.get("feature_version")),
        "training_cutoff": embedded_meta.get("training_cutoff", embedded_meta.get("trained_at")),
    }
    raw["supported_intervals"] = [item for item in raw["supported_intervals"] if item]
    return _metadata_from_raw(path, raw)


def write_model_metadata_sidecar(path: Path, embedded_meta: dict[str, Any], metadata_dir: Path | None = None) -> ModelMetadata:
    metadata = build_sidecar_metadata(path, embedded_meta)
    sidecar_path = metadata_sidecar_path(path, metadata_dir=metadata_dir)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(metadata.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


class ModelRegistry:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    @property
    def model_dir(self) -> Path:
        return self.settings.model_dir

    def scan(self) -> list[ModelMetadata]:
        if not self.model_dir.exists():
            return []
        artifacts = sorted([*self.model_dir.glob("*.npz"), *self.model_dir.glob("*.pt")])
        return [metadata_for_artifact(path, metadata_dir=self.settings.metadata_dir) for path in artifacts]

    def list_model_info(self) -> list[ModelInfo]:
        infos: list[ModelInfo] = []
        for metadata in self.scan():
            infos.append(
                ModelInfo(
                    name=metadata.model_name,
                    label=metadata.model_name.replace("_", " ").title(),
                    model_type=metadata.model_type,
                    version=metadata.version,
                    status=metadata.status,
                    supported_intervals=metadata.supported_intervals,
                    supported_asset_classes=metadata.supported_asset_classes,
                    training_cutoff=metadata.training_cutoff,
                    feature_version=metadata.feature_set,
                    artifact_file=metadata.artifact_file,
                    metrics=metadata.metrics,
                    notes=metadata.notes,
                )
            )
        return infos

    def resolve(
        self,
        *,
        model_name: str = "pattern_mlp",
        interval: str | None = None,
        horizon: int | None = None,
        asset_class: str | None = None,
    ) -> ModelMetadata:
        candidates = [m for m in self.scan() if m.model_name == model_name or model_name in m.artifact_file]
        if interval:
            candidates = [m for m in candidates if not m.supported_intervals or interval in m.supported_intervals]
        if horizon:
            candidates = [m for m in candidates if m.horizon is None or int(m.horizon) == int(horizon)]
        if asset_class:
            candidates = [
                m
                for m in candidates
                if not m.supported_asset_classes
                or asset_class in m.supported_asset_classes
                or AssetClass.unknown.value in m.supported_asset_classes
            ]
        if not candidates:
            raise ModelArtifactNotFound(
                f"No model artifact for model={model_name}, interval={interval}, horizon={horizon}, asset_class={asset_class}."
            )
        return candidates[0]

    def health(self) -> dict[str, Any]:
        models = self.scan()
        return {
            "model_dir": str(self.model_dir),
            "model_dir_exists": self.model_dir.exists(),
            "artifact_count": len(models),
            "artifacts": [model.artifact_file for model in models],
        }
