from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from market_ai.config import Settings
from market_ai.modeling.deep.artifacts import deep_artifact_name, deep_metadata_name
from market_ai.modeling.registry import metadata_for_artifact


PRODUCTION_DEEP_STATUS = "available"
NON_PRODUCTION_DEEP_STATUSES = {"smoke_only", "synthetic_only", "failed", "metadata_only"}


@dataclass(frozen=True)
class DeepArtifactAvailability:
    model_name: str
    interval: str
    horizon: int
    status: str
    expected_artifact_file: str
    expected_metadata_file: str
    artifact_path: Path
    metadata_path: Path
    training_command: str
    metadata: dict[str, Any]
    reason: str | None = None

    @property
    def is_available(self) -> bool:
        return self.status == PRODUCTION_DEEP_STATUS

    def as_api_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "status": self.status,
            "interval": self.interval,
            "horizon": self.horizon,
            "expected_artifact_file": self.expected_artifact_file,
            "expected_metadata_file": self.expected_metadata_file,
            "training_command": self.training_command,
            "reason": self.reason,
        }


def deep_training_command(model_name: str, interval: str, horizon: int | None = None) -> str:
    resolved_horizon = int(horizon or 30)
    base = (
        "python scripts/train/train_deep_fusion_models.py "
        f"--model {model_name} --interval {interval} --horizon {resolved_horizon} --universe oil_core "
        "--use-processed-data "
        f"--market-panel data/processed/market_panel/{interval}/panel.csv "
        "--oil-fundamentals data/processed/oil_fundamentals/eia_weekly.csv "
        "--cot data/processed/oil_fundamentals/cftc_cot_weekly.csv "
        "--macro-panel data/processed/macro_panel/fred_daily_wide.csv "
        "--event-context data/processed/event_context/event_context_daily.csv "
        "--epochs 10 --batch-size 64 --force"
    )
    if model_name == "oil_context_fusion":
        return f"{base} --llm-context"
    if model_name == "llm_context_seq_moe":
        return f"{base} --llm-context --events-path data/external/events/sample_market_events.csv"
    return f"{base} --no-llm-context"


def _read_raw_metadata(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def deep_artifact_availability(
    *,
    settings: Settings,
    model_name: str,
    interval: str,
    horizon: int,
) -> DeepArtifactAvailability:
    expected_artifact_file = deep_artifact_name(model_name, interval, horizon)
    expected_metadata_file = deep_metadata_name(model_name, interval, horizon)
    artifact_path = Path(settings.model_dir) / expected_artifact_file
    metadata_path = Path(settings.metadata_dir) / expected_metadata_file
    training_command = deep_training_command(model_name, interval, horizon)

    if not artifact_path.exists():
        return DeepArtifactAvailability(
            model_name=model_name,
            interval=interval,
            horizon=horizon,
            status="artifact_missing",
            expected_artifact_file=expected_artifact_file,
            expected_metadata_file=expected_metadata_file,
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            training_command=training_command,
            metadata={},
            reason=f"Missing artifact file: {expected_artifact_file}",
        )

    metadata = metadata_for_artifact(artifact_path, metadata_dir=Path(settings.metadata_dir))
    raw_metadata = _read_raw_metadata(metadata_path)
    status = str(raw_metadata.get("status") or metadata.status or "metadata_missing")
    if raw_metadata.get("synthetic_used") is True and status == PRODUCTION_DEEP_STATUS:
        status = "synthetic_only"
    reason = None
    if status != PRODUCTION_DEEP_STATUS:
        reason = f"Artifact exists but metadata status is {status!r}; production inference requires 'available'."

    return DeepArtifactAvailability(
        model_name=model_name,
        interval=interval,
        horizon=horizon,
        status=status,
        expected_artifact_file=expected_artifact_file,
        expected_metadata_file=expected_metadata_file,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        training_command=training_command,
        metadata={**metadata.model_dump(), **raw_metadata},
        reason=reason,
    )


def production_deep_models(
    *,
    settings: Settings,
    model_names: tuple[str, ...],
    interval: str,
    horizon: int,
) -> list[str]:
    return [
        name
        for name in model_names
        if deep_artifact_availability(settings=settings, model_name=name, interval=interval, horizon=horizon).is_available
    ]
