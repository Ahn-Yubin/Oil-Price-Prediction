from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from market_ai.constants import FALLBACK_INTERVAL, INTERVAL_TO_HORIZON, SUPPORTED_DISPLAY_HORIZONS, SUPPORTED_FORECAST_INTERVALS
from market_ai.config import PROJECT_DIR, get_settings
from market_ai.modeling.deep.availability import deep_artifact_availability
from market_ai.modeling.model_catalog import BACKTEST_ONLY_MODELS, BASELINE_MODELS, CLASSICAL_MODELS, DEEP_MODELS, LEGACY_ARTIFACT_MODELS, REMOVED_LEGACY_MODELS, USER_FACING_MODELS
from market_ai.modeling.registry import ModelArtifactNotFound, ModelRegistry
from market_ai.data.manifests import LATEST_SNAPSHOT_PATH


router = APIRouter()


def _pattern_availability(registry: ModelRegistry, interval: str, horizon: int) -> dict[str, Any]:
    expected = f"global_dl_{interval}_h{horizon}.npz"
    try:
        metadata = registry.resolve(model_name="pattern_mlp", interval=interval, horizon=horizon)
    except ModelArtifactNotFound:
        return {
            "status": "artifact_missing",
            "expected_artifact_file": expected,
            "training_command": (
                f"python scripts/train/train_pretrained_models.py --interval {interval} --horizon {horizon} "
                f"--market-panel data/processed/market_panel/{interval}/panel.csv --force"
            ),
            "reason": f"Missing pattern artifact file: {expected}",
        }
    return {
        "status": metadata.status,
        "artifact_file": metadata.artifact_file,
        "expected_artifact_file": expected,
        "training_command": None,
        "reason": metadata.notes,
    }


@router.get("/api/models")
def models(
    interval: str | None = Query(default=None),
    horizon: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    settings = get_settings()
    registry = ModelRegistry(settings)
    default_interval = interval or settings.default_interval or FALLBACK_INTERVAL
    artifact_horizon = INTERVAL_TO_HORIZON.get(default_interval, INTERVAL_TO_HORIZON[FALLBACK_INTERVAL])
    requested_horizon = horizon or artifact_horizon
    display_horizon = max(1, min(int(requested_horizon), artifact_horizon))
    deep_availability = {
        model_name: deep_artifact_availability(
            settings=settings,
            model_name=model_name,
            interval=default_interval,
            horizon=artifact_horizon,
        )
        for model_name in DEEP_MODELS
    }
    pattern_availability = _pattern_availability(registry, default_interval, artifact_horizon)
    scanned_models = registry.scan()
    return {
        "models": [model.model_dump() for model in scanned_models],
        "logical_models": list(USER_FACING_MODELS),
        "user_facing_models": [
            {
                "id": name,
                "status": deep_availability[name].status
                if name in deep_availability
                else "available",
                "artifact_based": name in DEEP_MODELS,
                **(deep_availability[name].as_api_dict() if name in deep_availability else {}),
            }
            for name in USER_FACING_MODELS
        ],
        "deep_artifact_policy": {
            "default_interval": default_interval,
            "default_horizon": artifact_horizon,
            "artifact_horizon": artifact_horizon,
            "requested_horizon": requested_horizon,
            "display_horizon": display_horizon,
            "display_horizon_options": list(SUPPORTED_DISPLAY_HORIZONS),
            "operating_intervals": sorted(SUPPORTED_FORECAST_INTERVALS),
            "horizon_mode": "single_max_horizon_artifact_sliced_for_display",
            "production_status": "available",
            "non_production_statuses": ["smoke_only", "synthetic_only", "failed", "metadata_only"],
        },
        "data_pipeline_status": {
            "manifest_available": LATEST_SNAPSHOT_PATH.exists(),
            "latest_snapshot_path": str(LATEST_SNAPSHOT_PATH.relative_to(PROJECT_DIR))
            if LATEST_SNAPSHOT_PATH.exists()
            else None,
        },
        "llm_context_status": {
            "enabled": settings.enable_llm_context,
            "external_calls_enabled": settings.enable_external_llm_calls,
            "mode": settings.llm_context_mode,
            "role": "context/event encoder only",
        },
        "internal_benchmark_models": list(CLASSICAL_MODELS + LEGACY_ARTIFACT_MODELS + BASELINE_MODELS),
        "pattern_benchmark_availability": pattern_availability,
        "backtest_only_models": list(BACKTEST_ONLY_MODELS),
        "removed_models": [{"id": name, "reason": reason} for name, reason in REMOVED_LEGACY_MODELS.items()],
    }
