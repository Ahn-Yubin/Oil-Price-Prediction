from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from market_ai.constants import FALLBACK_INTERVAL, INTERVAL_TO_HORIZON
from market_ai.config import PROJECT_DIR, get_settings
from market_ai.modeling.deep.availability import deep_artifact_availability
from market_ai.modeling.model_catalog import BACKTEST_ONLY_MODELS, REMOVED_LEGACY_MODELS, USER_FACING_MODELS
from market_ai.modeling.registry import ModelRegistry
from market_ai.data.manifests import LATEST_SNAPSHOT_PATH


router = APIRouter()


@router.get("/api/models")
def models() -> dict[str, Any]:
    settings = get_settings()
    registry = ModelRegistry(settings)
    default_interval = settings.default_interval or FALLBACK_INTERVAL
    default_horizon = INTERVAL_TO_HORIZON.get(default_interval, INTERVAL_TO_HORIZON[FALLBACK_INTERVAL])
    deep_availability = {
        model_name: deep_artifact_availability(
            settings=settings,
            model_name=model_name,
            interval=default_interval,
            horizon=default_horizon,
        )
        for model_name in {"deep_lstm_tcn_fusion", "llm_context_seq_moe"}
    }
    scanned_models = registry.scan()
    return {
        "models": [model.model_dump() for model in scanned_models],
        "logical_models": list(USER_FACING_MODELS),
        "user_facing_models": [
            {
                "id": name,
                "status": deep_availability[name].status if name in deep_availability else "available",
                "artifact_based": name in {"pattern_mlp", "deep_lstm_tcn_fusion", "llm_context_seq_moe"},
                **(deep_availability[name].as_api_dict() if name in deep_availability else {}),
            }
            for name in USER_FACING_MODELS
        ],
        "deep_artifact_policy": {
            "default_interval": default_interval,
            "default_horizon": default_horizon,
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
        "backtest_only_models": list(BACKTEST_ONLY_MODELS),
        "removed_models": [{"id": name, "reason": reason} for name, reason in REMOVED_LEGACY_MODELS.items()],
    }
