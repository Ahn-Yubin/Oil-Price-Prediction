from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from market_ai.config import get_settings
from market_ai.modeling.model_catalog import BACKTEST_ONLY_MODELS, REMOVED_LEGACY_MODELS, USER_FACING_MODELS
from market_ai.modeling.registry import ModelRegistry


router = APIRouter()


@router.get("/api/models")
def models() -> dict[str, Any]:
    registry = ModelRegistry(get_settings())
    artifact_names = {model.model_name for model in registry.scan()}
    return {
        "models": [model.model_dump() for model in registry.scan()],
        "logical_models": list(USER_FACING_MODELS),
        "user_facing_models": [
            {
                "id": name,
                "status": "available" if name not in {"deep_lstm_tcn_fusion", "llm_context_seq_moe"} or name in artifact_names else "artifact_missing",
                "artifact_based": name in {"pattern_mlp", "deep_lstm_tcn_fusion", "llm_context_seq_moe"},
            }
            for name in USER_FACING_MODELS
        ],
        "backtest_only_models": list(BACKTEST_ONLY_MODELS),
        "removed_models": [{"id": name, "reason": reason} for name, reason in REMOVED_LEGACY_MODELS.items()],
    }
