from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from market_ai.config import get_settings
from market_ai.modeling.registry import ModelRegistry


router = APIRouter()


@router.get("/api/models")
def models() -> dict[str, Any]:
    registry = ModelRegistry(get_settings())
    return {
        "models": [model.model_dump() for model in registry.scan()],
        "logical_models": [
            "motif",
            "pattern_mlp",
            "lstm",
            "tcn",
            "cycle",
            "ensemble",
            "flat",
            "drift",
            "random_walk",
            "seasonal_naive",
            "volatility_scaled_naive",
            "simple_moving_average_path",
            "regime_ensemble",
        ],
    }
