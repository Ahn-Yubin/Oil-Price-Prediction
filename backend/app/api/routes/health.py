from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from market_ai.config import get_settings
from market_ai.modeling.registry import ModelRegistry


router = APIRouter()


@router.get("/api/health")
def health() -> dict[str, Any]:
    current_settings = get_settings()
    registry = ModelRegistry(current_settings)
    return {
        "status": "ok",
        "app_version": current_settings.app_version,
        "app_env": current_settings.app_env,
        "config": {
            "default_symbol": current_settings.default_symbol,
            "default_interval": current_settings.default_interval,
            "allow_mock_data": current_settings.allow_mock_data,
            "mock_data_enabled": current_settings.mock_data_enabled,
            "data_stale_threshold_seconds": current_settings.data_stale_threshold_seconds,
            "enable_llm_context": current_settings.enable_llm_context,
            "llm_model": current_settings.llm_model,
            "enable_external_features": current_settings.enable_external_features,
            "enable_cross_asset_features": current_settings.enable_cross_asset_features,
        },
        "models": registry.health(),
        "data_provider": {"name": "yfinance", "status": "configured"},
    }
