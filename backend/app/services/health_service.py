from __future__ import annotations

from market_ai.config import Settings
from market_ai.modeling.registry import ModelRegistry


def model_health(settings: Settings) -> dict:
    return ModelRegistry(settings).health()
