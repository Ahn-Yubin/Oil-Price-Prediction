from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EventContextVector(BaseModel):
    directional_bias_score: float = 0.0
    impact_strength: float = 0.0
    uncertainty: float = 1.0
    time_decay: float = 0.0
    event_count_1d: float = 0.0
    event_count_3d: float = 0.0
    event_count_7d: float = 0.0
    bullish_event_score: float = 0.0
    bearish_event_score: float = 0.0
    macro_event_score: float = 0.0
    energy_event_score: float = 0.0
    geopolitical_event_score: float = 0.0
    source_quality_score: float = 0.0
    news_volume_1d: float = 0.0
    news_volume_3d: float = 0.0
    news_volume_7d: float = 0.0
    news_volume_30d: float = 0.0
    news_selection_coverage: float = 0.0
    raw_bullish_pressure: float = 0.0
    raw_bearish_pressure: float = 0.0
    raw_net_pressure: float = 0.0
    raw_energy_pressure: float = 0.0
    raw_geopolitical_pressure: float = 0.0
    raw_macro_pressure: float = 0.0
    raw_supply_pressure: float = 0.0
    raw_demand_pressure: float = 0.0
    source_diversity_score: float = 0.0

    @field_validator("*")
    @classmethod
    def finite_number(cls, value: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return 0.0
        if out != out or out in {float("inf"), float("-inf")}:
            return 0.0
        return out

    def as_list(self) -> list[float]:
        return [float(getattr(self, name)) for name in type(self).model_fields]


class DeepLearningSample(BaseModel):
    symbol: str
    interval: str
    as_of_time: datetime
    lookback: int
    horizon: int
    x_price: list[list[float]]
    x_cross_asset: list[list[float]]
    x_event_context: list[float]
    x_static: list[float]
    y_vol_scaled_cum_return: list[float]
    y_direction: list[int]
    y_future_volatility: list[float]
    current_price: float
    recent_realized_volatility: float
    feature_version: str
    data_status: dict[str, Any] = Field(default_factory=dict)


class DeepDatasetConfig(BaseModel):
    interval: str = "1d"
    lookback: int = 128
    horizon: int = 45
    symbols: list[str] = Field(default_factory=lambda: ["CL=F"])
    related_assets_enabled: bool = True
    llm_context_enabled: bool = True
    event_context_enabled: bool = True
    max_samples: int | None = None
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    min_history: int = 160
    seed: int = 42

    @field_validator("validation_ratio", "test_ratio")
    @classmethod
    def valid_ratio(cls, value: float) -> float:
        return min(max(float(value), 0.0), 0.45)

    @field_validator("lookback", "horizon", "min_history")
    @classmethod
    def positive_int(cls, value: int) -> int:
        return max(1, int(value))
