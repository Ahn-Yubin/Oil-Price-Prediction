from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


DirectionalBias = Literal["bullish", "bearish", "neutral", "mixed", "unknown"]


class RawNewsItem(BaseModel):
    title: str
    source: str | None = None
    published_at: datetime | None = None
    url: str | None = None
    text: str | None = None


class EconomicEvent(BaseModel):
    name: str
    country: str | None = None
    scheduled_at: datetime | None = None
    actual: str | None = None
    consensus: str | None = None
    previous: str | None = None
    importance: str | None = None


class MarketContextInput(BaseModel):
    symbol: str
    interval: str
    asset_class: str = "unknown"
    generated_at: datetime | None = None
    news: list[RawNewsItem] = Field(default_factory=list)
    economic_events: list[EconomicEvent] = Field(default_factory=list)
    data_status: dict = Field(default_factory=dict)
    forecast_summary: dict = Field(default_factory=dict)


class StructuredEvent(BaseModel):
    event_type: str
    affected_assets: list[str] = Field(default_factory=list)
    directional_bias: DirectionalBias = "unknown"
    impact_strength: float = 0.0
    uncertainty: float = 1.0
    time_decay: float = 1.0
    summary: str
    risk_factors: list[str] = Field(default_factory=list)

    @field_validator("impact_strength", "uncertainty", "time_decay")
    @classmethod
    def clamp_probability_like(cls, value: float) -> float:
        return min(max(float(value), 0.0), 1.0)


class LLMContextOutput(BaseModel):
    events: list[StructuredEvent] = Field(default_factory=list)
    overall_bias: DirectionalBias = "unknown"
    impact_score: float = 0.0
    uncertainty: float = 1.0
    event_embedding: list[float] = Field(default_factory=list)
    explanation: str = ""
    warnings: list[str] = Field(default_factory=list)

    @field_validator("impact_score", "uncertainty")
    @classmethod
    def clamp_scores(cls, value: float) -> float:
        return min(max(float(value), 0.0), 1.0)


class ExplanationOutput(BaseModel):
    symbol: str
    interval: str
    generated_at: datetime
    mode: str
    summary: str
    main_drivers: list[str] = Field(default_factory=list)
    bull_case: str
    base_case: str
    bear_case: str
    confidence_warning: str | None = None
    llm_context: LLMContextOutput | None = None
    warnings: list[str] = Field(default_factory=list)
