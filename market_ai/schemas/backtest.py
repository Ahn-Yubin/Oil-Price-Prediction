from __future__ import annotations

from pydantic import BaseModel, Field


class BacktestSummary(BaseModel):
    model: str
    metrics: dict[str, float] = Field(default_factory=dict)
