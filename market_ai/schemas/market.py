from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AssetClass(str, Enum):
    commodity = "commodity"
    equity = "equity"
    index = "index"
    crypto = "crypto"
    fx = "fx"
    rates = "rates"
    futures = "futures"
    etf = "etf"
    unknown = "unknown"


class DataStatusKind(str, Enum):
    real = "real"
    mock = "mock"
    stale = "stale"
    fallback = "fallback"
    error = "error"


class MarketSymbol(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    requested: str
    normalized: str
    provider: str = "yfinance"
    provider_symbol: str
    asset_class: AssetClass = AssetClass.unknown
    exchange: str | None = None
    root: str | None = None
    description: str | None = None


class Timeframe(BaseModel):
    requested: str
    normalized: str
    provider_interval: str
    provider_period: str
    seconds: int
    is_supported: bool = True
    warning: str | None = None


class Candle(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class DataStatus(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    status: DataStatusKind
    source: str
    symbol_requested: str
    symbol_resolved: str
    interval_requested: str
    interval_resolved: str
    last_bar_time: str | None = None
    updated_at: str
    is_stale: bool = False
    warnings: list[str] = Field(default_factory=list)


class MarketDataWindow(BaseModel):
    symbol: MarketSymbol
    timeframe: Timeframe
    candles: list[Candle]
    data_status: DataStatus


class AssetMetadata(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    symbol: str
    provider_symbol: str
    asset_class: AssetClass
    exchange: str | None = None
    currency: str | None = None
    quote_currency: str | None = None
    description: str | None = None
    roll_policy: str | None = None


class ForecastHorizon(BaseModel):
    steps: int
    interval: str


class ForecastPoint(BaseModel):
    time: int
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    expected_return: float
    expected_volatility: float
    prob_up: float
    confidence: float

    @field_validator("confidence", "prob_up")
    @classmethod
    def clamp_probability(cls, value: float) -> float:
        return min(max(float(value), 0.0), 1.0)


class ScenarioPoint(BaseModel):
    time: int
    value: float


class ScenarioResponse(BaseModel):
    bull: list[ScenarioPoint] = Field(default_factory=list)
    base: list[ScenarioPoint] = Field(default_factory=list)
    bear: list[ScenarioPoint] = Field(default_factory=list)


class RegimeProbabilities(BaseModel):
    trend_up: float = 0.2
    trend_down: float = 0.2
    range: float = 0.3
    high_volatility: float = 0.2
    event_driven: float = 0.1
    confidence: float = 0.5

    @field_validator("trend_up", "trend_down", "range", "high_volatility", "event_driven", "confidence")
    @classmethod
    def clamp(cls, value: float) -> float:
        return min(max(float(value), 0.0), 1.0)

    def normalized(self) -> "RegimeProbabilities":
        values = {
            "trend_up": self.trend_up,
            "trend_down": self.trend_down,
            "range": self.range,
            "high_volatility": self.high_volatility,
            "event_driven": self.event_driven,
        }
        total = sum(values.values())
        if total <= 0.0:
            values = {"trend_up": 0.2, "trend_down": 0.2, "range": 0.3, "high_volatility": 0.2, "event_driven": 0.1}
            total = 1.0
        return RegimeProbabilities(
            **{key: val / total for key, val in values.items()},
            confidence=self.confidence,
        )


class ModelInfo(BaseModel):
    name: str
    label: str | None = None
    model_type: str | None = None
    version: str | None = None
    status: str = "available"
    supported_intervals: list[str] = Field(default_factory=list)
    supported_asset_classes: list[str] = Field(default_factory=list)
    training_cutoff: str | None = None
    feature_version: str | None = None
    artifact_file: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class ModelMetadata(BaseModel):
    model_name: str
    model_type: str
    version: str = "legacy"
    artifact_file: str
    interval: str | None = None
    created_at: str | None = None
    train_start: str | None = None
    train_end: str | None = None
    training_cutoff: str | None = None
    asset_universe: list[str] = Field(default_factory=list)
    supported_asset_classes: list[str] = Field(default_factory=list)
    supported_intervals: list[str] = Field(default_factory=list)
    lookback: int | None = None
    horizon: int | None = None
    target: str | None = None
    feature_set: str | None = None
    scaler: str | None = None
    data_hash: str | None = None
    git_commit: str | None = None
    n_train: int | None = None
    n_val: int | None = None
    n_test: int | None = None
    data_source: str | None = None
    synthetic_used: bool | None = None
    event_context_enabled: bool | None = None
    events_path: list[str] = Field(default_factory=list)
    related_assets_enabled: bool | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    status: str = "available"
    deep_config: dict[str, Any] = Field(default_factory=dict)


class ForecastWarning(BaseModel):
    code: str
    severity: str = "warning"
    message: str
    action: str | None = None


class ForecastResponse(BaseModel):
    symbol: str
    asset_metadata: AssetMetadata
    interval: str
    generated_at: datetime
    current_price: float
    model_version: str | None = None
    training_cutoff: str | None = None
    data_status: DataStatus
    candles: list[Candle] = Field(default_factory=list)
    forecast: list[ForecastPoint]
    scenarios: ScenarioResponse = Field(default_factory=ScenarioResponse)
    regime: RegimeProbabilities = Field(default_factory=RegimeProbabilities)
    models: list[ModelInfo] = Field(default_factory=list)
    cross_asset_context: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    warning_objects: list[ForecastWarning] = Field(default_factory=list)
    model_paths: list[dict[str, Any]] = Field(default_factory=list)
    selected_models: list[str] = Field(default_factory=list)
    primary_model: str | None = None
    deprecated_models_requested: list[str] = Field(default_factory=list)
    removed_models_requested: list[str] = Field(default_factory=list)
    llm_context_summary: dict[str, Any] = Field(default_factory=dict)
    deep_model_info: dict[str, Any] = Field(default_factory=dict)
    feature_version: str | None = None
    artifact_status: dict[str, str] = Field(default_factory=dict)
    calibration_status: dict[str, Any] = Field(default_factory=dict)
    band_explanation: dict[str, Any] = Field(default_factory=dict)
