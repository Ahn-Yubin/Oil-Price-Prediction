from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from market_ai.config import Settings
from market_ai.schemas.llm_context import ExplanationOutput, LLMContextOutput, MarketContextInput, StructuredEvent


class BaseLLMEventEncoder(ABC):
    @abstractmethod
    def encode_events(self, context: MarketContextInput) -> LLMContextOutput:
        """Encode market context into structured event scores, not numeric price forecasts."""


class NullLLMEventEncoder(BaseLLMEventEncoder):
    def encode_events(self, context: MarketContextInput) -> LLMContextOutput:
        return LLMContextOutput(
            events=[],
            overall_bias="unknown",
            impact_score=0.0,
            uncertainty=1.0,
            event_embedding=[],
            explanation="LLM context encoder is disabled.",
            warnings=["LLM context disabled; using deterministic explanation."],
        )


class MockLLMEventEncoder(BaseLLMEventEncoder):
    def encode_events(self, context: MarketContextInput) -> LLMContextOutput:
        title = context.news[0].title if context.news else "No external event feed configured"
        event = StructuredEvent(
            event_type="market_context",
            affected_assets=[context.symbol],
            directional_bias="neutral",
            impact_strength=0.2 if context.news or context.economic_events else 0.0,
            uncertainty=0.7,
            time_decay=0.5,
            summary=title,
            risk_factors=["Mock encoder output must not alter numeric forecasts."],
        )
        return LLMContextOutput(
            events=[event],
            overall_bias="neutral",
            impact_score=event.impact_strength,
            uncertainty=event.uncertainty,
            event_embedding=[event.impact_strength, event.uncertainty, event.time_decay],
            explanation="Development mock context encoder produced structured context only.",
            warnings=["Mock LLM context does not generate numeric forecasts."],
        )


class OpenAICompatibleLLMEventEncoder(BaseLLMEventEncoder):
    def __init__(self, api_key: str | None, model: str):
        self.api_key = api_key
        self.model = model

    def encode_events(self, context: MarketContextInput) -> LLMContextOutput:
        if not self.api_key:
            return NullLLMEventEncoder().encode_events(context)
        return LLMContextOutput(
            events=[],
            overall_bias="unknown",
            impact_score=0.0,
            uncertainty=1.0,
            event_embedding=[],
            explanation=(
                "OpenAI-compatible LLM adapter is configured as an interface placeholder. "
                "No external call is made in this runtime path."
            ),
            warnings=["External LLM calls are not enabled by this adapter implementation."],
        )


def parse_llm_context_json(raw: str) -> LLMContextOutput:
    try:
        data: Any = json.loads(raw)
        return LLMContextOutput.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        return LLMContextOutput(
            explanation="Failed to parse LLM JSON output; using safe fallback context.",
            warnings=[f"Invalid LLM JSON fallback: {exc}"],
        )


def encoder_from_settings(settings: Settings) -> BaseLLMEventEncoder:
    if not settings.enable_llm_context:
        return NullLLMEventEncoder()
    if settings.is_development and not settings.llm_api_key:
        return MockLLMEventEncoder()
    return OpenAICompatibleLLMEventEncoder(api_key=settings.llm_api_key, model=settings.llm_model)


def deterministic_explanation(
    *,
    symbol: str,
    interval: str,
    forecast_summary: dict[str, Any],
    data_status: dict[str, Any],
    llm_context: LLMContextOutput | None = None,
) -> ExplanationOutput:
    confidence = forecast_summary.get("confidence")
    regime = forecast_summary.get("regime", "unknown")
    data_state = data_status.get("status", "unknown")
    drivers = [
        f"Price-only forecast generated for {symbol} on {interval}.",
        f"Current regime estimate: {regime}.",
        f"Data status: {data_state}.",
    ]
    if llm_context and llm_context.events:
        drivers.append("Structured event context is available but does not overwrite numeric forecasts.")
    warning = None
    if confidence is not None and float(confidence) < 0.45:
        warning = "Model confidence is low; inspect data quality, volatility, and model disagreement."
    if data_state != "real":
        warning = f"Forecast uses {data_state} data status; treat output as degraded."
    return ExplanationOutput(
        symbol=symbol,
        interval=interval,
        generated_at=datetime.now(timezone.utc),
        mode="llm_context" if llm_context and llm_context.events else "deterministic",
        summary="Forecast explanation is derived from model output, data status, and optional structured event context.",
        main_drivers=drivers,
        bull_case="Upside scenario follows upper quantile path if momentum and volatility remain favorable.",
        base_case="Base scenario follows the median price-only time-series forecast.",
        bear_case="Downside scenario follows lower quantile path if volatility rises or trend deteriorates.",
        confidence_warning=warning,
        llm_context=llm_context,
        warnings=list(llm_context.warnings if llm_context else []),
    )
