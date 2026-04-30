from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from market_ai.data.event_providers import FileEventProvider
from market_ai.config import Settings
from market_ai.schemas.llm_context import ExplanationOutput, LLMContextOutput, MarketContextInput, StructuredEvent

FORBIDDEN_NUMERIC_FORECAST_PATTERNS = (
    re.compile(r"\btarget[_\s-]?price\b", re.IGNORECASE),
    re.compile(r"\bp50\b", re.IGNORECASE),
    re.compile(r"\bp90\b", re.IGNORECASE),
    re.compile(r"\bfuture[_\s-]?price[_\s-]?path\b", re.IGNORECASE),
    re.compile(r"\bprice[_\s-]?target\b", re.IGNORECASE),
    re.compile(r"\breturn[_\s-]?path\b", re.IGNORECASE),
)


def _contains_forbidden_numeric_forecast(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_forbidden_numeric_forecast(k) or _contains_forbidden_numeric_forecast(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_numeric_forecast(item) for item in value)
    text = str(value)
    return any(pattern.search(text) for pattern in FORBIDDEN_NUMERIC_FORECAST_PATTERNS)


def validate_llm_context_output(output: LLMContextOutput, raw: Any | None = None) -> LLMContextOutput:
    warnings = list(output.warnings)
    if raw is not None and _contains_forbidden_numeric_forecast(raw):
        warnings.append("LLM output contained forbidden numeric forecast fields; structured context was rejected.")
        return LLMContextOutput(
            events=[],
            overall_bias="unknown",
            impact_score=0.0,
            uncertainty=1.0,
            event_embedding=[],
            explanation="Structured context rejected because the LLM output attempted to include numeric forecast fields.",
            warnings=warnings,
        )
    if _contains_forbidden_numeric_forecast(output.explanation):
        output = output.model_copy(update={"explanation": "Structured context only; numeric forecast text was removed."})
        warnings.append("LLM explanation contained forbidden numeric forecast text and was replaced.")
    return output.model_copy(update={"warnings": warnings})


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


class LocalEventContextEncoder(BaseLLMEventEncoder):
    def __init__(self, event_provider: FileEventProvider | None = None):
        self.event_provider = event_provider or FileEventProvider.from_env()

    def encode_events(self, context: MarketContextInput) -> LLMContextOutput:
        as_of = context.generated_at or datetime.now(timezone.utc)
        vector = self.event_provider.context_vector(symbol=context.symbol, as_of_time=as_of)
        events = self.event_provider.events_as_of(symbol=context.symbol, as_of_time=as_of)
        structured: list[StructuredEvent] = []
        for event in events[-12:]:
            structured.append(
                StructuredEvent(
                    event_type=event.event_type,
                    affected_assets=[event.symbol or context.symbol],
                    directional_bias=event.directional_bias if event.directional_bias in {"bullish", "bearish", "neutral", "mixed", "unknown"} else "unknown",
                    impact_strength=event.impact_strength,
                    uncertainty=event.uncertainty,
                    time_decay=vector.time_decay,
                    summary=event.summary or event.event_type,
                    risk_factors=[],
                )
            )
        bias = "neutral"
        if vector.directional_bias_score > 0.15:
            bias = "bullish"
        elif vector.directional_bias_score < -0.15:
            bias = "bearish"
        elif vector.event_count_7d > 0:
            bias = "mixed"
        return LLMContextOutput(
            events=structured,
            overall_bias=bias,
            impact_score=min(max(vector.impact_strength, 0.0), 1.0),
            uncertainty=min(max(vector.uncertainty, 0.0), 1.0),
            event_embedding=vector.as_list(),
            explanation="Deterministic local event context encoder produced structured context only.",
            warnings=[] if structured else ["No point-in-time event records available; using zero context."],
        )


class OpenAICompatibleLLMEventEncoder(BaseLLMEventEncoder):
    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        enabled: bool = False,
        api_base: str | None = None,
        timeout: float = 8.0,
        fallback_provider: FileEventProvider | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.enabled = enabled
        self.api_base = api_base or "https://api.openai.com/v1/chat/completions"
        self.timeout = timeout
        self.fallback_provider = fallback_provider

    def encode_events(self, context: MarketContextInput) -> LLMContextOutput:
        if not self.enabled or not self.api_key:
            return LocalEventContextEncoder(self.fallback_provider).encode_events(context)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a market context/event encoder. Do not output price targets. "
                        "Do not output p50/p90 price. Do not output direct future return path. "
                        "Only produce structured context and explanation."
                    ),
                },
                {"role": "user", "content": context.model_dump_json()},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                request = urllib.request.Request(
                    self.api_base,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                return parse_llm_context_json(content)
            except (KeyError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                time.sleep(0.15 * (attempt + 1))
        fallback = LocalEventContextEncoder(self.fallback_provider).encode_events(context)
        return fallback.model_copy(update={"warnings": [*fallback.warnings, f"External LLM fallback: {last_error}"]})


class LocalHTTPLLMEventEncoder(BaseLLMEventEncoder):
    def __init__(
        self,
        api_base: str,
        model: str,
        *,
        enabled: bool = False,
        timeout: float = 8.0,
        fallback_provider: FileEventProvider | None = None,
    ):
        self.api_base = api_base
        self.model = model
        self.enabled = enabled
        self.timeout = timeout
        self.fallback_provider = fallback_provider

    def encode_events(self, context: MarketContextInput) -> LLMContextOutput:
        if not self.enabled:
            fallback = LocalEventContextEncoder(self.fallback_provider).encode_events(context)
            return fallback.model_copy(update={"warnings": [*fallback.warnings, "Local HTTP LLM dry-run; local_rules fallback used."]})
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Encode market events as structured context only. "
                        "Do not output price targets, p50/p90 prices, or future return paths."
                    ),
                },
                {"role": "user", "content": context.model_dump_json()},
            ],
            "stream": False,
            "temperature": 0.0,
        }
        try:
            request = urllib.request.Request(
                self.api_base,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = (
                body.get("choices", [{}])[0].get("message", {}).get("content")
                or body.get("message", {}).get("content")
                or body.get("response")
                or body.get("content")
            )
            return parse_llm_context_json(str(content or "{}"))
        except Exception as exc:
            fallback = LocalEventContextEncoder(self.fallback_provider).encode_events(context)
            return fallback.model_copy(update={"warnings": [*fallback.warnings, f"Local HTTP LLM fallback: {exc}"]})


class OfflineFileLLMEventEncoder(BaseLLMEventEncoder):
    def __init__(self, path: str | Any):
        from pathlib import Path

        self.path = Path(path)
        self._cache: dict[tuple[str, str], LLMContextOutput] | None = None

    def _load(self) -> dict[tuple[str, str], LLMContextOutput]:
        if self._cache is not None:
            return self._cache
        cache: dict[tuple[str, str], LLMContextOutput] = {}
        if not self.path.exists():
            self._cache = cache
            return cache
        lines = self.path.read_text(encoding="utf-8").splitlines() if self.path.suffix == ".jsonl" else [self.path.read_text(encoding="utf-8")]
        for line in lines:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                symbol = str(raw.get("symbol") or raw.get("context", {}).get("symbol") or "")
                as_of = str(raw.get("as_of_time") or raw.get("generated_at") or "")
                output = raw.get("output", raw)
                cache[(symbol, as_of[:10])] = validate_llm_context_output(LLMContextOutput.model_validate(output), raw=output)
            except Exception:
                continue
        self._cache = cache
        return cache

    def encode_events(self, context: MarketContextInput) -> LLMContextOutput:
        as_of = (context.generated_at or datetime.now(timezone.utc)).date().isoformat()
        cached = self._load().get((context.symbol, as_of))
        if cached is not None:
            return cached
        fallback = LocalEventContextEncoder().encode_events(context)
        return fallback.model_copy(update={"warnings": [*fallback.warnings, "Offline LLM context cache miss; local_rules fallback used."]})


def parse_llm_context_json(raw: str) -> LLMContextOutput:
    try:
        data: Any = json.loads(raw)
        return validate_llm_context_output(LLMContextOutput.model_validate(data), raw=data)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        return LLMContextOutput(
            explanation="Failed to parse LLM JSON output; using safe fallback context.",
            warnings=[f"Invalid LLM JSON fallback: {exc}"],
        )


def encoder_from_settings(settings: Settings) -> BaseLLMEventEncoder:
    if not settings.enable_llm_context:
        return NullLLMEventEncoder()
    mode = (settings.llm_context_mode or "local_rules").strip().lower()
    if mode == "none":
        return NullLLMEventEncoder()
    if mode == "local_http":
        return LocalHTTPLLMEventEncoder(
            api_base=settings.local_llm_api_base,
            model=settings.local_llm_model,
            enabled=settings.enable_external_llm_calls,
        )
    if mode == "local_rules":
        return LocalEventContextEncoder()
    if settings.is_development and not settings.llm_api_key:
        return LocalEventContextEncoder()
    return OpenAICompatibleLLMEventEncoder(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        enabled=settings.enable_external_llm_calls,
        api_base=settings.llm_api_base,
    )


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
