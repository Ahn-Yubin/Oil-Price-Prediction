from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from market_ai.data.event_providers import FileEventProvider
from market_ai.config import Settings
from market_ai.schemas.llm_context import ExplanationOutput, LLMContextOutput, MarketContextInput, StructuredEvent

try:
    import certifi
except Exception:  # pragma: no cover - certifi is declared in requirements.
    certifi = None

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


def _default_https_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _read_json_request(request: urllib.request.Request, *, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=_default_https_context()) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_text(raw: str) -> str:
    text = _strip_json_fences(raw)
    if text.startswith("{") or text.startswith("["):
        return text
    starts = [idx for idx in [text.find("{"), text.find("[")] if idx >= 0]
    if not starts:
        return text
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end > start:
        return text[start : end + 1]
    return text


def _normalize_bias(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text in {"bull", "bullish", "positive", "up", "upside"}:
        return "bullish"
    if text in {"bear", "bearish", "negative", "down", "downside"}:
        return "bearish"
    if text in {"neutral", "flat", "none"}:
        return "neutral"
    if text in {"mixed", "two-sided", "two_sided"}:
        return "mixed"
    return "unknown"


def _coerce_score(value: Any, default: float) -> float:
    if isinstance(value, str):
        text = value.strip().lower()
        qualitative = {
            "very low": 0.1,
            "low": 0.25,
            "medium": 0.5,
            "moderate": 0.5,
            "high": 0.75,
            "very high": 0.9,
            "short-term": 0.8,
            "short term": 0.8,
            "medium-term": 0.5,
            "medium term": 0.5,
            "long-term": 0.25,
            "long term": 0.25,
        }
        if text in qualitative:
            return qualitative[text]
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return min(max(score, 0.0), 1.0)


def _sanitize_llm_context_data(data: Any) -> Any:
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        return data
    cleaned = dict(data)
    cleaned["overall_bias"] = _normalize_bias(cleaned.get("overall_bias"))
    cleaned["impact_score"] = _coerce_score(cleaned.get("impact_score"), 0.0)
    cleaned["uncertainty"] = _coerce_score(cleaned.get("uncertainty"), 1.0)
    events = []
    for event in cleaned.get("events") or []:
        if not isinstance(event, dict):
            continue
        item = dict(event)
        item["directional_bias"] = _normalize_bias(item.get("directional_bias"))
        item["impact_strength"] = _coerce_score(item.get("impact_strength"), 0.0)
        item["uncertainty"] = _coerce_score(item.get("uncertainty"), 1.0)
        item["time_decay"] = _coerce_score(item.get("time_decay"), 1.0)
        item.setdefault("affected_assets", [])
        item.setdefault("risk_factors", [])
        item.setdefault("summary", "")
        item.setdefault("event_type", "market_event")
        events.append(item)
    cleaned["events"] = events
    if not isinstance(cleaned.get("event_embedding"), list):
        cleaned["event_embedding"] = []
    if len(cleaned["event_embedding"]) < 13:
        cleaned["event_embedding"] = _embedding_from_structured_context(cleaned)
    if not isinstance(cleaned.get("warnings"), list):
        cleaned["warnings"] = [str(cleaned["warnings"])] if cleaned.get("warnings") else []
    return cleaned


def _embedding_from_structured_context(data: dict[str, Any]) -> list[float]:
    events = [event for event in data.get("events") or [] if isinstance(event, dict)]
    event_count = float(len(events))
    impact = _coerce_score(data.get("impact_score"), 0.0)
    uncertainty = _coerce_score(data.get("uncertainty"), 1.0)
    if not events:
        return [
            _bias_numeric(data.get("overall_bias")),
            impact,
            uncertainty,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    weighted_bias = 0.0
    bullish = 0.0
    bearish = 0.0
    macro = 0.0
    energy = 0.0
    geo = 0.0
    total_weight = 0.0
    time_decay = 0.0
    for event in events:
        event_impact = _coerce_score(event.get("impact_strength"), impact)
        event_decay = _coerce_score(event.get("time_decay"), 1.0)
        weight = max(event_impact, 0.05) * max(event_decay, 0.05)
        bias = _bias_numeric(event.get("directional_bias"))
        event_type = str(event.get("event_type") or "").lower()
        total_weight += weight
        weighted_bias += bias * weight
        bullish += max(bias, 0.0) * weight
        bearish += max(-bias, 0.0) * weight
        time_decay += event_decay
        if "macro" in event_type or "economic" in event_type or "policy" in event_type:
            macro += weight
        if "energy" in event_type or "oil" in event_type or "supply" in event_type or "demand" in event_type:
            energy += weight
        if "geo" in event_type or "war" in event_type or "sanction" in event_type:
            geo += weight
    denom = max(total_weight, 1e-8)
    return [
        float(weighted_bias / denom),
        impact,
        uncertainty,
        min(float(time_decay / max(event_count, 1.0)), 1.0),
        min(event_count, 1.0),
        min(event_count, 3.0),
        min(event_count, 7.0),
        float(bullish / denom),
        float(bearish / denom),
        float(macro / denom),
        float(energy / denom),
        float(geo / denom),
        0.65,
    ]


def _bias_numeric(value: Any) -> float:
    normalized = _normalize_bias(value)
    if normalized == "bullish":
        return 1.0
    if normalized == "bearish":
        return -1.0
    return 0.0


class BaseLLMEventEncoder(ABC):
    @abstractmethod
    def encode_events(self, context: MarketContextInput) -> LLMContextOutput:
        """Encode market context into structured event scores, not numeric price forecasts."""

    def encode_event_batch(self, contexts: list[MarketContextInput]) -> list[LLMContextOutput]:
        return [self.encode_events(context) for context in contexts]


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
        timeout: float = 20.0,
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
            fallback = LocalEventContextEncoder(self.fallback_provider).encode_events(context)
            reason = (
                "External LLM API key missing; local_rules fallback used."
                if self.enabled
                else "External LLM disabled; local_rules fallback used."
            )
            return fallback.model_copy(update={"warnings": [*fallback.warnings, reason]})
        if self.model.lower().startswith("gemma-") and "generativelanguage.googleapis.com" in self.api_base:
            return GoogleGenerativeLLMEventEncoder(
                api_key=self.api_key,
                model=self.model,
                enabled=self.enabled,
                api_base="https://generativelanguage.googleapis.com/v1beta",
                timeout=self.timeout,
                fallback_provider=self.fallback_provider,
            ).encode_events(context)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a market context/event encoder. Do not output price targets. "
                        "Do not output p50/p90 price. Do not output direct future return path. "
                        "Only produce structured context and explanation. "
                        "Return valid JSON with keys: events, overall_bias, impact_score, uncertainty, "
                        "event_embedding, explanation, warnings. Each event must include event_type, "
                        "affected_assets, directional_bias, impact_strength, uncertainty, time_decay, "
                        "summary, risk_factors. Use lowercase directional_bias values and numeric "
                        "0.0-1.0 floats for impact_strength, uncertainty, time_decay, and impact_score."
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
                body = _read_json_request(request, timeout=self.timeout)
                content = body["choices"][0]["message"]["content"]
                return parse_llm_context_json(content)
            except (KeyError, RuntimeError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                time.sleep(0.15 * (attempt + 1))
        fallback = LocalEventContextEncoder(self.fallback_provider).encode_events(context)
        return fallback.model_copy(update={"warnings": [*fallback.warnings, f"External LLM fallback: {last_error}"]})


class GoogleGenerativeLLMEventEncoder(BaseLLMEventEncoder):
    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        enabled: bool = False,
        api_base: str | None = None,
        timeout: float = 60.0,
        fallback_provider: FileEventProvider | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.enabled = enabled
        self.api_base = api_base or "https://generativelanguage.googleapis.com/v1beta"
        self.timeout = timeout
        self.fallback_provider = fallback_provider

    def _url(self) -> str:
        base = self.api_base.strip().rstrip("/")
        if "openai" in base or "chat/completions" in base:
            base = "https://generativelanguage.googleapis.com/v1beta"
        if ":generateContent" in base:
            url = base
        else:
            encoded_model = urllib.parse.quote(self.model, safe="-_.~/")
            url = f"{base}/models/{encoded_model}:generateContent"
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}key={urllib.parse.quote(self.api_key or '')}"

    def _prompt(self, context: MarketContextInput) -> str:
        return (
            "You are a market context/event encoder. Do not output price targets, p50/p90 prices, "
            "or future return paths. Return only valid JSON with keys: events, overall_bias, "
            "impact_score, uncertainty, event_embedding, explanation, warnings. Each event must "
            "include event_type, affected_assets, directional_bias, impact_strength, uncertainty, "
            "time_decay, summary, risk_factors. Use lowercase directional_bias values and numeric "
            "0.0-1.0 floats for impact_strength, uncertainty, time_decay, and impact_score.\n\n"
            f"MarketContextInput JSON:\n{context.model_dump_json()}"
        )

    def _batch_prompt(self, contexts: list[MarketContextInput]) -> str:
        payload = [json.loads(context.model_dump_json()) for context in contexts]
        return (
            "You are a market context/event encoder. Do not output price targets, p50/p90 prices, "
            "or future return paths. Return only a valid JSON array with exactly one output object "
            "for each MarketContextInput, in the same order. Each output object must have keys: "
            "events, overall_bias, impact_score, uncertainty, event_embedding, explanation, warnings. "
            "Each event must include event_type, affected_assets, directional_bias, impact_strength, "
            "uncertainty, time_decay, summary, risk_factors. Use lowercase directional_bias values and "
            "numeric 0.0-1.0 floats for impact_strength, uncertainty, time_decay, and impact_score.\n\n"
            f"MarketContextInput JSON array:\n{json.dumps(payload, ensure_ascii=False)}"
        )

    def encode_events(self, context: MarketContextInput) -> LLMContextOutput:
        if not self.enabled or not self.api_key:
            fallback = LocalEventContextEncoder(self.fallback_provider).encode_events(context)
            reason = (
                "External LLM API key missing; local_rules fallback used."
                if self.enabled
                else "External LLM disabled; local_rules fallback used."
            )
            return fallback.model_copy(update={"warnings": [*fallback.warnings, reason]})

        base_payload = {
            "contents": [{"role": "user", "parts": [{"text": self._prompt(context)}]}],
        }
        payloads = [
            {
                **base_payload,
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json",
                },
            },
            {**base_payload, "generationConfig": {"temperature": 0.0}},
        ]
        last_error: Exception | None = None
        for payload in payloads:
            try:
                request = urllib.request.Request(
                    self._url(),
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                body = _read_json_request(request, timeout=self.timeout)
                parts = body["candidates"][0]["content"]["parts"]
                content = "".join(str(part.get("text", "")) for part in parts if not part.get("thought"))
                return parse_llm_context_json(content)
            except (KeyError, RuntimeError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                time.sleep(0.15)
        fallback = LocalEventContextEncoder(self.fallback_provider).encode_events(context)
        return fallback.model_copy(update={"warnings": [*fallback.warnings, f"External LLM fallback: {last_error}"]})

    def encode_event_batch(self, contexts: list[MarketContextInput]) -> list[LLMContextOutput]:
        if not contexts:
            return []
        if len(contexts) == 1:
            return [self.encode_events(contexts[0])]
        if not self.enabled or not self.api_key:
            outputs = [LocalEventContextEncoder(self.fallback_provider).encode_events(context) for context in contexts]
            reason = (
                "External LLM API key missing; local_rules fallback used."
                if self.enabled
                else "External LLM disabled; local_rules fallback used."
            )
            return [output.model_copy(update={"warnings": [*output.warnings, reason]}) for output in outputs]

        base_payload = {
            "contents": [{"role": "user", "parts": [{"text": self._batch_prompt(contexts)}]}],
        }
        payloads = [
            {
                **base_payload,
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json",
                },
            },
            {**base_payload, "generationConfig": {"temperature": 0.0}},
        ]
        last_error: Exception | None = None
        for payload in payloads:
            try:
                request = urllib.request.Request(
                    self._url(),
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                body = _read_json_request(request, timeout=self.timeout)
                parts = body["candidates"][0]["content"]["parts"]
                content = "".join(str(part.get("text", "")) for part in parts if not part.get("thought"))
                return parse_llm_context_json_array(content, expected_count=len(contexts))
            except (KeyError, RuntimeError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError, ValidationError, TypeError, ValueError) as exc:
                last_error = exc
                time.sleep(0.15)
        outputs = [LocalEventContextEncoder(self.fallback_provider).encode_events(context) for context in contexts]
        return [output.model_copy(update={"warnings": [*output.warnings, f"External LLM fallback: {last_error}"]}) for output in outputs]


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
            body = _read_json_request(request, timeout=self.timeout)
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
        data: Any = _sanitize_llm_context_data(json.loads(_extract_json_text(raw)))
        return validate_llm_context_output(LLMContextOutput.model_validate(data), raw=data)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        return LLMContextOutput(
            explanation="Failed to parse LLM JSON output; using safe fallback context.",
            warnings=[f"Invalid LLM JSON fallback: {exc}"],
        )


def parse_llm_context_json_array(raw: str, *, expected_count: int) -> list[LLMContextOutput]:
    data: Any = json.loads(_extract_json_text(raw))
    if isinstance(data, dict):
        for key in ("outputs", "results", "items", "contexts"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise TypeError("LLM batch output must be a JSON array")
    if len(data) != expected_count:
        raise ValueError(f"LLM batch output length {len(data)} != expected {expected_count}")
    outputs: list[LLMContextOutput] = []
    for item in data:
        cleaned = _sanitize_llm_context_data(item)
        outputs.append(validate_llm_context_output(LLMContextOutput.model_validate(cleaned), raw=cleaned))
    return outputs


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
    if mode == "google_generative":
        return GoogleGenerativeLLMEventEncoder(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            enabled=settings.enable_external_llm_calls,
            api_base=settings.llm_api_base,
        )
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
