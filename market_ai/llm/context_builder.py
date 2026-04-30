from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from market_ai.data.event_providers import FileEventProvider, MarketEvent
from market_ai.data.news_events import events_to_frame, load_event_sources
from market_ai.features.context_features import EVENT_CONTEXT_DIM
from market_ai.llm.event_encoder import (
    LocalEventContextEncoder,
    LocalHTTPLLMEventEncoder,
    NullLLMEventEncoder,
    OfflineFileLLMEventEncoder,
    OpenAICompatibleLLMEventEncoder,
)
from market_ai.schemas.deep_learning import EventContextVector
from market_ai.schemas.llm_context import MarketContextInput


def _date_bounds(events: list[MarketEvent], start: str | None, end: str | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    if start:
        start_ts = pd.to_datetime(start, utc=True)
    elif events:
        start_ts = pd.Timestamp(min(event.timestamp for event in events)).floor("D")
    else:
        start_ts = pd.Timestamp(datetime.now(timezone.utc)).floor("D")
    if end:
        end_ts = pd.to_datetime(end, utc=True)
    elif events:
        end_ts = pd.Timestamp(max(event.timestamp for event in events)).ceil("D")
    else:
        end_ts = start_ts
    return start_ts, end_ts


def encoder_for_mode(
    mode: str,
    *,
    provider: FileEventProvider,
    api_key: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
    live: bool = False,
    offline_file: str | Path | None = None,
):
    normalized = mode.strip().lower()
    if normalized == "none":
        return NullLLMEventEncoder()
    if normalized == "local_rules":
        return LocalEventContextEncoder(provider)
    if normalized == "openai_compatible":
        return OpenAICompatibleLLMEventEncoder(
            api_key=api_key,
            model=model or "context-encoder",
            api_base=api_base,
            enabled=live,
            fallback_provider=provider,
        )
    if normalized == "local_http":
        return LocalHTTPLLMEventEncoder(
            api_base=api_base or "http://localhost:11434/api/chat",
            model=model or "local-context-encoder",
            enabled=live,
            fallback_provider=provider,
        )
    if normalized == "offline_file":
        if not offline_file:
            raise ValueError("offline_file mode requires --offline-file")
        return OfflineFileLLMEventEncoder(offline_file)
    raise ValueError(f"Unsupported LLM context mode: {mode}")


def build_event_context_daily(
    *,
    symbols: list[str],
    events_paths: list[str | Path] | None = None,
    news_paths: list[str | Path] | None = None,
    mode: str = "local_rules",
    start: str | None = None,
    end: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
    live: bool = False,
    offline_file: str | Path | None = None,
    cache_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = load_event_sources(events_paths=events_paths, news_paths=news_paths)
    raw_frame = events_to_frame(events)
    provider = FileEventProvider(paths=[])
    provider._cache = events
    encoder = encoder_for_mode(
        mode,
        provider=provider,
        api_key=api_key,
        api_base=api_base,
        model=model,
        live=live,
        offline_file=offline_file,
    )
    start_ts, end_ts = _date_bounds(events, start, end)
    rows: list[dict] = []
    cache_rows: list[dict] = []
    vector_names = list(EventContextVector.model_fields)
    for symbol in symbols:
        for as_of in pd.date_range(start_ts.floor("D"), end_ts.floor("D"), freq="D", tz="UTC"):
            context = MarketContextInput(symbol=symbol, interval="1d", generated_at=as_of.to_pydatetime())
            encoded = encoder.encode_events(context)
            embedding = list(encoded.event_embedding or [])
            if len(embedding) < EVENT_CONTEXT_DIM:
                embedding = embedding + [0.0] * (EVENT_CONTEXT_DIM - len(embedding))
            row = {
                "timestamp": as_of.isoformat(),
                "symbol": symbol,
                "feature_available_at": as_of.isoformat(),
                "llm_mode": mode,
                "overall_bias": encoded.overall_bias,
                "impact_score": encoded.impact_score,
                "uncertainty": encoded.uncertainty,
                "event_count": len(encoded.events),
                "explanation": encoded.explanation,
                "warnings": "|".join(encoded.warnings),
            }
            for idx, name in enumerate(vector_names):
                row[name] = float(embedding[idx])
            rows.append(row)
            cache_rows.append(
                {
                    "as_of_time": as_of.isoformat(),
                    "symbol": symbol,
                    "mode": mode,
                    "output": encoded.model_dump(mode="json"),
                }
            )
    context_frame = pd.DataFrame(rows)
    cache_frame = pd.DataFrame(cache_rows)
    if cache_path:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in cache_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return context_frame, raw_frame
