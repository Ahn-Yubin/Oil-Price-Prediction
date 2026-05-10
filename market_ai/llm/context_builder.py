from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
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
from market_ai.schemas.llm_context import MarketContextInput, RawNewsItem


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
    if normalized == "google_generative":
        from market_ai.llm.event_encoder import GoogleGenerativeLLMEventEncoder

        return GoogleGenerativeLLMEventEncoder(
            api_key=api_key,
            model=model or "gemma-3-27b-it",
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


def _recent_news_items(
    provider: FileEventProvider,
    *,
    symbol: str,
    as_of_time: datetime,
    lookback_days: int = 7,
    limit: int = 5,
) -> list[RawNewsItem]:
    cutoff = as_of_time - timedelta(days=lookback_days)
    events = [
        event
        for event in provider.events_as_of(symbol=symbol, as_of_time=as_of_time)
        if event.timestamp >= cutoff and event.summary
    ]
    events = sorted(events, key=lambda event: event.timestamp, reverse=True)[:limit]
    return [
        RawNewsItem(
            title=event.summary,
            source=event.source,
            published_at=event.timestamp,
        )
        for event in events
    ]


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
    progress_callback: Callable[[dict[str, object]], None] | None = None,
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
    external_mode = mode.strip().lower() in {"openai_compatible", "google_generative"}
    dates = list(pd.date_range(start_ts.floor("D"), end_ts.floor("D"), freq="D", tz="UTC"))
    schedule: list[tuple[str, pd.Timestamp, list[RawNewsItem]]] = []
    for symbol in symbols:
        for as_of in dates:
            news_items = _recent_news_items(provider, symbol=symbol, as_of_time=as_of.to_pydatetime())
            schedule.append((symbol, as_of, news_items))
    total_rows = len(schedule)
    total_llm_calls = sum(1 for _, _, news_items in schedule if external_mode and news_items)
    started = time.monotonic()
    completed_rows = 0
    completed_llm_calls = 0
    for symbol, as_of, news_items in schedule:
        external_call = external_mode and bool(news_items)
        if progress_callback and external_call:
            progress_callback(
                {
                    "phase": "llm_start",
                    "completed_rows": completed_rows,
                    "total_rows": total_rows,
                    "completed_llm_calls": completed_llm_calls,
                    "total_llm_calls": total_llm_calls,
                    "symbol": symbol,
                    "timestamp": as_of.isoformat(),
                    "llm_input_news_count": len(news_items),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
        context = MarketContextInput(symbol=symbol, interval="1d", generated_at=as_of.to_pydatetime(), news=news_items)
        if external_mode and not news_items:
            encoded = LocalEventContextEncoder(provider).encode_events(context)
        else:
            encoded = encoder.encode_events(context)
        if external_call:
            completed_llm_calls += 1
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
            "llm_input_news_count": len(news_items),
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
                "input_news_count": len(news_items),
                "output": encoded.model_dump(mode="json"),
            }
        )
        completed_rows += 1
        if progress_callback:
            progress_callback(
                {
                    "phase": "row_done",
                    "completed_rows": completed_rows,
                    "total_rows": total_rows,
                    "completed_llm_calls": completed_llm_calls,
                    "total_llm_calls": total_llm_calls,
                    "symbol": symbol,
                    "timestamp": as_of.isoformat(),
                    "llm_input_news_count": len(news_items),
                    "event_count": len(encoded.events),
                    "warnings": "|".join(encoded.warnings),
                    "elapsed_seconds": time.monotonic() - started,
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
