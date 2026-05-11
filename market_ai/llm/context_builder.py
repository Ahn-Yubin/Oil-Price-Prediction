from __future__ import annotations

import json
import hashlib
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
from market_ai.schemas.llm_context import LLMContextOutput, MarketContextInput, RawNewsItem


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


def _news_input_hash(news_items: list[RawNewsItem]) -> str:
    payload = [
        {
            "title": item.title,
            "source": item.source,
            "published_at": item.published_at.isoformat() if item.published_at else "",
            "url": item.url,
        }
        for item in news_items
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_context_cache(cache_path: str | Path | None) -> dict[tuple[str, str, str, str], LLMContextOutput]:
    if not cache_path:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    out: dict[tuple[str, str, str, str], LLMContextOutput] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                input_hash = str(raw.get("input_hash") or "")
                if not input_hash:
                    continue
                key = (
                    str(raw.get("symbol") or ""),
                    str(raw.get("as_of_time") or "")[:10],
                    str(raw.get("mode") or ""),
                    input_hash,
                )
                out[key] = LLMContextOutput.model_validate(raw.get("output") or {})
            except Exception:
                continue
    return out


def _append_context_cache_row(cache_path: str | Path | None, row: dict) -> None:
    if not cache_path:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _is_external_fallback(output: LLMContextOutput) -> bool:
    warnings = [str(warning).lower() for warning in output.warnings]
    return any(
        "external llm fallback" in warning
        or "external llm api key missing" in warning
        or "external llm disabled" in warning
        or "invalid llm json fallback" in warning
        or "invalid llm batch json fallback" in warning
        for warning in warnings
    )


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
    resume_from_cache: bool = True,
    news_limit_per_context: int = 5,
    llm_batch_size: int = 1,
    llm_min_interval_seconds: float = 0.0,
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
            news_items = _recent_news_items(
                provider,
                symbol=symbol,
                as_of_time=as_of.to_pydatetime(),
                limit=max(1, int(news_limit_per_context)),
            )
            schedule.append((symbol, as_of, news_items))
    total_rows = len(schedule)
    total_llm_contexts = sum(1 for _, _, news_items in schedule if external_mode and news_items)
    cached_outputs = _load_context_cache(cache_path) if resume_from_cache else {}
    started = time.monotonic()
    completed_rows = 0
    completed_llm_contexts = 0
    completed_llm_requests = 0
    max_batch = max(1, int(llm_batch_size))
    min_interval = max(0.0, float(llm_min_interval_seconds))
    last_llm_request_at: float | None = None

    def wait_for_llm_slot() -> None:
        nonlocal last_llm_request_at
        if min_interval > 0.0 and last_llm_request_at is not None:
            delay = min_interval - (time.monotonic() - last_llm_request_at)
            if delay > 0.0:
                time.sleep(delay)
        last_llm_request_at = time.monotonic()

    def append_result(
        *,
        symbol: str,
        as_of: pd.Timestamp,
        news_items: list[RawNewsItem],
        input_hash: str,
        encoded: LLMContextOutput,
        cached: bool,
    ) -> None:
        nonlocal completed_rows
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
        cache_row = {
            "as_of_time": as_of.isoformat(),
            "symbol": symbol,
            "mode": mode,
            "input_hash": input_hash,
            "input_news_count": len(news_items),
            "output": encoded.model_dump(mode="json"),
        }
        cache_rows.append(cache_row)
        if not cached and not (external_mode and _is_external_fallback(encoded)):
            _append_context_cache_row(cache_path, cache_row)
        completed_rows += 1
        if progress_callback:
            progress_callback(
                {
                    "phase": "row_done",
                    "completed_rows": completed_rows,
                    "total_rows": total_rows,
                    "completed_llm_calls": completed_llm_contexts,
                    "total_llm_calls": total_llm_contexts,
                    "completed_llm_contexts": completed_llm_contexts,
                    "total_llm_contexts": total_llm_contexts,
                    "completed_llm_requests": completed_llm_requests,
                    "symbol": symbol,
                    "timestamp": as_of.isoformat(),
                    "llm_input_news_count": len(news_items),
                    "event_count": len(encoded.events),
                    "warnings": "|".join([*encoded.warnings, "cache_hit"] if cached else encoded.warnings),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )

    idx = 0
    while idx < len(schedule):
        symbol, as_of, news_items = schedule[idx]
        input_hash = _news_input_hash(news_items)
        cache_key = (symbol, as_of.date().isoformat(), mode, input_hash)
        cached = cached_outputs.get(cache_key)
        external_call = external_mode and bool(news_items)
        context = MarketContextInput(symbol=symbol, interval="1d", generated_at=as_of.to_pydatetime(), news=news_items)

        if cached is not None:
            append_result(symbol=symbol, as_of=as_of, news_items=news_items, input_hash=input_hash, encoded=cached, cached=True)
            idx += 1
            continue
        if external_mode and not news_items:
            encoded = LocalEventContextEncoder(provider).encode_events(context)
            append_result(symbol=symbol, as_of=as_of, news_items=news_items, input_hash=input_hash, encoded=encoded, cached=False)
            idx += 1
            continue

        if external_call and max_batch > 1:
            batch: list[tuple[str, pd.Timestamp, list[RawNewsItem], str, MarketContextInput]] = []
            lookahead = idx
            while lookahead < len(schedule) and len(batch) < max_batch:
                batch_symbol, batch_as_of, batch_news = schedule[lookahead]
                batch_hash = _news_input_hash(batch_news)
                batch_cache_key = (batch_symbol, batch_as_of.date().isoformat(), mode, batch_hash)
                if cached_outputs.get(batch_cache_key) is not None or not (external_mode and bool(batch_news)):
                    break
                batch_context = MarketContextInput(
                    symbol=batch_symbol,
                    interval="1d",
                    generated_at=batch_as_of.to_pydatetime(),
                    news=batch_news,
                )
                batch.append((batch_symbol, batch_as_of, batch_news, batch_hash, batch_context))
                lookahead += 1
            if batch:
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "llm_batch_start",
                            "completed_rows": completed_rows,
                            "total_rows": total_rows,
                            "completed_llm_calls": completed_llm_contexts,
                            "total_llm_calls": total_llm_contexts,
                            "completed_llm_contexts": completed_llm_contexts,
                            "total_llm_contexts": total_llm_contexts,
                            "completed_llm_requests": completed_llm_requests,
                            "batch_contexts": len(batch),
                            "symbol": batch[0][0],
                            "timestamp": batch[0][1].isoformat(),
                            "llm_input_news_count": sum(len(item[2]) for item in batch),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                wait_for_llm_slot()
                encoded_batch = encoder.encode_event_batch([item[4] for item in batch])
                completed_llm_requests += 1
                completed_llm_contexts += len(batch)
                for (batch_symbol, batch_as_of, batch_news, batch_hash, _), encoded in zip(batch, encoded_batch, strict=True):
                    append_result(
                        symbol=batch_symbol,
                        as_of=batch_as_of,
                        news_items=batch_news,
                        input_hash=batch_hash,
                        encoded=encoded,
                        cached=False,
                    )
                idx += len(batch)
                continue

        if progress_callback and external_call:
            progress_callback(
                {
                    "phase": "llm_start",
                    "completed_rows": completed_rows,
                    "total_rows": total_rows,
                    "completed_llm_calls": completed_llm_contexts,
                    "total_llm_calls": total_llm_contexts,
                    "completed_llm_contexts": completed_llm_contexts,
                    "total_llm_contexts": total_llm_contexts,
                    "completed_llm_requests": completed_llm_requests,
                    "symbol": symbol,
                    "timestamp": as_of.isoformat(),
                    "llm_input_news_count": len(news_items),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
        if external_call:
            wait_for_llm_slot()
        encoded = encoder.encode_events(context)
        if external_call:
            completed_llm_requests += 1
            completed_llm_contexts += 1
        append_result(symbol=symbol, as_of=as_of, news_items=news_items, input_hash=input_hash, encoded=encoded, cached=False)
        idx += 1
    context_frame = pd.DataFrame(rows)
    cache_frame = pd.DataFrame(cache_rows)
    return context_frame, raw_frame
