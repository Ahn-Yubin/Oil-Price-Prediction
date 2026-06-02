from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from market_ai.config import Settings
from market_ai.data.event_providers import FileEventProvider
from market_ai.data.news_events import news_to_market_events
from market_ai.data.providers.public_news_provider import (
    GOOGLE_NEWS_TOPIC_QUERIES,
    fetch_google_news_rss,
    fetch_yahoo_finance_rss,
    normalize_public_news,
)
from market_ai.features.context_features import EVENT_CONTEXT_DIM
from market_ai.llm.context_builder import encoder_for_mode
from market_ai.schemas.deep_learning import EventContextVector
from market_ai.schemas.llm_context import LLMContextOutput, MarketContextInput, RawNewsItem


LIVE_NEWS_LIMIT = 24

OIL_RELEVANCE_TERMS = {
    "crude oil": 4,
    "oil price": 4,
    "oil prices": 4,
    "brent": 3,
    "wti": 3,
    "opec": 4,
    "eia": 3,
    "iea": 3,
    "inventory": 2,
    "inventories": 2,
    "refinery": 2,
    "refineries": 2,
    "tanker": 2,
    "sanction": 2,
    "sanctions": 2,
    "iran": 2,
    "russia": 2,
    "ukraine": 2,
    "middle east": 2,
    "red sea": 2,
    "supply cut": 3,
    "production cut": 3,
    "oil demand": 3,
    "energy market": 2,
    "gasoline": 2,
    "diesel": 2,
    "lng": 2,
    "natural gas prices": 2,
}

OIL_NOISE_TERMS = {
    "water heater",
    "appliance",
    "building code",
    "homeowner",
    "residential",
    "bay area",
    "climate fight",
    "natural gas ban",
}


def _is_oil_symbol(symbol: str) -> bool:
    normalized = symbol.upper()
    return any(token in normalized for token in ("CL", "BZ", "BRN", "USOIL", "UKOIL", "OIL", "USO", "XLE"))


def _query_for_symbol(symbol: str) -> str:
    if _is_oil_symbol(symbol):
        return GOOGLE_NEWS_TOPIC_QUERIES["energy"]
    normalized = symbol.upper()
    if any(token in normalized for token in ("GC", "SI", "HG", "XAU", "XAG", "COPPER", "GOLD", "SILVER")):
        return GOOGLE_NEWS_TOPIC_QUERIES["metals"]
    if any(token in normalized for token in ("USD", "EUR", "JPY", "KRW", "DXY", "DX-")):
        return GOOGLE_NEWS_TOPIC_QUERIES["fx_macro"]
    return GOOGLE_NEWS_TOPIC_QUERIES["equities_vol"]


def _oil_relevance_score(text: str) -> int:
    lowered = text.lower()
    if any(term in lowered for term in OIL_NOISE_TERMS):
        return 0
    return sum(score for term, score in OIL_RELEVANCE_TERMS.items() if term in lowered)


def _filter_relevant_news(symbol: str, news: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if news.empty or not _is_oil_symbol(symbol):
        return news, []
    frame = news.copy()
    combined = (
        frame.get("headline", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("body", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("source", pd.Series("", index=frame.index)).fillna("").astype(str)
    )
    frame["relevance_score"] = combined.map(_oil_relevance_score)
    filtered = frame[frame["relevance_score"] >= 2].copy()
    if filtered.empty:
        return filtered.drop(columns=["relevance_score"], errors="ignore"), [
            "Live news fetched, but no oil-price-relevant headlines passed the relevance filter."
        ]
    filtered = filtered.sort_values(["published_at", "relevance_score"], ascending=[True, True])
    return filtered.drop(columns=["relevance_score"], errors="ignore"), []


def fetch_live_news(symbol: str, *, limit: int = LIVE_NEWS_LIMIT) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    query = _query_for_symbol(symbol)
    try:
        frames.append(fetch_google_news_rss("live_market", query))
    except Exception as exc:
        warnings.append(f"Google News RSS unavailable: {exc}")
    try:
        frames.append(fetch_yahoo_finance_rss(symbol))
    except Exception as exc:
        warnings.append(f"Yahoo Finance RSS unavailable: {exc}")

    news = normalize_public_news(frames)
    if news.empty:
        return news, warnings
    news, relevance_warnings = _filter_relevant_news(symbol, news)
    warnings.extend(relevance_warnings)
    if news.empty:
        return news, warnings
    news = news.tail(max(1, int(limit))).copy()
    # Keep the requested symbol on broad topic news so downstream point-in-time filters can use it.
    news["symbol"] = news["symbol"].replace({"ALL": symbol, "*": symbol, "": symbol}).fillna(symbol)
    return news.reset_index(drop=True), warnings


def _raw_news_items(news: pd.DataFrame, *, limit: int = 8) -> list[RawNewsItem]:
    rows = []
    if news.empty:
        return rows
    frame = news.copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["published_at", "headline"]).sort_values("published_at")
    for row in frame.tail(max(1, int(limit))).to_dict(orient="records"):
        rows.append(
            RawNewsItem(
                title=str(row.get("headline") or ""),
                source=str(row.get("source") or ""),
                url=str(row.get("url") or "") or None,
                published_at=pd.Timestamp(row["published_at"]).to_pydatetime(),
            )
        )
    return rows


def _provider_from_news(news: pd.DataFrame) -> FileEventProvider:
    provider = FileEventProvider(paths=[])
    provider._cache = news_to_market_events(news) if not news.empty else []
    return provider


def _context_row(
    *,
    symbol: str,
    as_of_time: datetime,
    mode: str,
    encoded: LLMContextOutput,
    news_count: int,
    warnings: list[str],
) -> dict[str, Any]:
    embedding = list(encoded.event_embedding or [])
    if len(embedding) < EVENT_CONTEXT_DIM:
        embedding.extend([0.0] * (EVENT_CONTEXT_DIM - len(embedding)))
    vector_names = list(EventContextVector.model_fields)
    row: dict[str, Any] = {
        "timestamp": pd.Timestamp(as_of_time).isoformat(),
        "symbol": symbol,
        "feature_available_at": pd.Timestamp(as_of_time).isoformat(),
        "llm_mode": mode,
        "overall_bias": encoded.overall_bias,
        "impact_score": float(encoded.impact_score),
        "uncertainty": float(encoded.uncertainty),
        "event_count": len(encoded.events),
        "llm_input_news_count": news_count,
        "explanation": encoded.explanation,
        "warnings": "|".join([*encoded.warnings, *warnings]),
    }
    for idx, name in enumerate(vector_names):
        row[name] = float(embedding[idx]) if idx < len(embedding) else 0.0
    return row


def build_live_event_context(
    *,
    symbol: str,
    settings: Settings,
    as_of_time: datetime | None = None,
    news_limit: int = LIVE_NEWS_LIMIT,
) -> dict[str, Any]:
    resolved_as_of = as_of_time or datetime.now(timezone.utc)
    news, fetch_warnings = fetch_live_news(symbol, limit=news_limit)
    provider = _provider_from_news(news)
    mode = settings.llm_context_mode if settings.enable_llm_context else "local_rules"
    encoder = encoder_for_mode(
        mode,
        provider=provider,
        api_key=settings.llm_api_key,
        api_base=settings.llm_api_base if mode != "local_http" else settings.local_llm_api_base,
        model=settings.llm_model if mode != "local_http" else settings.local_llm_model,
        live=bool(settings.enable_llm_context and settings.enable_external_llm_calls),
    )
    news_items = _raw_news_items(news)
    encoded = encoder.encode_events(
        MarketContextInput(
            symbol=symbol,
            interval="1d",
            generated_at=resolved_as_of,
            news=news_items,
        )
    )
    row = _context_row(
        symbol=symbol,
        as_of_time=resolved_as_of,
        mode=mode,
        encoded=encoded,
        news_count=len(news_items),
        warnings=fetch_warnings,
    )
    return {
        "news": news,
        "context_frame": pd.DataFrame([row]),
        "context_points": [row],
        "llm_context": encoded,
        "warnings": fetch_warnings,
        "source": "live_public_news",
    }
