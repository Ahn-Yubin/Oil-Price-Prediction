from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from market_ai.data.event_providers import MarketEvent, load_events_from_file
from market_ai.data.storage import read_table


NEWS_COLUMNS = ("published_at", "symbol", "headline", "body", "source", "url", "retrieved_at")


def load_news_csv(path: str | Path) -> pd.DataFrame:
    frame = read_table(path)
    out = frame.copy()
    out = out.rename(columns={"title": "headline", "text": "body", "date": "published_at", "timestamp": "published_at"})
    if "published_at" not in out.columns or "headline" not in out.columns:
        raise ValueError("news CSV requires published_at/date and headline/title columns")
    out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce", utc=True)
    if "retrieved_at" in out.columns:
        out["retrieved_at"] = pd.to_datetime(out["retrieved_at"], errors="coerce", utc=True)
    else:
        out["retrieved_at"] = out["published_at"]
    if "symbol" not in out.columns:
        out["symbol"] = "ALL"
    if "source" not in out.columns:
        out["source"] = "manual_news"
    if "url" not in out.columns:
        out["url"] = ""
    if "body" not in out.columns:
        out["body"] = ""
    return out.dropna(subset=["published_at", "headline"]).sort_values("published_at")[list(NEWS_COLUMNS)].reset_index(drop=True)


def _local_rule_event_from_news(row: dict[str, Any]) -> MarketEvent:
    text = f"{row.get('headline', '')} {row.get('body', '')}".lower()
    bullish_terms = ("supply cut", "draw", "sanction", "disruption", "attack", "opec cut", "inventory draw")
    bearish_terms = ("build", "surplus", "demand weak", "recession", "inventory build", "production rise")
    impact = 0.25
    bias = "neutral"
    if any(term in text for term in bullish_terms):
        bias = "bullish"
        impact = 0.55
    if any(term in text for term in bearish_terms):
        bias = "bearish" if bias == "neutral" else "mixed"
        impact = max(impact, 0.55)
    event_type = "energy_news" if any(term in text for term in ("oil", "crude", "opec", "inventory", "refinery")) else "macro_news"
    return MarketEvent(
        timestamp=pd.Timestamp(row["published_at"]).to_pydatetime(),
        symbol=str(row.get("symbol") or "ALL"),
        event_type=event_type,
        directional_bias=bias,
        impact_strength=impact,
        uncertainty=0.6,
        source_quality_score=0.55,
        summary=str(row.get("headline") or ""),
        source=str(row.get("source") or "manual_news"),
    )


def news_to_market_events(news: pd.DataFrame) -> list[MarketEvent]:
    return [_local_rule_event_from_news(row) for row in news.to_dict(orient="records")]


def load_event_sources(events_paths: list[str | Path] | None = None, news_paths: list[str | Path] | None = None) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    for path in events_paths or []:
        if str(path):
            events.extend(load_events_from_file(Path(path)))
    for path in news_paths or []:
        if str(path):
            events.extend(news_to_market_events(load_news_csv(path)))
    return sorted(events, key=lambda event: event.timestamp)


def events_to_frame(events: list[MarketEvent]) -> pd.DataFrame:
    return pd.DataFrame([event.as_dict() for event in events])
