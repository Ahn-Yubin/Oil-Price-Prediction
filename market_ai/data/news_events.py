from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from market_ai.data.event_providers import MarketEvent, load_events_from_file
from market_ai.data.storage import read_table


NEWS_COLUMNS = ("published_at", "symbol", "headline", "body", "source", "url", "retrieved_at")


BULLISH_OIL_TERMS = (
    "supply cut",
    "output cut",
    "production cut",
    "opec cut",
    "deeper cut",
    "extend cuts",
    "inventory draw",
    "crude draw",
    "stockpile draw",
    "drawdown",
    "draws down",
    "sanction",
    "sanctions",
    "embargo",
    "disruption",
    "supply disruption",
    "outage",
    "shutdown",
    "shut down",
    "attack",
    "attacks",
    "strike",
    "strikes",
    "war",
    "conflict",
    "tension",
    "tensions",
    "geopolitical",
    "hormuz",
    "red sea",
    "middle east",
    "iran",
    "russia",
    "ukraine",
    "refinery fire",
    "pipeline leak",
    "shortage",
    "tight supply",
    "supply risk",
    "supply fears",
    "oil jumps",
    "oil rallies",
    "oil surges",
    "prices jump",
    "prices rally",
    "prices surge",
)

BEARISH_OIL_TERMS = (
    "inventory build",
    "crude build",
    "stockpile build",
    "stockpiles rise",
    "glut",
    "surplus",
    "oversupply",
    "output boost",
    "production rise",
    "raises output",
    "raise output",
    "output hike",
    "production hike",
    "demand weak",
    "weak demand",
    "demand concerns",
    "demand slowdown",
    "slowing demand",
    "china demand",
    "recession",
    "slowdown",
    "ceasefire",
    "peace deal",
    "truce",
    "oil falls",
    "oil drops",
    "oil slides",
    "oil slumps",
    "prices fall",
    "prices drop",
    "prices slide",
    "prices slump",
)

HIGH_IMPACT_TERMS = (
    "war",
    "hormuz",
    "red sea",
    "sanction",
    "sanctions",
    "supply disruption",
    "inventory draw",
    "inventory build",
    "opec cut",
    "output cut",
    "ceasefire",
    "attack",
    "strike",
)

GEOPOLITICAL_TERMS = (
    "war",
    "conflict",
    "tension",
    "tensions",
    "geopolitical",
    "hormuz",
    "red sea",
    "iran",
    "russia",
    "ukraine",
    "sanction",
    "sanctions",
    "attack",
    "strike",
)


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
    bullish_hits = sum(1 for term in BULLISH_OIL_TERMS if term in text)
    bearish_hits = sum(1 for term in BEARISH_OIL_TERMS if term in text)
    high_impact_hits = sum(1 for term in HIGH_IMPACT_TERMS if term in text)
    impact = min(0.25 + 0.12 * max(bullish_hits, bearish_hits) + 0.08 * high_impact_hits, 0.85)
    bias = "neutral"
    if bullish_hits > bearish_hits:
        bias = "bullish"
    elif bearish_hits > bullish_hits:
        bias = "bearish"
    elif bullish_hits and bearish_hits:
        bias = "mixed"
    event_type = (
        "geopolitical_oil_news"
        if any(term in text for term in GEOPOLITICAL_TERMS)
        else "energy_news"
        if any(term in text for term in ("oil", "crude", "opec", "inventory", "refinery", "brent", "wti"))
        else "macro_news"
    )
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
