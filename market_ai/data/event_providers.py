from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from market_ai.features.context_features import build_event_context_vector
from market_ai.schemas.deep_learning import EventContextVector


EVENT_FILE_ENV_VARS = ("NEWS_EVENTS_PATH", "ECONOMIC_EVENTS_PATH", "MARKET_EVENTS_PATH")


@dataclass(frozen=True)
class MarketEvent:
    timestamp: datetime
    symbol: str | None
    event_type: str
    directional_bias: str = "neutral"
    impact_strength: float = 0.0
    uncertainty: float = 0.5
    source_quality_score: float = 0.5
    summary: str = ""
    source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "event_type": self.event_type,
            "directional_bias": self.directional_bias,
            "impact_strength": self.impact_strength,
            "uncertainty": self.uncertainty,
            "source_quality_score": self.source_quality_score,
            "summary": self.summary,
            "source": self.source,
        }


def _parse_timestamp(value: Any) -> datetime | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out and out not in {float("inf"), float("-inf")} else default


def _event_from_mapping(raw: dict[str, Any]) -> MarketEvent | None:
    ts = _parse_timestamp(raw.get("timestamp") or raw.get("published_at") or raw.get("scheduled_at") or raw.get("date"))
    if ts is None:
        return None
    return MarketEvent(
        timestamp=ts,
        symbol=str(raw.get("symbol") or raw.get("asset") or "").strip() or None,
        event_type=str(raw.get("event_type") or raw.get("category") or "market_event"),
        directional_bias=str(raw.get("directional_bias") or raw.get("bias") or "neutral"),
        impact_strength=_safe_float(raw.get("impact_strength", raw.get("impact")), 0.0),
        uncertainty=_safe_float(raw.get("uncertainty"), 0.5),
        source_quality_score=_safe_float(raw.get("source_quality_score", raw.get("source_quality")), 0.5),
        summary=str(raw.get("summary") or raw.get("title") or raw.get("text") or ""),
        source=str(raw.get("source") or "") or None,
    )


def load_events_from_file(path: Path) -> list[MarketEvent]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("events", []) if isinstance(data, dict) else []
    else:
        rows = pd.read_csv(path).to_dict(orient="records")
    out: list[MarketEvent] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        event = _event_from_mapping(row)
        if event is not None:
            out.append(event)
    return sorted(out, key=lambda event: event.timestamp)


class FileEventProvider:
    def __init__(self, paths: Iterable[str | Path] | None = None):
        self.paths = [Path(path).expanduser() for path in paths or [] if str(path)]
        self._cache: list[MarketEvent] | None = None

    @classmethod
    def from_env(cls) -> "FileEventProvider":
        return cls([os.environ.get(name, "") for name in EVENT_FILE_ENV_VARS])

    def load_events(self) -> list[MarketEvent]:
        if self._cache is not None:
            return self._cache
        events: list[MarketEvent] = []
        for path in self.paths:
            events.extend(load_events_from_file(path))
        self._cache = sorted(events, key=lambda event: event.timestamp)
        return self._cache

    def events_as_of(self, *, symbol: str, as_of_time: datetime) -> list[MarketEvent]:
        as_of = as_of_time if as_of_time.tzinfo else as_of_time.replace(tzinfo=timezone.utc)
        symbol_upper = symbol.upper()
        out: list[MarketEvent] = []
        for event in self.load_events():
            if event.timestamp > as_of:
                continue
            if event.symbol and event.symbol.upper() not in {symbol_upper, "ALL", "*"}:
                continue
            out.append(event)
        return out

    def context_vector(self, *, symbol: str, as_of_time: datetime) -> EventContextVector:
        events = [event.as_dict() for event in self.events_as_of(symbol=symbol, as_of_time=as_of_time)]
        return build_event_context_vector(events, as_of_time=as_of_time)


class NullEventProvider(FileEventProvider):
    def __init__(self) -> None:
        super().__init__([])

    def load_events(self) -> list[MarketEvent]:
        return []
