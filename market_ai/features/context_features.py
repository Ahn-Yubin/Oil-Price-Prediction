from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping, Any

import numpy as np

from market_ai.schemas.deep_learning import EventContextVector


EVENT_CONTEXT_DIM = len(EventContextVector().as_list())


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        import pandas as pd

        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    except Exception:
        return None


def _bias_score(value: str | None) -> float:
    normalized = (value or "").strip().lower()
    if normalized == "bullish":
        return 1.0
    if normalized == "bearish":
        return -1.0
    if normalized == "mixed":
        return 0.0
    return 0.0


def build_event_context_vector(
    events: Iterable[Mapping[str, Any]],
    *,
    as_of_time: datetime,
) -> EventContextVector:
    as_of = as_of_time if as_of_time.tzinfo else as_of_time.replace(tzinfo=timezone.utc)
    usable: list[dict[str, Any]] = []
    for raw in events:
        ts = _parse_time(raw.get("timestamp") or raw.get("published_at") or raw.get("scheduled_at"))
        if ts is None or ts > as_of:
            continue
        age_days = max((as_of - ts).total_seconds() / 86_400.0, 0.0)
        item = dict(raw)
        item["_age_days"] = age_days
        item["_decay"] = float(np.exp(-age_days / 7.0))
        usable.append(item)

    if not usable:
        return EventContextVector()

    weighted_bias = 0.0
    weighted_impact = 0.0
    weighted_uncertainty = 0.0
    bullish_score = 0.0
    bearish_score = 0.0
    macro_score = 0.0
    energy_score = 0.0
    geo_score = 0.0
    source_quality = 0.0
    total_weight = 0.0
    counts = {1: 0.0, 3: 0.0, 7: 0.0}

    for event in usable:
        age = float(event["_age_days"])
        decay = float(event["_decay"])
        impact = float(event.get("impact_strength", event.get("impact", 0.0)) or 0.0)
        uncertainty = float(event.get("uncertainty", 0.5) or 0.5)
        quality = float(event.get("source_quality_score", event.get("source_quality", 0.5)) or 0.5)
        event_type = str(event.get("event_type", event.get("category", ""))).lower()
        bias = _bias_score(str(event.get("directional_bias", event.get("bias", ""))))
        weight = max(impact, 0.05) * decay * max(quality, 0.05)
        total_weight += weight
        weighted_bias += bias * weight
        weighted_impact += impact * weight
        weighted_uncertainty += uncertainty * weight
        source_quality += quality * weight
        bullish_score += max(bias, 0.0) * weight
        bearish_score += max(-bias, 0.0) * weight
        if age <= 1.0:
            counts[1] += 1.0
        if age <= 3.0:
            counts[3] += 1.0
        if age <= 7.0:
            counts[7] += 1.0
        if "macro" in event_type or "economic" in event_type or "policy" in event_type:
            macro_score += weight
        if "energy" in event_type or "oil" in event_type or "supply" in event_type or "demand" in event_type:
            energy_score += weight
        if "geo" in event_type or "war" in event_type or "sanction" in event_type:
            geo_score += weight

    denom = max(total_weight, 1e-8)
    return EventContextVector(
        directional_bias_score=weighted_bias / denom,
        impact_strength=min(weighted_impact / denom, 1.0),
        uncertainty=min(weighted_uncertainty / denom, 1.0),
        time_decay=min(float(np.mean([event["_decay"] for event in usable])), 1.0),
        event_count_1d=counts[1],
        event_count_3d=counts[3],
        event_count_7d=counts[7],
        bullish_event_score=bullish_score / denom,
        bearish_event_score=bearish_score / denom,
        macro_event_score=macro_score / denom,
        energy_event_score=energy_score / denom,
        geopolitical_event_score=geo_score / denom,
        source_quality_score=source_quality / denom,
    )
