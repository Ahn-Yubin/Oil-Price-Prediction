from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from market_ai.config import Settings, get_settings
from market_ai.data.event_providers import FileEventProvider
from market_ai.data.news_events import news_to_market_events
from market_ai.data.providers.yfinance_provider import load_market_data_window
from market_ai.forecasting.service import ForecastBundle, build_forecast
from market_ai.llm.context_builder import encoder_for_mode, raw_news_pool_features
from market_ai.schemas.deep_learning import EventContextVector
from market_ai.schemas.llm_context import LLMContextOutput, MarketContextInput, RawNewsItem
from market_ai.schemas.market import ForecastPoint, ForecastWarning, ScenarioEventInput, ScenarioForecastResponse, ScenarioPoint


class ScenarioForecastUnavailable(RuntimeError):
    pass


_EXTERNAL_LLM_MODES = {"openai_compatible", "google_generative", "local_http"}


@dataclass(frozen=True)
class ScenarioEventSpec:
    title: str
    content: str
    event_time: datetime | None


@dataclass(frozen=True)
class ScenarioEventSignal:
    title: str
    content: str
    event_time: datetime | None
    bias: float
    impact: float
    kind: str


def _normalize_mode(value: str) -> str:
    normalized = str(value or "local_rules").strip().lower()
    if normalized in {"openai", "openai-compatible", "openai compatible"}:
        return "openai_compatible"
    if normalized in {"google", "google-generative", "google generative", "gemini", "gemma"}:
        return "google_generative"
    return normalized


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _scenario_id(title: str, content: str, event_time: datetime | None) -> str:
    payload = f"{title.strip()}|{content.strip()}|{event_time.isoformat() if event_time else ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _event_specs(
    *,
    title: str,
    content: str,
    event_time: datetime | None,
    events: list[ScenarioEventInput] | None,
) -> list[ScenarioEventSpec]:
    specs = [
        ScenarioEventSpec(
            title=str(event.title or "").strip(),
            content=str(event.content or "").strip(),
            event_time=_as_utc(event.event_time),
        )
        for event in (events or [])
        if str(event.title or "").strip() and str(event.content or "").strip()
    ]
    if specs:
        return specs
    return [ScenarioEventSpec(title=title.strip(), content=content.strip(), event_time=_as_utc(event_time))]


def _event_specs_content(specs: list[ScenarioEventSpec], fallback: str) -> str:
    if not specs:
        return fallback.strip()
    blocks = []
    for index, spec in enumerate(specs, start=1):
        event_time = spec.event_time.isoformat() if spec.event_time else "not specified"
        blocks.append(
            "\n".join(
                [
                    f"Event {index}: {spec.title}",
                    f"Occurrence time: {event_time}",
                    f"Details: {spec.content}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _scenario_text(title: str, content: str, event_time: datetime | None, as_of: datetime) -> str:
    event_time_text = event_time.isoformat() if event_time else "not specified; infer from the scenario text relative to generated_at"
    return (
        "Hypothetical future oil-market scenario.\n"
        f"Forecast origin / generated_at: {as_of.isoformat()}\n"
        f"Expected event time: {event_time_text}\n"
        f"Title: {title.strip()}\n"
        f"Scenario: {content.strip()}\n\n"
        "Encode this as structured market event context known at the forecast origin. "
        "Do not output oil price targets, p50/p90 prices, or future return paths."
    )


_BULLISH_TERMS = (
    "blockade",
    "blocked",
    "closure",
    "disruption",
    "outage",
    "attack",
    "strike",
    "war",
    "invasion",
    "sanction",
    "embargo",
    "supply shock",
    "supply disruption",
    "production cut",
    "output cut",
    "opec cut",
    "hormuz",
    "red sea",
    "봉쇄",
    "재봉쇄",
    "차단",
    "차질",
    "공급 충격",
    "공급 차질",
    "감산",
    "공격",
    "침공",
    "전쟁",
    "제재",
    "호르무즈",
)

_BEARISH_TERMS = (
    "production increase",
    "output increase",
    "supply increase",
    "raise production",
    "inventory build",
    "stockpile build",
    "ceasefire",
    "deal",
    "peace",
    "demand slowdown",
    "recession",
    "weak demand",
    "opec increase",
    "증산",
    "생산량 증가",
    "공급 증가",
    "재고 증가",
    "휴전",
    "합의",
    "평화",
    "수요 둔화",
    "경기 침체",
)

_HIGH_IMPACT_TERMS = (
    "hormuz",
    "war",
    "invasion",
    "attack",
    "sanction",
    "blockade",
    "opec",
    "호르무즈",
    "전쟁",
    "침공",
    "공격",
    "제재",
    "봉쇄",
    "증산",
    "감산",
)


def _term_hits(text: str, terms: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def _signal_for_event(spec: ScenarioEventSpec) -> ScenarioEventSignal:
    text = f"{spec.title}\n{spec.content}"
    bullish_hits = _term_hits(text, _BULLISH_TERMS)
    bearish_hits = _term_hits(text, _BEARISH_TERMS)
    if bullish_hits > bearish_hits:
        bias = 1.0
        kind = "bullish"
    elif bearish_hits > bullish_hits:
        bias = -1.0
        kind = "bearish"
    elif bullish_hits and bearish_hits:
        bias = 0.0
        kind = "mixed"
    else:
        bias = 0.0
        kind = "neutral"
    impact = min(0.95, 0.45 + 0.12 * max(bullish_hits, bearish_hits) + 0.08 * _term_hits(text, _HIGH_IMPACT_TERMS))
    return ScenarioEventSignal(
        title=spec.title,
        content=spec.content,
        event_time=spec.event_time,
        bias=bias,
        impact=max(0.35, impact),
        kind=kind,
    )


def _signals_from_specs(specs: list[ScenarioEventSpec]) -> list[ScenarioEventSignal]:
    return [_signal_for_event(spec) for spec in specs]


def _bias_from_features(raw_features: dict[str, float], fallback: str) -> str:
    fallback_normalized = str(fallback or "unknown").strip().lower()
    if fallback_normalized in {"bullish", "bearish", "neutral", "mixed"}:
        return fallback_normalized
    bullish = float(raw_features.get("raw_bullish_pressure", 0.0))
    bearish = float(raw_features.get("raw_bearish_pressure", 0.0))
    net = float(raw_features.get("raw_net_pressure", bullish - bearish))
    if bullish >= 0.25 and bearish >= 0.25 and abs(net) < 0.22:
        return "mixed"
    if net > 0.12:
        return "bullish"
    if net < -0.12:
        return "bearish"
    return "neutral"


def _raw_features_from_event_signals(signals: list[ScenarioEventSignal]) -> dict[str, float]:
    if not signals:
        return {}
    bullish = sum(max(signal.bias, 0.0) * signal.impact for signal in signals)
    bearish = sum(max(-signal.bias, 0.0) * signal.impact for signal in signals)
    total = max(bullish + bearish, 1e-8)
    geopolitical = sum(signal.impact for signal in signals if _contains_any(f"{signal.title} {signal.content}", ("hormuz", "war", "attack", "iran", "red sea", "호르무즈", "전쟁", "공격", "이란")))
    supply = sum(signal.impact for signal in signals if _contains_any(f"{signal.title} {signal.content}", ("supply", "production", "opec", "inventory", "공급", "생산", "증산", "감산", "재고")))
    max_impact = max(signal.impact for signal in signals)
    return {
        "raw_bullish_pressure": float(bullish / total) if bullish or bearish else 0.0,
        "raw_bearish_pressure": float(bearish / total) if bullish or bearish else 0.0,
        "raw_net_pressure": float((bullish - bearish) / total) if bullish or bearish else 0.0,
        "raw_energy_pressure": min(1.0, max_impact + 0.10),
        "raw_geopolitical_pressure": min(1.0, geopolitical / max(total, 1e-8)),
        "raw_supply_pressure": min(1.0, supply / max(total, 1e-8)),
        "source_diversity_score": min(0.80, 0.45 + 0.08 * len(signals)),
    }


def _scenario_news_frame(*, title: str, content: str, event_time: datetime | None, as_of: datetime, symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "published_at": pd.Timestamp(as_of),
                "symbol": symbol,
                "headline": title.strip(),
                "body": _scenario_text(title, content, event_time, as_of),
                "source": "user_scenario",
                "url": "",
                "retrieved_at": pd.Timestamp(as_of),
            }
        ]
    )


def _provider_from_scenario(news: pd.DataFrame) -> FileEventProvider:
    provider = FileEventProvider(paths=[])
    provider._cache = news_to_market_events(news) if not news.empty else []
    return provider


def _bias_numeric(value: str | None) -> float:
    normalized = str(value or "").strip().lower()
    if normalized == "bullish":
        return 1.0
    if normalized == "bearish":
        return -1.0
    return 0.0


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _raw_features_from_encoded(encoded: LLMContextOutput) -> dict[str, float]:
    events = list(encoded.events or [])
    if not events:
        return {
            "news_volume_1d": 0.0,
            "news_volume_3d": 0.0,
            "news_volume_7d": 0.0,
            "news_volume_30d": 0.0,
            "news_selection_coverage": 0.0,
            "raw_bullish_pressure": 0.0,
            "raw_bearish_pressure": 0.0,
            "raw_net_pressure": 0.0,
            "raw_energy_pressure": 0.0,
            "raw_geopolitical_pressure": 0.0,
            "raw_macro_pressure": 0.0,
            "raw_supply_pressure": 0.0,
            "raw_demand_pressure": 0.0,
            "source_diversity_score": 0.0,
        }

    bullish = 0.0
    bearish = 0.0
    energy = 0.0
    geopolitical = 0.0
    macro = 0.0
    supply = 0.0
    demand = 0.0
    weighted_total = 0.0
    for event in events:
        impact = min(max(float(event.impact_strength or 0.0), 0.0), 1.0)
        uncertainty_discount = 1.0 - 0.35 * min(max(float(event.uncertainty or 0.0), 0.0), 1.0)
        time_weight = max(float(event.time_decay or 0.0), 0.05)
        weight = max(impact, 0.05) * max(uncertainty_discount, 0.10) * time_weight
        bias = _bias_numeric(event.directional_bias)
        event_type = str(event.event_type or "")
        summary = f"{event.summary or ''} {' '.join(event.risk_factors or [])} {event_type}"
        weighted_total += weight
        bullish += max(bias, 0.0) * weight
        bearish += max(-bias, 0.0) * weight
        if _contains_any(summary, ("energy", "oil", "crude", "wti", "brent", "opec", "supply", "demand")):
            energy += weight
        if _contains_any(summary, ("geo", "war", "conflict", "attack", "strike", "sanction", "iran", "hormuz", "red sea")):
            geopolitical += weight
        if _contains_any(summary, ("macro", "rates", "dollar", "fed", "growth", "recession", "china")):
            macro += weight
        if _contains_any(summary, ("supply", "disruption", "blockade", "embargo", "sanction", "attack", "strike", "hormuz", "outage")):
            supply += weight
        if _contains_any(summary, ("demand", "consumption", "china", "growth", "recession", "slowdown")):
            demand += weight

    denom = max(weighted_total, 1e-8)
    return {
        "news_volume_1d": min(math.log1p(len(events)) / math.log1p(12.0), 1.0),
        "news_volume_3d": min(math.log1p(len(events)) / math.log1p(24.0), 1.0),
        "news_volume_7d": min(math.log1p(len(events)) / math.log1p(48.0), 1.0),
        "news_volume_30d": min(math.log1p(len(events)) / math.log1p(120.0), 1.0),
        "news_selection_coverage": 1.0,
        "raw_bullish_pressure": float(bullish / denom),
        "raw_bearish_pressure": float(bearish / denom),
        "raw_net_pressure": float((bullish - bearish) / denom),
        "raw_energy_pressure": float(energy / denom),
        "raw_geopolitical_pressure": float(geopolitical / denom),
        "raw_macro_pressure": float(macro / denom),
        "raw_supply_pressure": float(supply / denom),
        "raw_demand_pressure": float(demand / denom),
        "source_diversity_score": 0.65,
    }


def _merge_raw_features(*feature_sets: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for features in feature_sets:
        for name, value in features.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = 0.0
            existing = merged.get(name)
            if existing is None or abs(numeric) > abs(existing):
                merged[name] = numeric
    return merged


def _context_frame(
    *,
    symbol: str,
    as_of: datetime,
    event_time: datetime | None,
    mode: str,
    encoded: LLMContextOutput,
    raw_features: dict[str, float],
) -> pd.DataFrame:
    embedding = list(encoded.event_embedding or [])
    vector_names = list(EventContextVector.model_fields)
    if len(embedding) < len(vector_names):
        embedding.extend([0.0] * (len(vector_names) - len(embedding)))
    row: dict[str, Any] = {
        "timestamp": pd.Timestamp(as_of).isoformat(),
        "symbol": symbol,
        "feature_available_at": pd.Timestamp(as_of).isoformat(),
        "scenario_event_time": pd.Timestamp(event_time).isoformat() if event_time else None,
        "llm_mode": mode,
        "overall_bias": encoded.overall_bias,
        "impact_score": float(encoded.impact_score),
        "uncertainty": float(encoded.uncertainty),
        "event_count": len(encoded.events),
        "llm_input_news_count": 1,
        "explanation": encoded.explanation,
        "warnings": "|".join(encoded.warnings),
    }
    for idx, name in enumerate(vector_names):
        row[name] = float(embedding[idx]) if idx < len(embedding) else 0.0
    for name, value in raw_features.items():
        row[name] = float(value)
    return pd.DataFrame([row])


def _scenario_points_from_bundle(bundle: ForecastBundle) -> list[ScenarioPoint]:
    primary = next(
        (model for model in bundle.forecast_models if str(model.get("id")) == str(bundle.response.primary_model)),
        bundle.forecast_models[0] if bundle.forecast_models else None,
    )
    if primary and primary.get("points"):
        return [ScenarioPoint(time=int(point["time"]), value=float(point["value"])) for point in primary["points"]]
    anchor = ScenarioPoint(time=bundle.market_data.candles[-1].time, value=bundle.response.current_price)
    return [anchor, *[ScenarioPoint(time=point.time, value=point.p50) for point in bundle.response.forecast]]


def _signal_signature(signals: list[ScenarioEventSignal]) -> tuple[tuple[str, str, str, float], ...]:
    return tuple(
        (
            signal.title,
            signal.event_time.isoformat() if signal.event_time else "",
            signal.kind,
            round(float(signal.impact), 6),
        )
        for signal in signals
    )


def _active_signals_for_time(signals: list[ScenarioEventSignal], forecast_time: int) -> list[ScenarioEventSignal]:
    active: list[ScenarioEventSignal] = []
    for signal in signals:
        if signal.event_time is None or int(signal.event_time.timestamp()) <= forecast_time:
            active.append(signal)
    return active


def _encoded_for_active_signals(
    *,
    encoded: LLMContextOutput,
    active_signals: list[ScenarioEventSignal],
    all_signals: list[ScenarioEventSignal],
) -> tuple[LLMContextOutput, dict[str, float]]:
    if not active_signals:
        return encoded.model_copy(
            update={
                "events": [],
                "overall_bias": "neutral",
                "impact_score": 0.0,
                "uncertainty": 0.0,
                "event_embedding": [0.0] * len(EventContextVector.model_fields),
                "explanation": "No scheduled scenario event is active for this forecast horizon.",
                "warnings": [],
            }
        ), _raw_features_from_event_signals([])

    raw_features = (
        _merge_raw_features(_raw_features_from_encoded(encoded), _raw_features_from_event_signals(active_signals))
        if len(all_signals) <= 1
        else _raw_features_from_event_signals(active_signals)
    )
    overall_bias = _bias_from_features(raw_features, encoded.overall_bias)
    active_impact = max([0.0, *[signal.impact for signal in active_signals]])
    total_impact = max(sum(signal.impact for signal in all_signals), active_impact, 1e-8)
    active_ratio = min(max(sum(signal.impact for signal in active_signals) / total_impact, 0.0), 1.0)
    embedding = [float(value) * active_ratio for value in list(encoded.event_embedding or [])]
    return encoded.model_copy(
        update={
            "overall_bias": overall_bias,
            "impact_score": active_impact,
            "event_embedding": embedding,
        }
    ), raw_features


def _context_frame_for_active_signals(
    *,
    symbol: str,
    as_of: datetime,
    mode: str,
    encoded: LLMContextOutput,
    active_signals: list[ScenarioEventSignal],
    all_signals: list[ScenarioEventSignal],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    active_encoded, raw_features = _encoded_for_active_signals(
        encoded=encoded,
        active_signals=active_signals,
        all_signals=all_signals,
    )
    active_event_time = min(
        (signal.event_time for signal in active_signals if signal.event_time is not None),
        default=None,
    )
    frame = _context_frame(
        symbol=symbol,
        as_of=as_of,
        event_time=active_event_time,
        mode=mode,
        encoded=active_encoded,
        raw_features=raw_features,
    )
    frame.loc[:, "event_count"] = len(active_signals)
    frame.loc[:, "llm_input_news_count"] = len(active_signals)
    frame.loc[:, "scenario_schedule_active"] = True
    summary = {
        "active_event_count": len(active_signals),
        "overall_bias": active_encoded.overall_bias,
        "impact_score": float(active_encoded.impact_score),
        "events": [
            {
                "title": signal.title,
                "event_time": signal.event_time.isoformat() if signal.event_time else None,
                "bias": signal.kind,
                "impact": signal.impact,
            }
            for signal in active_signals
        ],
    }
    return frame, summary


def _scheduled_model_bundle(
    *,
    symbol: str,
    interval: str,
    horizon: int | None,
    models: str | None,
    settings: Settings,
    market: Any,
    as_of: datetime,
    mode: str,
    encoded: LLMContextOutput,
    signals: list[ScenarioEventSignal],
    context_summary: dict[str, Any],
) -> tuple[ForecastBundle, list[ScenarioPoint], list[ForecastPoint], dict[str, Any]]:
    bundle_cache: dict[tuple[tuple[str, str, str, float], ...], ForecastBundle] = {}
    frame_cache: dict[tuple[tuple[str, str, str, float], ...], dict[str, Any]] = {}

    def bundle_for(active: list[ScenarioEventSignal]) -> ForecastBundle:
        signature = _signal_signature(active)
        if signature not in bundle_cache:
            frame, active_summary = _context_frame_for_active_signals(
                symbol=market.symbol.provider_symbol,
                as_of=as_of,
                mode=mode,
                encoded=encoded,
                active_signals=active,
                all_signals=signals,
            )
            frame_cache[signature] = active_summary
            bundle_cache[signature] = build_forecast(
                symbol=symbol,
                interval=interval,
                horizon=horizon,
                models=models or "oil_context_fusion",
                include_scenarios=True,
                settings=settings,
                market_override=market,
                event_context_frame_override=frame,
                llm_context_summary_override={
                    **context_summary,
                    "scenario_schedule_active_event_count": len(active),
                    "scenario_schedule_active_events": active_summary["events"],
                },
                apply_event_path_adapter=False,
            )
        return bundle_cache[signature]

    step_seconds = max(int(getattr(market.timeframe, "seconds", 0) or 86_400), 1)
    first_time = int(market.candles[-1].time) + step_seconds
    first_active = _active_signals_for_time(signals, first_time)
    first_bundle = bundle_for(first_active)
    forecast_count = len(first_bundle.response.forecast)
    scheduled_forecast: list[ForecastPoint] = []
    schedule: list[dict[str, Any]] = []
    last_signature: tuple[tuple[str, str, str, float], ...] | None = None

    for idx in range(forecast_count):
        forecast_time = int(first_bundle.response.forecast[idx].time)
        active = _active_signals_for_time(signals, forecast_time)
        signature = _signal_signature(active)
        horizon_bundle = bundle_for(active)
        if idx < len(horizon_bundle.response.forecast):
            scheduled_forecast.append(horizon_bundle.response.forecast[idx])
        if signature != last_signature:
            schedule.append(
                {
                    "from_horizon": idx + 1,
                    "from_time": forecast_time,
                    **frame_cache.get(signature, {}),
                }
            )
            last_signature = signature

    current_price = first_bundle.response.current_price
    anchor_time = first_bundle.market_data.candles[-1].time
    points = [ScenarioPoint(time=anchor_time, value=current_price)] + [
        ScenarioPoint(time=point.time, value=point.p50) for point in scheduled_forecast
    ]
    model_schedule = {
        "mode": "horizon_event_context_schedule",
        "output_postprocessing": False,
        "model_calls": len(bundle_cache),
        "segments": schedule,
    }
    return first_bundle, points, scheduled_forecast, model_schedule


def _external_llm_ready(settings: Settings, mode: str) -> bool:
    if not settings.enable_llm_context or not settings.enable_external_llm_calls:
        return False
    if mode not in _EXTERNAL_LLM_MODES:
        return False
    if mode != "local_http" and not settings.llm_api_key:
        return False
    return True


def build_scenario_forecast(
    *,
    title: str,
    content: str,
    event_time: datetime | None = None,
    events: list[ScenarioEventInput] | None = None,
    symbol: str,
    interval: str,
    horizon: int | None = None,
    models: str | None = None,
    settings: Settings | None = None,
) -> ScenarioForecastResponse:
    settings = settings or get_settings()
    forecast_symbol = settings.default_symbol or "CL=F"
    market = load_market_data_window(forecast_symbol, interval, settings=settings)
    if not market.candles:
        raise ScenarioForecastUnavailable("Scenario forecast requires at least one market candle.")

    resolved_interval = market.timeframe.normalized
    as_of = datetime.fromtimestamp(market.candles[-1].time, tz=timezone.utc)
    event_specs = _event_specs(title=title, content=content, event_time=event_time, events=events)
    event_signals = _signals_from_specs(event_specs)
    scenario_content = _event_specs_content(event_specs, content)
    resolved_event_time = min((spec.event_time for spec in event_specs if spec.event_time is not None), default=_as_utc(event_time))
    mode = _normalize_mode(settings.llm_context_mode) if settings.enable_llm_context else "local_rules"
    warnings: list[str] = []
    warning_objects: list[ForecastWarning] = []
    if resolved_event_time is None:
        warnings.append("No scenario event_time supplied; timing was inferred only from the free-text scenario.")
        warning_objects.append(
            ForecastWarning(
                code="scenario_event_time_missing",
                severity="info",
                message="No scenario event_time supplied; timing was inferred only from the free-text scenario.",
            )
        )
    if not settings.is_development and not _external_llm_ready(settings, mode):
        raise ScenarioForecastUnavailable(
            "Scenario mode requires an enabled external LLM context encoder in production. "
            "Set ENABLE_LLM_CONTEXT=true, ENABLE_EXTERNAL_LLM_CALLS=true, LLM_CONTEXT_MODE, and LLM_API_KEY."
        )

    news = _scenario_news_frame(
        title=title,
        content=scenario_content,
        event_time=resolved_event_time,
        as_of=as_of,
        symbol=market.symbol.provider_symbol,
    )
    provider = _provider_from_scenario(news)
    encoder = encoder_for_mode(
        mode,
        provider=provider,
        api_key=settings.llm_api_key,
        api_base=settings.llm_api_base if mode != "local_http" else settings.local_llm_api_base,
        model=settings.llm_model if mode != "local_http" else settings.local_llm_model,
        live=bool(settings.enable_llm_context and settings.enable_external_llm_calls),
    )
    encoded = encoder.encode_events(
        MarketContextInput(
            symbol=market.symbol.provider_symbol,
            interval=resolved_interval,
            generated_at=as_of,
            news=[
                RawNewsItem(
                    title=title.strip(),
                    source="user_scenario",
                    published_at=as_of,
                    text=_scenario_text(title, scenario_content, resolved_event_time, as_of),
                )
            ],
            forecast_summary={
                "scenario_mode": True,
                "scenario_event_time": resolved_event_time.isoformat() if resolved_event_time else None,
                "instruction": "Encode event context only; numeric price forecasting is handled by the time-series model.",
            },
        )
    )
    warnings.extend(encoded.warnings)
    for warning in encoded.warnings:
        warning_objects.append(ForecastWarning(code="scenario_llm_context_warning", severity="warning", message=str(warning)))

    local_events = provider.load_events()
    raw_features = _merge_raw_features(
        raw_news_pool_features(local_events, as_of_time=as_of, selected_news_count=1),
        _raw_features_from_encoded(encoded),
        _raw_features_from_event_signals(event_signals),
    )
    overall_bias = _bias_from_features(raw_features, encoded.overall_bias)
    effective_impact = max([float(encoded.impact_score or 0.0), 0.0, *[signal.impact for signal in event_signals]])
    effective_encoded = encoded.model_copy(update={"overall_bias": overall_bias, "impact_score": effective_impact})
    context_summary = {
        "enabled": settings.enable_llm_context,
        "external_calls_enabled": settings.enable_external_llm_calls,
        "role": "context/event encoder only",
        "event_context_source": "scenario_override",
        "event_count": len(encoded.events),
        "overall_bias": overall_bias,
        "impact_score": effective_impact,
        "uncertainty": encoded.uncertainty,
        "scenario_event_time": resolved_event_time.isoformat() if resolved_event_time else None,
        "scenario_events": [
            {
                "title": signal.title,
                "event_time": signal.event_time.isoformat() if signal.event_time else None,
                "bias": signal.kind,
                "impact": signal.impact,
            }
            for signal in event_signals
        ],
    }
    bundle, scenario_points, scheduled_forecast, model_context_schedule = _scheduled_model_bundle(
        symbol=symbol,
        interval=resolved_interval,
        horizon=horizon,
        models=models,
        settings=settings,
        market=market,
        as_of=as_of,
        mode=mode,
        encoded=effective_encoded,
        signals=event_signals,
        context_summary=context_summary,
    )
    context_summary["model_context_schedule"] = model_context_schedule
    return ScenarioForecastResponse(
        scenario_id=_scenario_id(title, scenario_content, resolved_event_time),
        title=title.strip(),
        content=scenario_content.strip(),
        symbol=bundle.response.symbol,
        interval=bundle.response.interval,
        generated_at=bundle.response.generated_at,
        event_time=resolved_event_time,
        current_price=bundle.response.current_price,
        points=scenario_points,
        forecast=scheduled_forecast,
        data_status=bundle.response.data_status,
        primary_model=bundle.response.primary_model,
        llm_context_summary=context_summary,
        llm_context=effective_encoded.model_dump(mode="json"),
        warnings=[*warnings, *bundle.response.warnings],
        warning_objects=[*warning_objects, *bundle.response.warning_objects],
    )


__all__ = [
    "ScenarioForecastUnavailable",
    "ScenarioForecastResponse",
    "ScenarioPoint",
    "build_scenario_forecast",
]
