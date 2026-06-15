from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.api.dependencies import service_error
from backend.app.api.routes.report import (
    ForecastReport,
    ReportSection,
    _dominant_regime,
    _fmt_pct,
    _fmt_price,
    _horizon_label,
    _label_direction,
    _local_date,
    _local_datetime,
    _markdown,
    _pct,
    _period_label,
)
from market_ai.config import get_settings
from market_ai.data.related_assets import get_related_assets
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable, load_market_data_window
from market_ai.data.storage import DATA_ROOT, read_table
from market_ai.forecasting.service import ForecastUnavailable, build_forecast
from market_ai.llm.event_encoder import _default_https_context, _extract_json_text
from market_ai.llm.live_context import build_live_event_context
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError
from market_ai.modeling.model_catalog import InvalidModelRequest


router = APIRouter()
_MODEL_COMMENTARY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_DASHBOARD_ANALYSIS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_MARKET_CONTEXT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_LLM_CALL_TIMESTAMPS: list[float] = []
_MODEL_COMMENTARY_CACHE_TTL_SECONDS = 900
_MARKET_CONTEXT_CACHE_TTL_SECONDS = 300
_LLM_CALL_WINDOW_SECONDS = 60
_LLM_CALL_LIMIT_PER_WINDOW = 12
_MODEL_COMMENTARY_PROMPT_VERSION = "external-required-v3"
_DASHBOARD_ANALYSIS_PROMPT_VERSION = "split-panels-v1"
_DASHBOARD_ANALYSIS_PANEL_COUNT = 3
_NON_PUBLIC_CONTEXT_FRAGMENTS = (
    "deterministic local event context encoder",
    "structured context only",
)


class AssistantChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1200)
    symbol: str | None = None
    interval: str | None = None
    horizon: int | None = Field(default=None, ge=1)
    models: str | None = None
    language: str = "ko"


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return read_table(path)
    except Exception:
        return pd.DataFrame()


def _iso(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.isoformat()


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def _public_context_text(value: Any) -> str:
    text = _text(value).strip()
    if not text or text == "-":
        return ""
    lowered = text.lower()
    if any(fragment in lowered for fragment in _NON_PUBLIC_CONTEXT_FRAGMENTS):
        return ""
    return text


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _context_symbol_set(symbol: str) -> set[str]:
    return {symbol.upper(), "ALL", "*", *(item.upper() for item in get_related_assets(symbol))}


def _filter_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns:
        return frame
    return frame[frame["symbol"].astype(str).str.upper().isin(_context_symbol_set(symbol))].copy()


def _spread_frame_by_time(frame: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    if frame.empty or len(frame) <= limit:
        return frame
    if limit <= 1:
        return frame.tail(1)
    indexes = sorted({round(idx * (len(frame) - 1) / (limit - 1)) for idx in range(limit)})
    return frame.iloc[indexes]


def _news_items(
    *,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    limit: int,
    sampling: str = "tail",
) -> list[dict[str, Any]]:
    frame = _read_optional(DATA_ROOT / "raw" / "news" / "public_market_news.csv")
    if frame.empty:
        return []
    frame = _filter_symbol(frame, symbol)
    frame["published_at"] = pd.to_datetime(frame.get("published_at"), errors="coerce", utc=True)
    frame = frame.dropna(subset=["published_at"])
    frame = frame[(frame["published_at"] >= start_ts) & (frame["published_at"] <= end_ts)].sort_values("published_at")
    selected = _spread_frame_by_time(frame, limit=limit) if sampling == "spread" else frame.tail(limit)
    rows = []
    for row in selected.to_dict(orient="records"):
        rows.append(
            {
                "time": int(pd.Timestamp(row["published_at"]).timestamp()),
                "published_at": _iso(row.get("published_at")),
                "symbol": row.get("symbol"),
                "headline": _text(row.get("headline")),
                "source": _text(row.get("source")),
                "url": _text(row.get("url")),
            }
        )
    return rows


def _news_items_from_frame(frame: pd.DataFrame, *, limit: int) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    out = frame.copy()
    out["published_at"] = pd.to_datetime(out.get("published_at"), errors="coerce", utc=True)
    out = out.dropna(subset=["published_at"]).sort_values("published_at")
    rows = []
    for row in out.tail(limit).to_dict(orient="records"):
        rows.append(
            {
                "time": int(pd.Timestamp(row["published_at"]).timestamp()),
                "published_at": _iso(row.get("published_at")),
                "symbol": row.get("symbol"),
                "headline": _text(row.get("headline")),
                "source": _text(row.get("source")),
                "url": _text(row.get("url")),
            }
        )
    return rows


def _context_points_from_frame(frame: pd.DataFrame, *, limit: int) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out.get("timestamp"), errors="coerce", utc=True)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    rows = []
    for row in out.tail(limit).to_dict(orient="records"):
        ts = pd.Timestamp(row["timestamp"])
        rows.append(
            {
                "time": int(ts.timestamp()),
                "timestamp": ts.isoformat(),
                "symbol": row.get("symbol"),
                "llm_mode": row.get("llm_mode"),
                "overall_bias": row.get("overall_bias"),
                "impact_score": float(pd.to_numeric(pd.Series([row.get("impact_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                "uncertainty": float(pd.to_numeric(pd.Series([row.get("uncertainty", 1.0)]), errors="coerce").fillna(1.0).iloc[0]),
                "event_count": int(pd.to_numeric(pd.Series([row.get("event_count", 0)]), errors="coerce").fillna(0).iloc[0]),
                "explanation": _text(row.get("explanation")),
                "warnings": _text(row.get("warnings")),
            }
        )
    return rows


def _context_points(
    *,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    limit: int,
    sampling: str = "tail",
) -> list[dict[str, Any]]:
    frame = _read_optional(DATA_ROOT / "processed" / "event_context" / "event_context_daily.csv")
    if frame.empty:
        return []
    frame = _filter_symbol(frame, symbol)
    frame["timestamp"] = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"])
    frame = frame[(frame["timestamp"] >= start_ts.floor("D")) & (frame["timestamp"] <= end_ts.ceil("D"))].sort_values("timestamp")
    if "event_count" in frame.columns:
        event_count = pd.to_numeric(frame["event_count"], errors="coerce").fillna(0.0)
        impact = pd.to_numeric(frame.get("impact_score", 0.0), errors="coerce").fillna(0.0)
        frame = frame[(event_count > 0) | (impact > 0)]
    selected = _spread_frame_by_time(frame, limit=limit) if sampling == "spread" else frame.tail(limit)
    rows = []
    for row in selected.to_dict(orient="records"):
        ts = pd.Timestamp(row["timestamp"])
        rows.append(
            {
                "time": int(ts.timestamp()),
                "timestamp": ts.isoformat(),
                "symbol": row.get("symbol"),
                "llm_mode": row.get("llm_mode"),
                "overall_bias": row.get("overall_bias"),
                "impact_score": float(pd.to_numeric(pd.Series([row.get("impact_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                "uncertainty": float(pd.to_numeric(pd.Series([row.get("uncertainty", 1.0)]), errors="coerce").fillna(1.0).iloc[0]),
                "event_count": int(pd.to_numeric(pd.Series([row.get("event_count", 0)]), errors="coerce").fillna(0).iloc[0]),
                "explanation": _text(row.get("explanation")),
                "warnings": _text(row.get("warnings")),
            }
        )
    return rows


def _spread_by_time(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    ordered = sorted(
        [row for row in rows if row.get("time") is not None],
        key=lambda row: int(row.get("time") or 0),
    )
    if len(ordered) <= limit:
        return ordered
    if limit <= 1:
        return ordered[-1:]
    indexes = {round(idx * (len(ordered) - 1) / (limit - 1)) for idx in range(limit)}
    return [ordered[idx] for idx in sorted(indexes)]


def _nearest_context_point(context_points: list[dict[str, Any]], news_time: int) -> dict[str, Any]:
    if not context_points:
        return {}
    return min(context_points, key=lambda point: abs(int(point.get("time") or 0) - news_time))


def _time_span_days(rows: list[dict[str, Any]]) -> float:
    times = [int(row.get("time") or 0) for row in rows if row.get("time") is not None]
    if len(times) < 2:
        return 0.0
    return (max(times) - min(times)) / 86_400


def _news_near_time(news: list[dict[str, Any]], target_time: int, *, max_gap_days: int = 3) -> list[dict[str, Any]]:
    max_gap = max_gap_days * 86_400
    nearby = [
        row
        for row in news
        if abs(int(row.get("time") or 0) - target_time) <= max_gap
    ]
    return sorted(nearby, key=lambda row: abs(int(row.get("time") or 0) - target_time))


def _coalesce_nearby_chart_points(
    points: list[dict[str, Any]],
    *,
    min_gap_seconds: int = 2 * 86_400,
) -> list[dict[str, Any]]:
    def unique_news(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[int, str]] = set()
        unique: list[dict[str, Any]] = []
        for item in items:
            key = (int(item.get("time") or 0), str(item.get("headline") or ""))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    if not points:
        return []
    ordered = sorted(points, key=lambda point: int(point.get("time") or 0))
    merged: list[dict[str, Any]] = []
    for point in ordered:
        point_time = int(point.get("time") or 0)
        if merged and point_time - int(merged[-1].get("time") or 0) < min_gap_seconds:
            previous = merged[-1]
            previous_news = list(previous.get("news_items") or [])
            next_news = list(point.get("news_items") or [])
            merged_news = unique_news(previous_news + next_news)
            previous["news_items"] = merged_news[:8]
            previous["event_count"] = len(merged_news)
            if point.get("impact_score") and float(point.get("impact_score") or 0.0) > float(previous.get("impact_score") or 0.0):
                previous["overall_bias"] = point.get("overall_bias") or previous.get("overall_bias")
                previous["impact_score"] = point.get("impact_score")
                previous["uncertainty"] = point.get("uncertainty")
                previous["explanation"] = point.get("explanation") or previous.get("explanation")
            continue
        merged.append({**point})
    return merged


def _chart_context_points(
    *,
    news: list[dict[str, Any]],
    context_points: list[dict[str, Any]],
    limit: int = 6,
) -> list[dict[str, Any]]:
    news_span_days = _time_span_days(news)
    context_span_days = _time_span_days(context_points)
    use_context_window = bool(context_points) and (
        not news or (context_span_days >= 21 and news_span_days < min(21, context_span_days * 0.35))
    )
    if use_context_window:
        markers = []
        for point in _spread_by_time(context_points, limit=limit):
            point_time = int(point.get("time") or 0)
            nearby_news = _news_near_time(news, point_time)
            markers.append(
                {
                    **point,
                    "time": point_time,
                    "event_count": len(nearby_news) or int(point.get("event_count") or 0),
                    "news_items": nearby_news[:5],
                }
            )
        return _coalesce_nearby_chart_points(markers)
    if news:
        selected_news = _spread_by_time(news, limit=limit)
        markers: list[dict[str, Any]] = []
        for item in selected_news:
            news_time = int(item.get("time") or 0)
            nearby_news = _news_near_time(news, news_time)
            point = _nearest_context_point(context_points, news_time)
            markers.append(
                {
                    "time": news_time,
                    "overall_bias": point.get("overall_bias") or "neutral",
                    "impact_score": float(point.get("impact_score") or 0.0),
                    "uncertainty": float(point.get("uncertainty") or 1.0),
                    "event_count": len(nearby_news) or 1,
                    "explanation": point.get("explanation") or item.get("headline") or "",
                    "news_items": nearby_news[:5] or [item],
                }
            )
        return _coalesce_nearby_chart_points(markers)
    return _coalesce_nearby_chart_points(_spread_by_time(context_points, limit=limit))


def _scenario_commentary(bundle) -> dict[str, Any]:
    response = bundle.response
    first = response.forecast[0] if response.forecast else None
    last = response.forecast[-1] if response.forecast else None
    direction = "flat"
    if last is not None and last.p50 > response.current_price:
        direction = "upside"
    elif last is not None and last.p50 < response.current_price:
        direction = "downside"
    regime_values = response.regime.model_dump()
    regime_label = max((key for key in regime_values if key != "confidence"), key=lambda key: regime_values[key])
    return {
        "mode": "deterministic_context_narrative",
        "summary": (
            f"{response.primary_model or 'selected model'} median path currently leans {direction}; "
            f"dominant regime is {regime_label}. News and event context is used only to explain drivers and risks."
        ),
        "bull": "Bull case follows the upper quantile path when trend, event impulse, and volatility remain favorable.",
        "base": "Base case follows the median cumulative log-return path restored back into price space.",
        "bear": "Bear case follows the lower quantile path when volatility expands or recent event/context pressure turns adverse.",
        "confidence_warning": None
        if first and first.confidence >= 0.45
        else "Scenario text is explanatory only; probability bands are not validated confidence intervals unless calibration says so.",
    }


def _model_path_summaries(bundle) -> list[dict[str, Any]]:
    rows = []
    current_price = float(bundle.response.current_price)
    for model in bundle.forecast_models:
        points = model.get("points") or []
        if len(points) < 2:
            continue
        start = float(points[0].get("value", current_price))
        end = float(points[-1].get("value", start))
        pct_change = ((end / start) - 1.0) * 100.0 if start else 0.0
        direction = "flat"
        if pct_change > 0.25:
            direction = "up"
        elif pct_change < -0.25:
            direction = "down"
        path_adapter = (bundle.response.deep_model_info.get(str(model.get("id")) or "", {}) or {}).get("path_adapter") or {}
        rows.append(
            {
                "id": model.get("id"),
                "label": model.get("label") or model.get("id"),
                "direction": direction,
                "pct_change": round(pct_change, 3),
                "start": round(start, 4),
                "end": round(end, 4),
                "steps": max(0, len(points) - 1),
                "path_adapter": path_adapter,
            }
        )
    return rows


def _adapter_analyst_read(path_adapter: dict[str, Any] | None, *, language: str) -> str:
    adapter = str((path_adapter or {}).get("adapter") or "")
    if not adapter:
        return ""
    lang = _language(language)
    if adapter == "geopolitical_supply_shock":
        return (
            "recent geopolitical and supply-risk headlines raised the supply-risk premium"
            if lang == "en"
            else "최근 지정학 및 공급 차질 뉴스가 공급 리스크 프리미엄을 높였습니다"
        )
    if adapter == "bullish_event_breakout":
        return (
            "event tone and recent momentum support an upside breakout setup"
            if lang == "en"
            else "이벤트 분위기와 최근 모멘텀이 상방 돌파 가능성을 뒷받침합니다"
        )
    if adapter == "event_risk_premium":
        return (
            "persistent geopolitical and energy headlines added an upside risk premium"
            if lang == "en"
            else "지속적인 지정학 및 에너지 뉴스 압력이 상방 리스크 프리미엄을 더했습니다"
        )
    if adapter == "overextended_mean_reversion":
        return (
            "recent price action looks overextended, so the path allows a pullback before stabilization"
            if lang == "en"
            else "최근 가격 흐름이 과열권이라 되돌림 후 안정 경로를 반영합니다"
        )
    if adapter == "pattern_residual_detemplate":
        return (
            "the path follows recent chart pattern variation rather than a repeated horizon template"
            if lang == "en"
            else "반복 템플릿이 아니라 최근 차트 패턴 변화를 반영합니다"
        )
    return ""


def _price_action_snapshot(bundle) -> dict[str, Any]:
    candles = bundle.market_data.candles
    close = [float(candle.close) for candle in candles if candle.close and candle.close > 0]
    if len(close) < 2:
        return {
            "short_trend": "unknown",
            "medium_trend": "unknown",
            "range_position": "unknown",
            "recent_change_pct": None,
            "medium_change_pct": None,
            "pattern_read": "insufficient chart history",
        }
    last = close[-1]

    def pct_change(steps: int) -> float | None:
        if len(close) <= steps or close[-steps - 1] <= 0:
            return None
        return (last / close[-steps - 1] - 1.0) * 100.0

    short_change = pct_change(min(5, len(close) - 1))
    medium_change = pct_change(min(20, len(close) - 1))
    window = close[-min(40, len(close)) :]
    low = min(window)
    high = max(window)
    pos = (last - low) / max(high - low, 1e-8)
    short_trend = "up" if short_change is not None and short_change > 0.75 else "down" if short_change is not None and short_change < -0.75 else "sideways"
    medium_trend = "up" if medium_change is not None and medium_change > 2.0 else "down" if medium_change is not None and medium_change < -2.0 else "sideways"
    range_position = "upper" if pos >= 0.66 else "lower" if pos <= 0.34 else "middle"
    if short_trend == "down" and range_position == "upper":
        pattern_read = "recent pullback from the upper part of the range"
    elif short_trend == "up" and range_position == "lower":
        pattern_read = "rebound attempt from the lower part of the range"
    elif medium_trend == "up":
        pattern_read = "uptrend with short-term consolidation"
    elif medium_trend == "down":
        pattern_read = "downtrend with fragile rebounds"
    else:
        pattern_read = "range-bound trade without a decisive breakout"
    return {
        "short_trend": short_trend,
        "medium_trend": medium_trend,
        "range_position": range_position,
        "recent_change_pct": round(short_change, 3) if short_change is not None else None,
        "medium_change_pct": round(medium_change, 3) if medium_change is not None else None,
        "pattern_read": pattern_read,
    }


def _commentary_market_context(bundle, settings, *, limit: int = 12, origin_time: str | None = None) -> dict[str, Any]:
    candles = bundle.market_data.candles
    resolved_symbol = bundle.response.symbol
    if candles:
        end_ts = pd.to_datetime(candles[-1].time, unit="s", utc=True)
        if not origin_time:
            end_ts = max(end_ts, pd.Timestamp(datetime.now(timezone.utc)))
        start_ts = end_ts - pd.Timedelta(days=128 if origin_time else 90)
    else:
        end_ts = pd.Timestamp(datetime.now(timezone.utc))
        start_ts = end_ts - pd.Timedelta(days=90)
    pool_limit = max(limit, 128)
    cached_news_pool = _news_items(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=pool_limit)
    cached_context_pool = _context_points(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=pool_limit)
    chart_news_pool = _news_items(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=pool_limit, sampling="spread")
    chart_context_pool = _context_points(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=pool_limit, sampling="spread")
    cached_news = cached_news_pool[-limit:]
    cached_context_points = cached_context_pool[-limit:]
    if origin_time:
        return {
            "source": "point_in_time_news_cache",
            "news": cached_news[-limit:],
            "context_points": cached_context_points[-limit:],
            "warnings": [],
        }
    try:
        live_payload = _load_live_context_payload(symbol=resolved_symbol, settings=settings, limit=limit)
        news = live_payload.get("news") or cached_news
        context_points = live_payload.get("context_points") or cached_context_points
        return {
            "source": live_payload.get("source") or "live_public_news",
            "news": news[-limit:],
            "context_points": context_points[-limit:],
            "warnings": [str(item) for item in live_payload.get("warnings") or []],
        }
    except Exception as exc:
        return {
            "source": "offline_cache",
            "news": cached_news[-limit:],
            "context_points": cached_context_points[-limit:],
            "warnings": [f"Live context unavailable; offline context used: {exc}"],
        }


def _language(value: str | None) -> str:
    return "en" if str(value or "").lower().startswith("en") else "ko"


def _display_symbol(bundle) -> str:
    return bundle.response.symbol


def _symbol_payload(bundle) -> dict[str, str]:
    display = _display_symbol(bundle)
    provider = bundle.response.symbol
    return {
        "display_symbol": display,
        "provider_symbol": provider,
    }


def _reserve_llm_calls(count: int = 1) -> bool:
    now = time.time()
    cutoff = now - _LLM_CALL_WINDOW_SECONDS
    _LLM_CALL_TIMESTAMPS[:] = [stamp for stamp in _LLM_CALL_TIMESTAMPS if stamp >= cutoff]
    requested = max(1, int(count))
    if len(_LLM_CALL_TIMESTAMPS) + requested > _LLM_CALL_LIMIT_PER_WINDOW:
        return False
    _LLM_CALL_TIMESTAMPS.extend([now] * requested)
    return True


def _reserve_llm_call() -> bool:
    return _reserve_llm_calls(1)


def _commentary_cache_key(bundle, model_summaries: list[dict[str, Any]], origin_time: str | None, language: str) -> str:
    model_fingerprint = [
        {
            "id": item.get("id"),
            "direction": item.get("direction"),
            "steps": item.get("steps"),
            "pct_bucket": round(float(item.get("pct_change") or 0.0), 1),
            "path_adapter": (item.get("path_adapter") or {}).get("adapter"),
        }
        for item in model_summaries
    ]
    candles = getattr(bundle.market_data, "candles", []) or []
    last_candle_time = candles[-1].time if candles else None
    payload = {
        "symbol": bundle.response.symbol,
        "display_symbol": _display_symbol(bundle),
        "interval": bundle.response.interval,
        "primary_model": bundle.response.primary_model,
        "horizon": bundle.horizon,
        "origin_time": origin_time,
        "last_candle_time": last_candle_time,
        "language": _language(language),
        "commentary_prompt_version": _MODEL_COMMENTARY_PROMPT_VERSION,
        "models": model_fingerprint,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _extract_commentary_json(raw: str) -> dict[str, Any]:
    text = _extract_json_text(raw).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM commentary response must be a JSON object.")
    return parsed


def _llm_request_timeout_seconds(settings) -> float:
    return max(1.0, float(getattr(settings, "llm_request_timeout_seconds", 45.0)))


def _google_model_commentary(settings, prompt: str) -> dict[str, Any]:
    base = settings.llm_api_base.strip().rstrip("/")
    if "openai" in base or "chat/completions" in base:
        base = "https://generativelanguage.googleapis.com/v1beta"
    if ":generateContent" in base:
        url = base
    else:
        encoded_model = urllib.parse.quote(settings.llm_model, safe="-_.~/")
        url = f"{base}/models/{encoded_model}:generateContent"
    separator = "&" if "?" in url else "?"
    url = f"{url}{separator}key={urllib.parse.quote(settings.llm_api_key or '')}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_llm_request_timeout_seconds(settings), context=_default_https_context()) as response:
        body = json.loads(response.read().decode("utf-8"))
    parts = body["candidates"][0]["content"]["parts"]
    content = "".join(str(part.get("text", "")) for part in parts if not part.get("thought"))
    return _extract_commentary_json(content)


def _openai_compatible_model_commentary(settings, prompt: str) -> dict[str, Any]:
    payload = {
        "model": settings.llm_model,
        "messages": [
            {
                "role": "system",
                "content": "Explain model forecast outputs. Do not create new price targets or trading instructions. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        settings.llm_api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_llm_request_timeout_seconds(settings), context=_default_https_context()) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _extract_commentary_json(body["choices"][0]["message"]["content"])


def _llm_unavailable_message(reason: str, language: str = "ko", detail: str | None = None) -> str:
    lang = _language(language)
    if lang == "en":
        base = "The AI analyst is not responding."
        reasons = {
            "not_configured": "External LLM calls are not configured.",
            "rate_guard": "The local external LLM request guard is active. Please retry shortly.",
            "provider": "The external LLM provider did not respond successfully.",
        }
        suffix = reasons.get(reason, "External LLM service is unavailable.")
        return f"{base} {suffix}{(' ' + detail) if detail else ''}"
    base = "인공지능 해설가가 응답하지 않아요."
    reasons = {
        "not_configured": "외부 LLM 설정을 확인해 주세요.",
        "rate_guard": "외부 LLM 요청 제한이 적용 중입니다. 잠시 후 다시 시도해 주세요.",
        "provider": "외부 LLM 제공자가 정상 응답하지 않았습니다.",
    }
    suffix = reasons.get(reason, "외부 LLM 연결 또는 사용량을 확인해 주세요.")
    return f"{base} {suffix}{(' ' + detail) if detail else ''}"


def _llm_model_commentary(
    settings,
    bundle,
    model_summaries: list[dict[str, Any]],
    origin_time: str | None,
    language: str = "ko",
    market_context: dict[str, Any] | None = None,
    price_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lang = _language(language)
    market_context = market_context or {}
    price_action = price_action or _price_action_snapshot(bundle)
    if not settings.enable_external_llm_calls or not settings.llm_api_key:
        raise HTTPException(
            status_code=503,
            detail={"message": _llm_unavailable_message("not_configured", lang), "warnings": []},
        )
    if not _reserve_llm_call():
        raise HTTPException(
            status_code=503,
            detail={"message": _llm_unavailable_message("rate_guard", lang), "warnings": []},
        )
    output_language = "English" if lang == "en" else "Korean"
    prompt = (
        "너는 원유 시장 애널리스트다. "
        "아래 JSON에는 이미 계산된 단일 운영 모델(oil_context_fusion)의 예측 경로, 최근 차트 흐름, 뉴스/이벤트 맥락이 있다. "
        "새로운 가격 목표, 새 수익률 경로, 매매 지시를 만들지 말고 제공된 정보만 바탕으로 왜 이런 방향성이 나왔는지 설명하라. "
        "사용자가 읽는 화면에 들어갈 글이므로 내부 기술 설명을 하지 마라. "
        "금지어: 텐서, 피처 벡터, 보정 상태, calibration, coverage, quantile, 분위수, 잔차, data status, 신뢰구간. "
        "영어 뉴스 제목을 원문 그대로 인용하지 말고, 요청 언어로 번역하거나 의미를 풀어서 설명하라. "
        "문장 앞에 bullet marker를 붙이지 말고 본문 문단처럼 쓸 수 있는 완전한 문장으로 작성하라. "
        "뉴스, 공급/수요, 재고, OPEC, 달러/금리, 위험선호, 차트 흐름 같은 애널리스트 언어로 설명하라. "
        "복수 모델 비교 표현은 쓰지 마라. "
        f"응답 언어는 반드시 {output_language}로 맞춰라. "
        "반드시 JSON object만 반환하라. 필수 key: summary, model_interpretation, risk_notes, warnings. "
        "summary는 2문장 이내, model_interpretation은 핵심 근거 1문단, risk_notes는 2~3개 문자열 배열, warnings는 필요할 때만 짧게 작성하라.\n\n"
        + json.dumps(
            {
                "symbol": bundle.response.symbol,
                "display_symbol": _display_symbol(bundle),
                "provider_symbol": bundle.response.symbol,
                "interval": bundle.response.interval,
                "primary_model": bundle.response.primary_model,
                "current_price": round(float(bundle.response.current_price), 4),
                "horizon": bundle.horizon,
                "origin_time": origin_time,
                "regime": bundle.response.regime.model_dump(),
                "price_action": price_action,
                "market_context": {
                    "source": market_context.get("source"),
                    "latest_news": (market_context.get("news") or [])[-8:],
                    "latest_context_points": (market_context.get("context_points") or [])[-3:],
                },
                "model_summaries": model_summaries,
                "forecast_context": {
                    "primary_path_adapter": (
                        next(
                            (
                                item.get("path_adapter")
                                for item in model_summaries
                                if item.get("id") == bundle.response.primary_model
                            ),
                            {},
                        )
                        or {}
                    ),
                },
            },
            ensure_ascii=False,
        )
    )
    try:
        if settings.llm_context_mode == "google_generative" or "generativelanguage.googleapis.com" in settings.llm_api_base or settings.llm_model.lower().startswith("gemma-"):
            raw = _google_model_commentary(settings, prompt)
        else:
            raw = _openai_compatible_model_commentary(settings, prompt)
        return {
            "symbol": bundle.response.symbol,
            **_symbol_payload(bundle),
            "interval": bundle.response.interval,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "llm_model_commentary",
            "summary": str(raw.get("summary") or ""),
            "model_interpretation": str(raw.get("model_interpretation") or raw.get("model_agreement") or ""),
            "model_agreement": "",
            "divergence": "",
            "risk_notes": _string_list(raw.get("risk_notes")),
            "market_context": market_context,
            "price_action": price_action,
            "model_summaries": model_summaries,
            "llm_used": True,
            "warnings": _string_list(raw.get("warnings")),
        }
    except (KeyError, RuntimeError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": _llm_provider_error_message(exc, language=lang),
                "warnings": [],
            },
        ) from exc


def _forecast_bundle_for_commentary(
    *,
    symbol: str,
    interval: str,
    horizon: int | None,
    models: str | None,
    origin_time: str | None,
    settings,
):
    if not origin_time:
        return build_forecast(symbol=symbol, interval=interval, horizon=horizon, models=models, settings=settings)

    from backend.app.api.routes.backtests import _origin_index, _parse_origin_time, _point_in_time_window

    origin_ts = _parse_origin_time(origin_time)
    full_market = load_market_data_window(symbol, interval, settings=settings)
    candles = sorted(full_market.candles, key=lambda candle: candle.time)
    origin_idx = _origin_index(candles, origin_ts)
    point_in_time_market = _point_in_time_window(
        full_market.model_copy(update={"candles": candles}),
        candles[: origin_idx + 1],
    )
    return build_forecast(
        symbol=symbol,
        interval=interval,
        horizon=horizon,
        models=models,
        allow_removed_models_as_warning=True,
        settings=settings,
        market_override=point_in_time_market,
    )


def _live_context_cache_key(*, symbol: str, limit: int, settings) -> str:
    payload = {
        "symbol": symbol,
        "limit": limit,
        "llm_context_enabled": bool(settings.enable_llm_context),
        "external_llm_enabled": bool(settings.enable_external_llm_calls),
        "llm_mode": settings.llm_context_mode,
        "llm_model": settings.llm_model,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _load_live_context_payload(*, symbol: str, settings, limit: int) -> dict[str, Any]:
    key = _live_context_cache_key(symbol=symbol, settings=settings, limit=limit)
    now = time.time()
    cached = _MARKET_CONTEXT_CACHE.get(key)
    if cached and now - cached[0] <= _MARKET_CONTEXT_CACHE_TTL_SECONDS:
        return {**cached[1], "cached": True}

    effective_settings = settings
    warnings: list[str] = []
    if settings.enable_external_llm_calls and not _reserve_llm_call():
        effective_settings = settings.model_copy(update={"enable_external_llm_calls": False})
        warnings.append("External LLM rate limit guard active; live context encoded with local rules for this refresh.")

    live_context = build_live_event_context(
        symbol=symbol,
        settings=effective_settings,
        as_of_time=datetime.now(timezone.utc),
        news_limit=limit,
    )
    payload = {
        "news": _news_items_from_frame(live_context.get("news"), limit=limit),
        "context_points": _context_points_from_frame(live_context.get("context_frame"), limit=limit),
        "warnings": [*warnings, *[str(item) for item in live_context.get("warnings") or []]],
        "source": str(live_context.get("source") or "live_public_news"),
        "cached": False,
    }
    _MARKET_CONTEXT_CACHE[key] = (now, payload)
    return payload


def _external_chat_answer_is_public(answer: str) -> bool:
    text = answer.strip().lower()
    if not text:
        return False
    forbidden_terms = (
        "oil_context_fusion",
        "oil context fusion",
        "primary_model",
        "target_price",
        "bias=",
        "impact=",
        "event score",
        "model-calculated",
        "pattern residual",
        "template",
        "adapter",
        "feature",
        "score",
        "단일 운영 모델",
        "내부 모델",
        "점수",
        "기술적 변수",
    )
    return not any(term in text for term in forbidden_terms)


def _public_chart_read(price_action: dict[str, Any], *, language: str) -> dict[str, Any]:
    lang = _language(language)
    pattern = str(price_action.get("pattern_read") or "range-bound trade without a decisive breakout")
    if lang == "en":
        return {
            "short_term": str(price_action.get("short_trend") or "unknown"),
            "medium_term": str(price_action.get("medium_trend") or "unknown"),
            "range_position": str(price_action.get("range_position") or "unknown"),
            "recent_move_pct": price_action.get("recent_change_pct"),
            "analyst_read": pattern,
        }
    pattern_ko = {
        "recent pullback from the upper part of the range": "최근 고점권에서 밀린 뒤 방향을 다시 확인하는 흐름",
        "rebound attempt from the lower part of the range": "최근 저점권에서 반등을 시도하는 흐름",
        "uptrend with short-term consolidation": "상승 추세 안에서 단기 숨고르기가 섞인 흐름",
        "downtrend with fragile rebounds": "하락 추세 안에서 반등이 아직 취약한 흐름",
        "range-bound trade without a decisive breakout": "뚜렷한 돌파 없이 박스권에서 흔들리는 흐름",
        "insufficient chart history": "차트 이력이 부족해 흐름 판단이 제한적인 상태",
    }.get(pattern, pattern)
    trend_ko = {"up": "상승", "down": "하락", "sideways": "횡보", "unknown": "불명"}
    position_ko = {"upper": "상단", "middle": "중간", "lower": "하단", "unknown": "불명"}
    return {
        "short_term": trend_ko.get(str(price_action.get("short_trend") or "unknown"), "불명"),
        "medium_term": trend_ko.get(str(price_action.get("medium_trend") or "unknown"), "불명"),
        "range_position": position_ko.get(str(price_action.get("range_position") or "unknown"), "불명"),
        "recent_move_pct": price_action.get("recent_change_pct"),
        "analyst_read": pattern_ko,
    }


def _public_news_evidence(context_payload: dict[str, Any] | None, *, limit: int = 6) -> list[dict[str, Any]]:
    rows = []
    for item in ((context_payload or {}).get("news") or [])[-limit:]:
        headline = str(item.get("headline") or "").strip()
        if not headline:
            continue
        rows.append(
            {
                "headline": headline,
                "source": str(item.get("source") or "").strip() or "unknown",
                "date": _iso(item.get("time")) or _iso(item.get("published_at")) or None,
            }
        )
    return rows


def _llm_provider_error_message(exc: BaseException, language: str = "ko") -> str:
    if isinstance(exc, urllib.error.HTTPError):
        detail = f"HTTP {exc.code}"
        return _llm_unavailable_message("provider", language, detail)
    if isinstance(exc, urllib.error.URLError):
        return _llm_unavailable_message("provider", language, str(exc.reason))
    return _llm_unavailable_message("provider", language, str(exc))


def _market_context_payload_from_bundle(
    *,
    bundle,
    settings,
    limit: int,
    live: bool,
    origin_time: str | None,
) -> dict[str, Any]:
    candles = bundle.market_data.candles
    if origin_time:
        from backend.app.api.routes.backtests import _parse_origin_time

        origin_ts = _parse_origin_time(origin_time)
        end_ts = pd.to_datetime(origin_ts, unit="s", utc=True)
        start_ts = end_ts - pd.Timedelta(days=128)
    elif candles:
        end_ts = max(pd.to_datetime(candles[-1].time, unit="s", utc=True), pd.Timestamp(datetime.now(timezone.utc)))
        start_ts = end_ts - pd.Timedelta(days=128)
    else:
        end_ts = pd.Timestamp(datetime.now(timezone.utc))
        start_ts = end_ts - pd.Timedelta(days=120)

    resolved_symbol = bundle.response.symbol
    pool_limit = max(limit, 128)
    cached_news_pool = _news_items(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=pool_limit)
    cached_context_pool = _context_points(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=pool_limit)
    chart_news_pool = _news_items(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=pool_limit, sampling="spread")
    chart_context_pool = _context_points(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=pool_limit, sampling="spread")
    cached_news = cached_news_pool[-limit:]
    cached_context_points = cached_context_pool[-limit:]
    live_news: list[dict[str, Any]] = []
    live_context_points: list[dict[str, Any]] = []
    live_warnings: list[str] = []
    live_source = "offline_cache"
    if live and not origin_time:
        live_source = "live_public_news"
        try:
            live_payload = _load_live_context_payload(symbol=resolved_symbol, settings=settings, limit=limit)
            live_news = live_payload["news"]
            live_context_points = live_payload["context_points"]
            live_warnings = [str(item) for item in live_payload.get("warnings") or []]
            live_source = str(live_payload.get("source") or "live_public_news")
            if live_payload.get("cached"):
                live_source = f"{live_source}_cached"
        except Exception as exc:
            live_source = "live_public_news_unavailable"
            live_warnings = [f"Live news context unavailable: {exc}"]
    response_news = live_news if live and not origin_time else cached_news
    response_context_points = live_context_points if live and not origin_time else cached_context_points
    chart_news_pool = chart_news_pool or cached_news_pool or live_news
    chart_context_pool = chart_context_pool or cached_context_pool or live_context_points
    chart_context_points = _chart_context_points(
        news=chart_news_pool,
        context_points=chart_context_pool,
        limit=6,
    )
    news_source = live_source if live and not origin_time else "point_in_time_news_cache" if origin_time else "offline_cache"
    return {
        "symbol": resolved_symbol,
        **_symbol_payload(bundle),
        "interval": bundle.response.interval,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_origin_time": origin_time,
        "llm_context_summary": bundle.response.llm_context_summary,
        "news": response_news,
        "context_points": response_context_points,
        "chart_context_points": chart_context_points,
        "news_source": news_source,
        "news_warnings": live_warnings,
        "offline_cache_available": {
            "news_count": len(cached_news_pool),
            "context_point_count": len(cached_context_pool),
        },
        "scenario_commentary": _scenario_commentary(bundle),
        "primary_model": bundle.response.primary_model,
        "calibration_status": bundle.response.calibration_status,
    }


def _dashboard_forecast_facts(
    bundle,
    language: str,
    generated_at: datetime | None = None,
    origin_time: str | None = None,
) -> dict[str, Any]:
    lang = _language(language)
    generated = generated_at or datetime.now(timezone.utc)
    response = bundle.response
    first = response.forecast[0] if response.forecast else None
    last = response.forecast[-1] if response.forecast else None
    current = float(response.current_price)
    median_end = float(last.p50) if last else current
    p10_end = float(last.p10) if last else None
    p90_end = float(last.p90) if last else None
    median_change = _pct(current, median_end)
    direction = "upside" if median_change > 0.25 else "downside" if median_change < -0.25 else "range-bound"
    regime = _dominant_regime(response.regime)
    horizon = len(response.forecast)
    horizon_text = _horizon_label(horizon, response.interval, lang)
    period_text = _period_label(first.time if first else None, last.time if last else None, lang)
    regime_label = _label_direction(regime, lang)
    checkpoint_steps = []
    for step in (7, 14, horizon):
        if step >= 1 and step <= horizon and step not in checkpoint_steps:
            checkpoint_steps.append(step)
    checkpoints = []
    for step in checkpoint_steps:
        point = response.forecast[step - 1]
        p50 = float(point.p50)
        p10 = float(point.p10) if point.p10 is not None else None
        p90 = float(point.p90) if point.p90 is not None else None
        checkpoints.append(
            {
                "label": _horizon_label(step, response.interval, lang),
                "time": point.time,
                "median": round(p50, 4),
                "median_change_pct": round(_pct(current, p50), 3),
                "lower_band": round(p10, 4) if p10 is not None else None,
                "upper_band": round(p90, 4) if p90 is not None else None,
            }
        )
    title = (
        f"{response.symbol} {response.interval.upper()} 예측 리포트"
        if lang == "ko"
        else f"{response.symbol} {response.interval.upper()} Forecast Report"
    )
    current_price_label = "기준가" if lang == "ko" and origin_time else "현재가" if lang == "ko" else "reference_price" if origin_time else "current_price"
    key_metrics = {
        ("작성일" if lang == "ko" else "report_date"): _local_date(generated),
        ("예측기간" if lang == "ko" else "forecast_period"): period_text,
        current_price_label: _fmt_price(current),
        (f"{horizon_text}_중앙값" if lang == "ko" else f"{horizon_text}_median"): _fmt_price(median_end),
        ("중앙값_변화율" if lang == "ko" else "median_change"): _fmt_pct(median_change),
        ("예상_변동_범위" if lang == "ko" else "expected_range"): f"{_fmt_price(p10_end)} - {_fmt_price(p90_end)}",
        ("시장_흐름" if lang == "ko" else "market_flow"): regime_label if lang == "ko" else regime,
        ("작성_시각" if lang == "ko" else "generated_at_local"): _local_datetime(generated),
    }
    return {
        "title": title,
        "horizon": horizon,
        "horizon_text": horizon_text,
        "period_text": period_text,
        "current_price": round(current, 4),
        "median_end": round(median_end, 4),
        "p10_end": round(p10_end, 4) if p10_end is not None else None,
        "p90_end": round(p90_end, 4) if p90_end is not None else None,
        "median_change_pct": round(median_change, 3),
        "direction": direction,
        "direction_label": _label_direction(direction, lang),
        "regime": regime,
        "regime_label": regime_label,
        "checkpoints": checkpoints,
        "key_metrics": key_metrics,
    }


def _dashboard_news_evidence(context_payload: dict[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    news = list((context_payload or {}).get("news") or [])
    tail = news[-limit:]
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(tail):
        headline = str(item.get("headline") or "").strip()
        if not headline:
            continue
        rows.append(
            {
                "source_index": index,
                "headline": headline,
                "source": str(item.get("source") or "").strip() or "unknown",
                "date": _iso(item.get("time")) or _iso(item.get("published_at")) or None,
            }
        )
    return rows


def _dashboard_context_evidence(context_payload: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    points = list((context_payload or {}).get("context_points") or [])[-limit:]
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        rows.append(
            {
                "source_index": index,
                "time": point.get("time"),
                "timestamp": point.get("timestamp"),
                "overall_bias": point.get("overall_bias"),
                "event_count": point.get("event_count"),
                "explanation": _public_context_text(point.get("explanation")),
            }
        )
    return rows


def _sanitize_display_context_payload(context_payload: dict[str, Any]) -> dict[str, Any]:
    output = dict(context_payload or {})
    for key in ("context_points", "chart_context_points"):
        rows = []
        for point in output.get(key) or []:
            next_point = dict(point)
            next_point["explanation"] = _public_context_text(next_point.get("explanation"))
            rows.append(next_point)
        if rows or key in output:
            output[key] = rows
    scenario = dict(output.get("scenario_commentary") or {})
    if scenario:
        scenario["summary"] = _public_context_text(scenario.get("summary"))
        output["scenario_commentary"] = scenario
    return output


def _dashboard_reference_time_label(origin_time: str | None) -> str | None:
    if not origin_time:
        return None
    try:
        from backend.app.api.routes.backtests import _parse_origin_time

        return _local_datetime(_parse_origin_time(origin_time))
    except Exception:
        return str(origin_time)


def _dashboard_analysis_cache_key(
    *,
    bundle,
    model_summaries: list[dict[str, Any]],
    context_payload: dict[str, Any],
    origin_time: str | None,
    language: str,
) -> str:
    news_fingerprint = [
        {
            "time": item.get("time"),
            "headline": item.get("headline"),
            "source": item.get("source"),
        }
        for item in ((context_payload.get("news") or [])[-8:])
    ]
    context_fingerprint = [
        {
            "time": item.get("time"),
            "bias": item.get("overall_bias"),
            "events": item.get("event_count"),
        }
        for item in ((context_payload.get("context_points") or [])[-5:])
    ]
    model_fingerprint = [
        {
            "id": item.get("id"),
            "direction": item.get("direction"),
            "steps": item.get("steps"),
            "pct_bucket": round(float(item.get("pct_change") or 0.0), 1),
            "path_adapter": (item.get("path_adapter") or {}).get("adapter"),
        }
        for item in model_summaries
    ]
    candles = getattr(bundle.market_data, "candles", []) or []
    payload = {
        "symbol": bundle.response.symbol,
        "interval": bundle.response.interval,
        "primary_model": bundle.response.primary_model,
        "horizon": bundle.horizon,
        "origin_time": origin_time,
        "last_candle_time": candles[-1].time if candles else None,
        "language": _language(language),
        "prompt_version": _DASHBOARD_ANALYSIS_PROMPT_VERSION,
        "models": model_fingerprint,
        "news": news_fingerprint,
        "context": context_fingerprint,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _translate_news_items(
    context_payload: dict[str, Any],
    translated_news: Any,
) -> dict[str, Any]:
    if not isinstance(translated_news, list):
        return context_payload
    news = [dict(item) for item in (context_payload.get("news") or [])]
    if not news:
        return context_payload
    latest_count = min(8, len(news))
    start = len(news) - latest_count
    by_original: dict[str, str] = {}
    for item in translated_news:
        if not isinstance(item, dict):
            continue
        try:
            source_index = int(item.get("source_index"))
        except (TypeError, ValueError):
            continue
        headline = str(item.get("headline") or "").strip()
        if not headline:
            continue
        absolute_index = start + source_index
        if 0 <= absolute_index < len(news):
            original = str(news[absolute_index].get("headline") or "")
            by_original[original] = headline
            news[absolute_index]["headline"] = headline
    if not by_original:
        return context_payload

    def translate_item(item: dict[str, Any]) -> dict[str, Any]:
        headline = str(item.get("headline") or "")
        if headline in by_original:
            return {**item, "headline": by_original[headline]}
        return dict(item)

    chart_points = []
    for point in context_payload.get("chart_context_points") or []:
        next_point = dict(point)
        next_point["news_items"] = [translate_item(item) for item in (point.get("news_items") or [])]
        chart_points.append(next_point)
    return {
        **context_payload,
        "news": news,
        "chart_context_points": chart_points,
    }


def _dashboard_market_context_from_llm(context_payload: dict[str, Any], raw_news_context: Any) -> dict[str, Any]:
    if not isinstance(raw_news_context, dict):
        raw_news_context = {}
    output = _sanitize_display_context_payload(_translate_news_items(context_payload, raw_news_context.get("translated_news")))
    summary = _public_context_text(raw_news_context.get("summary"))
    if summary:
        scenario = dict(output.get("scenario_commentary") or {})
        scenario.update({"mode": "llm_news_context", "summary": summary})
        output["scenario_commentary"] = scenario

    llm_points = raw_news_context.get("context_points")
    if isinstance(llm_points, list) and llm_points:
        base_points = [dict(point) for point in (output.get("context_points") or [])]
        latest_count = min(5, len(base_points))
        start = len(base_points) - latest_count
        mapped_points = base_points[:]
        if not mapped_points and output.get("news"):
            latest_news = (output.get("news") or [])[-1]
            mapped_points = [
                {
                    "time": latest_news.get("time"),
                    "timestamp": latest_news.get("published_at"),
                    "overall_bias": "mixed",
                    "impact_score": 0.0,
                    "uncertainty": 1.0,
                    "event_count": 1,
                    "news_items": [latest_news],
                }
            ]
            start = 0
        for offset, item in enumerate(llm_points[: max(1, latest_count or len(mapped_points))]):
            if not isinstance(item, dict) or not mapped_points:
                continue
            try:
                source_index = int(item.get("source_index", offset))
            except (TypeError, ValueError):
                source_index = offset
            absolute_index = start + source_index if latest_count else min(source_index, len(mapped_points) - 1)
            absolute_index = min(max(absolute_index, 0), len(mapped_points) - 1)
            explanation = _public_context_text(item.get("explanation"))
            if not explanation:
                continue
            mapped_points[absolute_index] = {
                **mapped_points[absolute_index],
                "overall_bias": str(item.get("overall_bias") or mapped_points[absolute_index].get("overall_bias") or "mixed"),
                "explanation": explanation,
            }
        output["context_points"] = mapped_points
    return _sanitize_display_context_payload(output)


def _dashboard_report_from_llm(
    *,
    bundle,
    raw_report: Any,
    forecast_facts: dict[str, Any],
    generated_at: datetime,
    language: str,
) -> dict[str, Any]:
    lang = _language(language)
    if not isinstance(raw_report, dict):
        raw_report = {}
    executive_summary = str(raw_report.get("executive_summary") or "").strip()
    if not executive_summary:
        raise ValueError("LLM dashboard report response missing executive_summary.")
    sections: list[ReportSection] = []
    for item in raw_report.get("sections") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        body = str(item.get("body") or "").strip()
        if not title or not body:
            continue
        sections.append(ReportSection(title=title, body=body, bullets=_string_list(item.get("bullets"))[:3]))
    if not sections:
        raise ValueError("LLM dashboard report response missing sections.")
    report = ForecastReport(
        generated_at=generated_at,
        symbol=bundle.response.symbol,
        interval=bundle.response.interval,
        horizon=int(forecast_facts.get("horizon") or bundle.horizon),
        mode="llm_dashboard_report",
        llm_used=True,
        source_note=(
            "외부 LLM을 패널별로 호출해 리포트 문장을 작성했습니다. 숫자는 제공된 모델 출력만 사용했습니다."
            if lang == "ko"
            else "Separate external LLM panel calls generated the report prose. Numeric values come only from supplied model outputs."
        ),
        title=str(raw_report.get("title") or forecast_facts.get("title") or ""),
        executive_summary=executive_summary,
        recommendation_note=str(raw_report.get("recommendation_note") or "").strip(),
        key_metrics=dict(forecast_facts.get("key_metrics") or {}),
        sections=sections,
        warnings=_string_list(raw_report.get("warnings")),
        markdown="",
    )
    report = report.model_copy(update={"markdown": _markdown(report, lang)})
    return report.model_dump(mode="json")


def _replace_backtest_relative_terms(value: Any) -> Any:
    replacements = (
        ("현재가", "기준가"),
        ("현재의", "기준 시점의"),
        ("현재 원유", "기준 시점의 원유"),
        ("현재 가격", "기준 시점 가격"),
        ("현재 시장", "기준 시점 시장"),
        ("현재 ", "기준 시점 "),
        ("최근 ", "기준 시점 전후 "),
        ("지금 ", "기준 시점 "),
        ("금일 ", "기준일 "),
    )
    if isinstance(value, str):
        text = value
        for old, new in replacements:
            text = text.replace(old, new)
        return text
    if isinstance(value, list):
        return [_replace_backtest_relative_terms(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_backtest_relative_terms(item) for key, item in value.items()}
    return value


def _normalize_dashboard_analysis(
    *,
    raw: dict[str, Any],
    bundle,
    market_context_payload: dict[str, Any],
    model_summaries: list[dict[str, Any]],
    price_action: dict[str, Any],
    forecast_facts: dict[str, Any],
    generated_at: datetime,
    language: str,
) -> dict[str, Any]:
    commentary_raw = raw.get("commentary") if isinstance(raw.get("commentary"), dict) else {}
    news_raw = raw.get("news_context") if isinstance(raw.get("news_context"), dict) else {}
    report_raw = raw.get("report") if isinstance(raw.get("report"), dict) else {}
    commentary_summary = str(commentary_raw.get("summary") or "").strip()
    commentary_interpretation = str(commentary_raw.get("model_interpretation") or "").strip()
    if not commentary_summary or not commentary_interpretation:
        raise ValueError("LLM dashboard commentary response missing required text.")
    if not str(news_raw.get("summary") or "").strip():
        raise ValueError("LLM dashboard news response missing summary.")
    normalized_market_context = _dashboard_market_context_from_llm(market_context_payload, news_raw)
    commentary = {
        "symbol": bundle.response.symbol,
        **_symbol_payload(bundle),
        "interval": bundle.response.interval,
        "generated_at": generated_at.isoformat(),
        "mode": "llm_dashboard_commentary",
        "summary": commentary_summary,
        "model_interpretation": commentary_interpretation,
        "model_agreement": "",
        "divergence": "",
        "risk_notes": _string_list(commentary_raw.get("risk_notes"))[:4],
        "market_context": normalized_market_context,
        "price_action": price_action,
        "model_summaries": model_summaries,
        "llm_used": True,
        "warnings": _string_list(commentary_raw.get("warnings")),
    }
    report = _dashboard_report_from_llm(
        bundle=bundle,
        raw_report=report_raw,
        forecast_facts=forecast_facts,
        generated_at=generated_at,
        language=language,
    )
    warnings = [
        *_string_list(commentary_raw.get("warnings")),
        *_string_list(news_raw.get("warnings")),
        *_string_list(report_raw.get("warnings")),
    ]
    return {
        "symbol": bundle.response.symbol,
        **_symbol_payload(bundle),
        "interval": bundle.response.interval,
        "generated_at": generated_at.isoformat(),
        "mode": "llm_dashboard_analysis",
        "llm_used": True,
        "commentary": commentary,
        "market_context": normalized_market_context,
        "report": report,
        "warnings": warnings,
    }


def _llm_dashboard_analysis(
    *,
    settings,
    bundle,
    model_summaries: list[dict[str, Any]],
    market_context_payload: dict[str, Any],
    origin_time: str | None,
    language: str,
) -> dict[str, Any]:
    lang = _language(language)
    if not settings.enable_external_llm_calls or not settings.llm_api_key:
        raise HTTPException(
            status_code=503,
            detail={"message": _llm_unavailable_message("not_configured", lang), "warnings": []},
        )
    if not _reserve_llm_calls(_DASHBOARD_ANALYSIS_PANEL_COUNT):
        raise HTTPException(
            status_code=503,
            detail={"message": _llm_unavailable_message("rate_guard", lang), "warnings": []},
        )
    generated_at = datetime.now(timezone.utc)
    forecast_facts = _dashboard_forecast_facts(bundle, lang, generated_at, origin_time=origin_time)
    price_action = _price_action_snapshot(bundle)
    output_language = "English" if lang == "en" else "Korean"
    reference_time_label = _dashboard_reference_time_label(origin_time)
    analysis_mode = "point_in_time_backtest" if origin_time else "live"
    shared_rules = [
        "Use only supplied numeric forecast facts. Do not invent new price targets or paths.",
        "Do not write trading instructions or investment advice.",
        "Do not expose internal model ids, adapters, features, scores, calibration, coverage, quantile, residual, or data-status jargon.",
        "If Korean is requested, translate or paraphrase English news headlines. Do not quote English headlines verbatim.",
        "Never output deterministic/local encoder placeholder text or say that direct headlines are insufficient.",
        "Treat the forecast as model output to explain, not as a recommendation.",
        "Use the same key drivers, reference-time phrasing, and analyst voice as the other dashboard panels.",
        "For Korean output, always use polite user-facing endings such as -습니다, -합니다, -입니다, -됩니다. Do not use terse report endings such as 국면이다, 전망된다, 유지한다, 보인다.",
    ]
    if origin_time:
        shared_rules.extend(
            [
                "Write as of reference_time_label and avoid relative live-market words such as current, currently, recent, now, today, 현재, 최근, 지금, 금일.",
                "The first sentence must explicitly include reference_time_label when the panel has a summary or executive_summary.",
            ]
        )
    prompt_context = {
        "language": output_language,
        "analysis_mode": analysis_mode,
        "reference_time_label": reference_time_label,
        "shared_voice": {
            "role": "oil market analyst",
            "tone": "calm, concise, user-facing, consistent across commentary, news interpretation, and report panels",
            "korean_style": "존댓말 설명체",
        },
        "symbol": bundle.response.symbol,
        "display_symbol": _display_symbol(bundle),
        "interval": bundle.response.interval,
        "origin_time": origin_time,
        "forecast_facts": forecast_facts,
        "price_action": _public_chart_read(price_action, language=lang),
        "regime": bundle.response.regime.model_dump(),
        "model_summaries": model_summaries,
        "forecast_context": {
            "primary_path_adapter": (
                next(
                    (item.get("path_adapter") for item in model_summaries if item.get("id") == bundle.response.primary_model),
                    {},
                )
                or {}
            )
        },
        "news_evidence": _dashboard_news_evidence(market_context_payload, limit=8),
        "context_evidence": _dashboard_context_evidence(market_context_payload, limit=5),
        "rules": shared_rules,
    }

    def panel_prompt(panel: str) -> str:
        panel_contracts = {
            "commentary": (
                "현재 요청은 AI 시황 해설 패널만 작성한다. "
                "반드시 JSON object만 반환하라. 필수 key는 summary, model_interpretation, risk_notes, warnings다. "
                "summary는 대시보드 상단 시황 카드용으로 2~3문장, 기준 가격/방향/핵심 배경을 압축하되 단순 반복으로 끝내지 마라. "
                "model_interpretation은 3~5문장의 한 문단으로, 예측 경로가 왜 그런 방향으로 읽히는지 뉴스와 차트 흐름을 연결해 설명하라. "
                "risk_notes는 3개 문자열 배열로, 전망이 흔들릴 수 있는 확인 변수를 각각 완전한 문장으로 적어라."
            ),
            "news_context": (
                "현재 요청은 뉴스 해석 패널만 작성한다. "
                "반드시 JSON object만 반환하라. 필수 key는 summary, translated_news, context_points, warnings다. "
                "summary는 2~4문장으로 최신 뉴스 묶음의 공통 주제와 반대 논리를 함께 설명하라. "
                "translated_news는 news_evidence의 source_index와 번역/요약 headline을 담고, 영어 원제목을 그대로 쓰지 마라. "
                "context_points는 context_evidence의 source_index를 참조하고 overall_bias, explanation을 담는다. "
                "각 explanation은 해당 시점의 뉴스/가격 맥락을 1~2문장으로 다르게 작성하고 summary 문장을 반복하지 마라."
            ),
            "report": (
                "현재 요청은 예측 리포트 패널만 작성한다. "
                "반드시 JSON object만 반환하라. 필수 key는 title, executive_summary, sections, recommendation_note, warnings다. "
                "executive_summary는 3~5문장으로 기간, 중앙 경로, 예상 범위, 핵심 조건을 모두 담아라. "
                "sections는 정확히 4개로 작성하고 각 section은 title, body, bullets 배열을 가진다. "
                "권장 section 관점은 핵심 전망, 시장 배경, 경로와 변동성, 확인 변수다. "
                "각 section.body는 2~4문장, bullets는 2개이며, bullets도 완전한 문장으로 작성하라."
            ),
        }
        return (
            "너는 원유 시장 전담 애널리스트이며, 원유 예측 대시보드의 한 패널 문안을 작성한다. "
            "시황 해설, 뉴스 해석, 예측 리포트는 서로 다른 요청으로 생성되므로 shared_voice와 rules를 최우선으로 적용해 말투, 용어, 기준 시점 표현을 일관되게 유지하라. "
            f"모든 사용자 표시 문장은 반드시 {output_language}로 작성하라. "
            "뉴스, 공급/수요, 재고, OPEC, 달러/금리, 위험선호, 차트 흐름 같은 애널리스트 언어를 사용하라. "
            "내부 구현, 로컬 인코더, deterministic, structured context, 직접 표시할 뉴스 부족 같은 표현은 절대 쓰지 마라. "
            f"{panel_contracts[panel]}\n\n"
            + json.dumps({**prompt_context, "target_panel": panel}, ensure_ascii=False)
        )

    def call_dashboard_llm(prompt: str) -> dict[str, Any]:
        return (
            _google_model_commentary(settings, prompt)
            if settings.llm_context_mode == "google_generative"
            or "generativelanguage.googleapis.com" in settings.llm_api_base
            or settings.llm_model.lower().startswith("gemma-")
            else _openai_compatible_model_commentary(settings, prompt)
        )

    def panel_result(panel: str) -> dict[str, Any]:
        raw_panel = call_dashboard_llm(panel_prompt(panel))
        nested = raw_panel.get(panel)
        return nested if isinstance(nested, dict) else raw_panel

    try:
        raw = {
            "commentary": panel_result("commentary"),
            "news_context": panel_result("news_context"),
            "report": panel_result("report"),
        }
        normalized = _normalize_dashboard_analysis(
            raw=raw,
            bundle=bundle,
            market_context_payload=market_context_payload,
            model_summaries=model_summaries,
            price_action=price_action,
            forecast_facts=forecast_facts,
            generated_at=generated_at,
            language=lang,
        )
        if origin_time and lang == "ko":
            normalized = _replace_backtest_relative_terms(normalized)
        return normalized
    except (KeyError, RuntimeError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": _llm_provider_error_message(exc, language=lang), "warnings": []},
        ) from exc


@router.get("/api/market-context")
def market_context(
    symbol: str = Query(default=None),
    interval: str = Query(default=None),
    horizon: int | None = Query(default=None, ge=1),
    models: str | None = Query(default=None),
    limit: int = Query(default=60, ge=1, le=200),
    live: bool = Query(default=False),
    origin_time: str | None = Query(default=None),
):
    current_settings = get_settings()
    requested_symbol = symbol or current_settings.default_symbol
    requested_interval = interval or current_settings.default_interval
    try:
        bundle = (
            _forecast_bundle_for_commentary(
                symbol=requested_symbol,
                interval=requested_interval,
                horizon=horizon,
                models=models,
                origin_time=origin_time,
                settings=current_settings,
            )
            if origin_time
            else build_forecast(
                symbol=requested_symbol,
                interval=requested_interval,
                horizon=horizon,
                models=models,
                settings=current_settings,
            )
        )
    except (MarketDataUnavailable, PretrainedModelNotFoundError, ForecastUnavailable) as exc:
        raise service_error(exc) from exc
    except InvalidModelRequest as exc:
        raise HTTPException(status_code=400, detail=exc.as_detail()) from exc

    return _market_context_payload_from_bundle(
        bundle=bundle,
        settings=current_settings,
        limit=limit,
        live=live,
        origin_time=origin_time,
    )


@router.get("/api/dashboard-analysis")
def dashboard_analysis(
    symbol: str = Query(default=None),
    interval: str = Query(default=None),
    horizon: int | None = Query(default=None, ge=1),
    models: str | None = Query(default=None),
    origin_time: str | None = Query(default=None),
    language: str = Query(default="ko"),
) -> dict[str, Any]:
    current_settings = get_settings()
    requested_symbol = symbol or current_settings.default_symbol
    requested_interval = interval or current_settings.default_interval
    lang = _language(language)
    try:
        bundle = (
            _forecast_bundle_for_commentary(
                symbol=requested_symbol,
                interval=requested_interval,
                horizon=horizon,
                models=models,
                origin_time=origin_time,
                settings=current_settings,
            )
            if origin_time
            else build_forecast(
                symbol=requested_symbol,
                interval=requested_interval,
                horizon=horizon,
                models=models,
                include_scenarios=True,
                settings=current_settings,
            )
        )
    except (MarketDataUnavailable, PretrainedModelNotFoundError, ForecastUnavailable) as exc:
        raise service_error(exc) from exc
    except InvalidModelRequest as exc:
        raise HTTPException(status_code=400, detail=exc.as_detail()) from exc

    model_summaries = _model_path_summaries(bundle)
    # The combined panel LLM call interprets news once, so live context collection must not
    # spend a second external LLM call on event encoding for this endpoint.
    context_settings = current_settings.model_copy(update={"enable_external_llm_calls": False})
    market_context_payload = _market_context_payload_from_bundle(
        bundle=bundle,
        settings=context_settings,
        limit=60,
        live=not bool(origin_time),
        origin_time=origin_time,
    )
    cache_key = _dashboard_analysis_cache_key(
        bundle=bundle,
        model_summaries=model_summaries,
        context_payload=market_context_payload,
        origin_time=origin_time,
        language=lang,
    )
    now = time.time()
    cached = _DASHBOARD_ANALYSIS_CACHE.get(cache_key)
    if cached and now - cached[0] <= _MODEL_COMMENTARY_CACHE_TTL_SECONDS:
        return {**cached[1], "cached": True}

    analysis = _llm_dashboard_analysis(
        settings=current_settings,
        bundle=bundle,
        model_summaries=model_summaries,
        market_context_payload=market_context_payload,
        origin_time=origin_time,
        language=lang,
    )
    analysis["cached"] = False
    _DASHBOARD_ANALYSIS_CACHE[cache_key] = (now, analysis)
    return analysis


@router.get("/api/model-commentary")
def model_commentary(
    symbol: str = Query(default=None),
    interval: str = Query(default=None),
    horizon: int | None = Query(default=None, ge=1),
    models: str | None = Query(default=None),
    origin_time: str | None = Query(default=None),
    language: str = Query(default="ko"),
) -> dict[str, Any]:
    current_settings = get_settings()
    requested_symbol = symbol or current_settings.default_symbol
    requested_interval = interval or current_settings.default_interval
    try:
        bundle = _forecast_bundle_for_commentary(
            symbol=requested_symbol,
            interval=requested_interval,
            horizon=horizon,
            models=models,
            origin_time=origin_time,
            settings=current_settings,
        )
    except (MarketDataUnavailable, PretrainedModelNotFoundError, ForecastUnavailable) as exc:
        raise service_error(exc) from exc
    except InvalidModelRequest as exc:
        raise HTTPException(status_code=400, detail=exc.as_detail()) from exc

    model_summaries = _model_path_summaries(bundle)
    lang = _language(language)
    cache_key = _commentary_cache_key(bundle, model_summaries, origin_time, lang)
    cached = _MODEL_COMMENTARY_CACHE.get(cache_key)
    now = time.time()
    if cached and now - cached[0] <= _MODEL_COMMENTARY_CACHE_TTL_SECONDS:
        return {**cached[1], "cached": True}

    market_context_payload = _commentary_market_context(bundle, current_settings, origin_time=origin_time)
    price_action = _price_action_snapshot(bundle)
    commentary = _llm_model_commentary(
        current_settings,
        bundle,
        model_summaries,
        origin_time,
        language=lang,
        market_context=market_context_payload,
        price_action=price_action,
    )
    commentary["cached"] = False
    _MODEL_COMMENTARY_CACHE[cache_key] = (now, commentary)
    return commentary


@router.post("/api/assistant-chat")
def assistant_chat(request: AssistantChatRequest) -> dict[str, Any]:
    current_settings = get_settings()
    requested_symbol = request.symbol or current_settings.default_symbol
    requested_interval = request.interval or current_settings.default_interval
    lang = _language(request.language)
    try:
        bundle = build_forecast(
            symbol=requested_symbol,
            interval=requested_interval,
            horizon=request.horizon,
            models=request.models,
            settings=current_settings,
        )
    except (MarketDataUnavailable, PretrainedModelNotFoundError, ForecastUnavailable) as exc:
        raise service_error(exc) from exc
    except InvalidModelRequest as exc:
        raise HTTPException(status_code=400, detail=exc.as_detail()) from exc

    model_summaries = _model_path_summaries(bundle)
    context_payload: dict[str, Any] | None = None
    context_warnings: list[str] = []
    try:
        context_payload = _load_live_context_payload(symbol=bundle.response.symbol, settings=current_settings, limit=24)
        context_warnings = [str(item) for item in context_payload.get("warnings") or []]
    except Exception as exc:
        context_warnings = [f"Live context unavailable for chat: {exc}"]

    if not current_settings.enable_external_llm_calls or not current_settings.llm_api_key:
        raise HTTPException(
            status_code=503,
            detail={
                "message": _llm_unavailable_message("not_configured", lang),
                "warnings": context_warnings,
            },
        )
    if not _reserve_llm_call():
        raise HTTPException(
            status_code=503,
            detail={
                "message": _llm_unavailable_message("rate_guard", lang),
                "warnings": context_warnings,
            },
        )

    primary_summary = next(
        (item for item in model_summaries if item.get("id") == bundle.response.primary_model),
        model_summaries[0] if model_summaries else {},
    )
    latest_context_points = (context_payload or {}).get("context_points") or []
    latest_context = latest_context_points[-1] if latest_context_points else {}
    price_action = _price_action_snapshot(bundle)
    public_context = {
        "question": request.question,
        "display_symbol": _display_symbol(bundle),
        "interval": bundle.response.interval,
        "current_price": round(float(bundle.response.current_price), 4),
        "horizon": bundle.horizon,
        "existing_forecast": {
            "direction": primary_summary.get("direction") or "mixed",
            "move_pct": primary_summary.get("pct_change"),
            "start_price": primary_summary.get("start"),
            "end_price": primary_summary.get("end"),
        },
        "chart_read": _public_chart_read(price_action, language=lang),
        "news_read": {
            "tone": latest_context.get("overall_bias") or "mixed",
            "available_items": len((context_payload or {}).get("news") or []),
            "evidence": _public_news_evidence(context_payload, limit=6),
        },
        "policy": "Answer the user's actual question about the existing dashboard view in natural language only.",
    }
    output_language = "English" if lang == "en" else "Korean"
    prompt = (
        "너는 유가 예측 대시보드 안에서 사용자와 대화하는 원유 시장 애널리스트 LLM이다. "
        "반드시 사용자의 질문에 직접 답하라. 질문이 상승 근거, 하락 리스크, 뉴스 영향, 차트 해석, 예측 기간, 불확실성 중 무엇을 묻는지 먼저 파악하고 그 관점으로 답하라. "
        "답변은 현재 화면의 이미 계산된 예측 경로, 차트 흐름, 뉴스/이벤트 근거를 사람이 이해할 수 있는 말로 연결해 설명한다. "
        "뉴스를 참고할 때는 제목을 그대로 나열하지 말고, 공급 차질/재고/OPEC/지정학/달러와 금리/위험선호 같은 시장 재료로 요약해 차트 흐름과 연결하라. "
        "질문이 모호하거나 장난성 문구라면 현재 화면 기준으로 답할 수 있는 범위를 짧게 말하고, 사용자가 다시 물어볼 수 있는 구체 질문 예시를 하나 제안하라. "
        "새 가격 목표, 새 수익률 경로, 매매 지시, 투자 권유는 만들지 마라. 이미 계산된 forecast 값은 '현재 화면의 예측'으로만 설명하라. "
        "내부 필드명, 모델 이름, 점수, bias/impact, score, event score, pattern residual, template, adapter, feature, 단일 운영 모델, 기술적 변수 같은 표현은 절대 쓰지 마라. "
        "답변 첫머리에 LLM, 시스템, fallback 같은 메타 표현을 붙이지 마라. "
        f"응답 언어는 반드시 {output_language}로 맞춰라. JSON object만 반환하라. key는 answer, warnings만 사용하라.\n\n"
        + json.dumps(public_context, ensure_ascii=False)
    )
    try:
        raw = None
        last_error: BaseException | None = None
        use_google = (
            current_settings.llm_context_mode == "google_generative"
            or "generativelanguage.googleapis.com" in current_settings.llm_api_base
            or current_settings.llm_model.lower().startswith("gemma-")
        )
        for attempt in range(2):
            try:
                raw = (
                    _google_model_commentary(current_settings, prompt)
                    if use_google
                    else _openai_compatible_model_commentary(current_settings, prompt)
                )
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                retryable = isinstance(exc, urllib.error.HTTPError) and exc.code in {429, 500, 502, 503, 504}
                if attempt == 0 and retryable:
                    time.sleep(0.6)
                    continue
                raise
        if raw is None:
            raise RuntimeError(_llm_provider_error_message(last_error or RuntimeError("empty response"), language=lang))
        answer = str(raw.get("answer") or "")
        if not _external_chat_answer_is_public(answer):
            raise HTTPException(
                status_code=502,
                detail={
                    "message": (
                        "외부 LLM 응답에 내부 모델 표현이 포함되어 표시하지 않았습니다. 다시 질문해 주세요."
                        if lang == "ko"
                        else "External LLM response exposed internal model terms and was not shown."
                    ),
                    "warnings": context_warnings,
                },
            )
        return {
            "mode": "llm_assistant",
            **_symbol_payload(bundle),
            "answer": answer,
            "warnings": [*context_warnings, *_string_list(raw.get("warnings"))],
            "llm_used": True,
            "question": request.question,
        }
    except HTTPException:
        raise
    except (KeyError, RuntimeError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": _llm_provider_error_message(exc, language=lang),
                "warnings": context_warnings,
            },
        ) from exc
