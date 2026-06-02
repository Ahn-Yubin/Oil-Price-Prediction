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
from market_ai.config import get_settings
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable, load_market_data_window
from market_ai.data.storage import DATA_ROOT, read_table
from market_ai.forecasting.service import ForecastUnavailable, build_forecast
from market_ai.llm.event_encoder import _default_https_context, _extract_json_text
from market_ai.llm.live_context import build_live_event_context
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError
from market_ai.modeling.model_catalog import InvalidModelRequest


router = APIRouter()
_MODEL_COMMENTARY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_MARKET_CONTEXT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_LLM_CALL_TIMESTAMPS: list[float] = []
_MODEL_COMMENTARY_CACHE_TTL_SECONDS = 900
_MARKET_CONTEXT_CACHE_TTL_SECONDS = 300
_LLM_CALL_WINDOW_SECONDS = 60
_LLM_CALL_LIMIT_PER_WINDOW = 12


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


def _filter_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns:
        return frame
    symbol_upper = symbol.upper()
    return frame[frame["symbol"].astype(str).str.upper().isin([symbol_upper, "ALL", "*"])].copy()


def _news_items(*, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, limit: int) -> list[dict[str, Any]]:
    frame = _read_optional(DATA_ROOT / "raw" / "news" / "public_market_news.csv")
    if frame.empty:
        return []
    frame = _filter_symbol(frame, symbol)
    frame["published_at"] = pd.to_datetime(frame.get("published_at"), errors="coerce", utc=True)
    frame = frame.dropna(subset=["published_at"])
    frame = frame[(frame["published_at"] >= start_ts) & (frame["published_at"] <= end_ts)].sort_values("published_at")
    rows = []
    for row in frame.tail(limit).to_dict(orient="records"):
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


def _context_points(*, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, limit: int) -> list[dict[str, Any]]:
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
    rows = []
    for row in frame.tail(limit).to_dict(orient="records"):
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
        rows.append(
            {
                "id": model.get("id"),
                "label": model.get("label") or model.get("id"),
                "direction": direction,
                "pct_change": round(pct_change, 3),
                "start": round(start, 4),
                "end": round(end, 4),
                "steps": max(0, len(points) - 1),
            }
        )
    return rows


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


def _commentary_market_context(bundle, settings, *, limit: int = 12) -> dict[str, Any]:
    candles = bundle.market_data.candles
    resolved_symbol = bundle.response.symbol
    if candles:
        end_ts = max(pd.to_datetime(candles[-1].time, unit="s", utc=True), pd.Timestamp(datetime.now(timezone.utc)))
        start_ts = end_ts - pd.Timedelta(days=90)
    else:
        end_ts = pd.Timestamp(datetime.now(timezone.utc))
        start_ts = end_ts - pd.Timedelta(days=90)
    cached_news = _news_items(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=limit)
    cached_context_points = _context_points(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=limit)
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


def _reserve_llm_call() -> bool:
    now = time.time()
    cutoff = now - _LLM_CALL_WINDOW_SECONDS
    _LLM_CALL_TIMESTAMPS[:] = [stamp for stamp in _LLM_CALL_TIMESTAMPS if stamp >= cutoff]
    if len(_LLM_CALL_TIMESTAMPS) >= _LLM_CALL_LIMIT_PER_WINDOW:
        return False
    _LLM_CALL_TIMESTAMPS.append(now)
    return True


def _deterministic_model_commentary(
    bundle,
    model_summaries: list[dict[str, Any]],
    warnings: list[str] | None = None,
    language: str = "ko",
    market_context: dict[str, Any] | None = None,
    price_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lang = _language(language)
    response = bundle.response
    display_symbol = _display_symbol(bundle)
    provider_note = f" ({response.symbol} provider 기준)" if lang == "ko" and display_symbol != response.symbol else ""
    provider_note_en = f" (provider symbol {response.symbol})" if lang == "en" and display_symbol != response.symbol else ""
    market_context = market_context or {}
    price_action = price_action or _price_action_snapshot(bundle)
    latest_news = [str(item.get("headline") or "").strip() for item in (market_context.get("news") or []) if item.get("headline")]
    latest_news = [item for item in latest_news if item][:3]
    context_points = market_context.get("context_points") or []
    latest_context = context_points[-1] if context_points else {}
    bias = str(latest_context.get("overall_bias") or "neutral").lower()
    impact = float(latest_context.get("impact_score") or 0.0)
    regime_values = response.regime.model_dump()
    regime_label = max((key for key in regime_values if key != "confidence"), key=lambda key: regime_values[key])
    if not model_summaries:
        summary = "No displayable model forecast path is available." if lang == "en" else "표시 가능한 모델 예측 경로가 없습니다."
        model_interpretation = (
            "There is not enough market data to form a readable analyst view for this refresh."
            if lang == "en"
            else "이번 갱신에서는 시황 해설을 구성할 만큼의 시장 데이터가 부족합니다."
        )
    else:
        primary_id = response.primary_model or model_summaries[0].get("id") or "oil_context_fusion"
        primary_summary = next((item for item in model_summaries if item.get("id") == primary_id), model_summaries[0])
        start_price = float(response.current_price)
        end_price = float(primary_summary.get("end") or start_price)
        move_pct = (end_price / max(start_price, 1e-8) - 1.0) * 100.0
        direction = str(primary_summary.get("direction") or "flat")
        if lang == "en":
            lead_direction = "upside" if direction == "up" else "downside" if direction == "down" else "sideways"
            news_text = f" Recent headlines include: {' / '.join(latest_news)}." if latest_news else ""
            summary = (
                f"{display_symbol}{provider_note_en} is leaning {lead_direction} over the next {bundle.horizon} steps, "
                f"with the path moving from {start_price:.2f} to {end_price:.2f} ({move_pct:+.2f}%)."
                f"{news_text}"
            )
            model_interpretation = (
                f"The read is consistent with {price_action.get('pattern_read')}, a {regime_label.replace('_', ' ')} regime, "
                f"and a news/event tone of {bias} with impact {impact:.2f}. The key question is whether the latest supply, "
                "demand, and macro headlines reinforce that tone or fade after the initial reaction."
            )
        else:
            lead_direction = "상방" if direction == "up" else "하방" if direction == "down" else "횡보"
            news_text = f" 최근 관련 뉴스는 {' / '.join(latest_news)}입니다." if latest_news else ""
            pattern_ko = {
                "recent pullback from the upper part of the range": "최근 고점권에서 밀리는 흐름",
                "rebound attempt from the lower part of the range": "최근 저점권에서 반등을 시도하는 흐름",
                "uptrend with short-term consolidation": "상승 추세 속 단기 숨고르기",
                "downtrend with fragile rebounds": "하락 추세 속 취약한 반등",
                "range-bound trade without a decisive breakout": "뚜렷한 돌파가 없는 박스권 흐름",
                "insufficient chart history": "차트 이력이 부족한 상태",
            }.get(str(price_action.get("pattern_read")), str(price_action.get("pattern_read") or "혼조 흐름"))
            regime_ko = str(regime_label).replace("_", " ")
            summary = (
                f"{display_symbol}{provider_note}는 앞으로 {bundle.horizon}개 봉에서 {start_price:.2f}에서 "
                f"{end_price:.2f}로 {move_pct:+.2f}% 움직이는 {lead_direction} 시나리오가 우세합니다."
                f"{news_text}"
            )
            model_interpretation = (
                f"이 판단은 {pattern_ko}, 현재 {regime_ko} 국면, 그리고 뉴스/이벤트 분위기({bias}, 중요도 {impact:.2f})가 "
                "같이 반영된 결과로 볼 수 있습니다. 핵심은 공급 차질, 재고, OPEC 관련 뉴스와 달러/금리 흐름이 "
                "현재 방향을 계속 밀어주는지 여부입니다."
            )
    risk_notes = (
        [
            "A sudden change in OPEC, Iran/Russia, inventory, or refinery headlines can quickly reverse the setup.",
            "A stronger dollar, higher rates, or a broad risk-off equity move can pressure crude even when supply headlines look supportive.",
        ]
        if lang == "en"
        else [
            "OPEC, 이란/러시아, 재고, 정제시설 관련 뉴스가 갑자기 바뀌면 현재 방향이 빠르게 뒤집힐 수 있습니다.",
            "달러 강세, 금리 상승, 증시 위험회피가 강해지면 공급 뉴스가 우호적이어도 유가에는 부담이 될 수 있습니다.",
        ]
    )
    return {
        "symbol": response.symbol,
        **_symbol_payload(bundle),
        "interval": response.interval,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "deterministic_model_commentary",
        "summary": summary,
        "model_interpretation": model_interpretation,
        "model_agreement": "",
        "divergence": "",
        "risk_notes": risk_notes,
        "market_context": market_context,
        "price_action": price_action,
        "model_summaries": model_summaries,
        "llm_used": False,
        "warnings": warnings or [],
    }


def _commentary_cache_key(bundle, model_summaries: list[dict[str, Any]], origin_time: str | None, language: str) -> str:
    model_fingerprint = [
        {
            "id": item.get("id"),
            "direction": item.get("direction"),
            "steps": item.get("steps"),
            "pct_bucket": round(float(item.get("pct_change") or 0.0), 1),
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
        "models": model_fingerprint,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _extract_commentary_json(raw: str) -> dict[str, Any]:
    parsed = json.loads(_extract_json_text(raw))
    if not isinstance(parsed, dict):
        raise ValueError("LLM commentary response must be a JSON object.")
    return parsed


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
    with urllib.request.urlopen(request, timeout=20.0, context=_default_https_context()) as response:
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
    with urllib.request.urlopen(request, timeout=20.0, context=_default_https_context()) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _extract_commentary_json(body["choices"][0]["message"]["content"])


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
        return _deterministic_model_commentary(
            bundle,
            model_summaries,
            ["External LLM commentary disabled or API key missing; deterministic commentary used."],
            language=lang,
            market_context=market_context,
            price_action=price_action,
        )
    if not _reserve_llm_call():
        return _deterministic_model_commentary(
            bundle,
            model_summaries,
            ["External LLM rate limit guard active; deterministic commentary used for this refresh."],
            language=lang,
            market_context=market_context,
            price_action=price_action,
        )
    output_language = "English" if lang == "en" else "Korean"
    prompt = (
        "너는 원유 시장 애널리스트다. "
        "아래 JSON에는 이미 계산된 단일 운영 모델(oil_context_fusion)의 예측 경로, 최근 차트 흐름, 뉴스/이벤트 맥락이 있다. "
        "새로운 가격 목표, 새 수익률 경로, 매매 지시를 만들지 말고 제공된 정보만 바탕으로 왜 이런 방향성이 나왔는지 설명하라. "
        "사용자가 읽는 화면에 들어갈 글이므로 내부 기술 설명을 하지 마라. "
        "금지어: 텐서, 피처 벡터, 보정 상태, calibration, coverage, quantile, 분위수, 잔차, data status, 신뢰구간. "
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
            "risk_notes": [str(item) for item in raw.get("risk_notes") or []],
            "market_context": market_context,
            "price_action": price_action,
            "model_summaries": model_summaries,
            "llm_used": True,
            "warnings": [str(item) for item in raw.get("warnings") or []],
        }
    except (KeyError, RuntimeError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return _deterministic_model_commentary(
            bundle,
            model_summaries,
            [f"LLM commentary fallback: {exc}"],
            language=lang,
            market_context=market_context,
            price_action=price_action,
        )


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


def _deterministic_chat_answer(
    *,
    question: str,
    bundle,
    model_summaries: list[dict[str, Any]],
    context_payload: dict[str, Any] | None,
    language: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    lang = _language(language)
    response = bundle.response
    primary = response.primary_model or "unknown"
    primary_summary = next((item for item in model_summaries if item.get("id") == primary), model_summaries[0] if model_summaries else {})
    direction = str(primary_summary.get("direction") or "unknown")
    end_price = primary_summary.get("end")
    latest_context = (context_payload or {}).get("context_points") or []
    point = latest_context[-1] if latest_context else {}
    bias = point.get("overall_bias") or "unknown"
    impact = float(point.get("impact_score") or 0.0)
    if lang == "en":
        end_text = f" ending near {float(end_price):.2f}" if end_price is not None else ""
        answer = (
            f"For {_display_symbol(bundle)}, the single operating model {primary} currently points {direction}{end_text}. "
            f"The live/news context encoder reports bias={bias} "
            f"with impact={impact:.2f}. This answer explains the program output only; it is not a new price forecast or trading instruction."
        )
    else:
        end_text = f", 말단 가격은 약 {float(end_price):.2f}" if end_price is not None else ""
        answer = (
            f"{_display_symbol(bundle)}의 단일 운영 모델 {primary}은 현재 방향을 {direction}으로 제시합니다{end_text}. "
            f"실시간 뉴스/이벤트 인코더는 현재 bias={bias}, 중요도={impact:.2f}로 요약했습니다. "
            "이 답변은 프로그램 출력과 시황 컨텍스트를 설명하는 것이며 새 가격 예측이나 매매 지시가 아닙니다."
        )
    return {
        "mode": "deterministic_assistant",
        **_symbol_payload(bundle),
        "answer": answer,
        "warnings": warnings or [],
        "llm_used": False,
        "question": question,
    }


@router.get("/api/market-context")
def market_context(
    symbol: str = Query(default=None),
    interval: str = Query(default=None),
    horizon: int | None = Query(default=None, ge=1),
    models: str | None = Query(default=None),
    limit: int = Query(default=60, ge=1, le=200),
    live: bool = Query(default=False),
):
    current_settings = get_settings()
    try:
        bundle = build_forecast(
            symbol=symbol or current_settings.default_symbol,
            interval=interval or current_settings.default_interval,
            horizon=horizon,
            models=models,
            settings=current_settings,
        )
    except (MarketDataUnavailable, PretrainedModelNotFoundError, ForecastUnavailable) as exc:
        raise service_error(exc) from exc
    except InvalidModelRequest as exc:
        raise HTTPException(status_code=400, detail=exc.as_detail()) from exc

    candles = bundle.market_data.candles
    if candles:
        start_ts = pd.to_datetime(candles[0].time, unit="s", utc=True)
        end_ts = max(pd.to_datetime(candles[-1].time, unit="s", utc=True), pd.Timestamp(datetime.now(timezone.utc)))
    else:
        end_ts = pd.Timestamp(datetime.now(timezone.utc))
        start_ts = end_ts - pd.Timedelta(days=120)
    resolved_symbol = bundle.response.symbol
    cached_news = _news_items(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=limit)
    cached_context_points = _context_points(symbol=resolved_symbol, start_ts=start_ts, end_ts=end_ts, limit=limit)
    live_news: list[dict[str, Any]] = []
    live_context_points: list[dict[str, Any]] = []
    live_warnings: list[str] = []
    live_source = "offline_cache"
    if live:
        live_source = "live_public_news"
        try:
            live_payload = _load_live_context_payload(symbol=resolved_symbol, settings=current_settings, limit=limit)
            live_news = live_payload["news"]
            live_context_points = live_payload["context_points"]
            live_warnings = [str(item) for item in live_payload.get("warnings") or []]
            live_source = str(live_payload.get("source") or "live_public_news")
            if live_payload.get("cached"):
                live_source = f"{live_source}_cached"
        except Exception as exc:
            live_source = "live_public_news_unavailable"
            live_warnings = [f"Live news context unavailable: {exc}"]
    response_news = live_news if live else cached_news
    response_context_points = live_context_points if live else cached_context_points
    return {
        "symbol": resolved_symbol,
        **_symbol_payload(bundle),
        "interval": bundle.response.interval,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_context_summary": bundle.response.llm_context_summary,
        "news": response_news,
        "context_points": response_context_points,
        "news_source": live_source,
        "news_warnings": live_warnings,
        "offline_cache_available": {
            "news_count": len(cached_news),
            "context_point_count": len(cached_context_points),
        },
        "scenario_commentary": _scenario_commentary(bundle),
        "primary_model": bundle.response.primary_model,
        "calibration_status": bundle.response.calibration_status,
    }


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

    market_context_payload = _commentary_market_context(bundle, current_settings)
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

    fallback = _deterministic_chat_answer(
        question=request.question,
        bundle=bundle,
        model_summaries=model_summaries,
        context_payload=context_payload,
        language=lang,
        warnings=context_warnings,
    )
    if not current_settings.enable_external_llm_calls or not current_settings.llm_api_key:
        return fallback
    if not _reserve_llm_call():
        return {
            **fallback,
            "warnings": [*fallback["warnings"], "External LLM rate limit guard active; deterministic answer used."],
        }

    output_language = "English" if lang == "en" else "Korean"
    prompt = (
        "너는 시장 예측 대시보드의 내장 분석 채팅이다. "
        "사용자 질문에 답하되, 새 가격 목표/새 수익률 경로/매매 지시는 만들지 마라. "
        "반드시 이미 계산된 단일 모델 예측 경로와 뉴스 이벤트 인코더 팩터만 설명하라. "
        "복수 모델 비교 표현은 쓰지 마라. "
        f"응답 언어는 반드시 {output_language}로 맞춰라. JSON object만 반환하라. key는 answer, warnings만 사용하라.\n\n"
        + json.dumps(
            {
                "question": request.question,
                "symbol": bundle.response.symbol,
                "display_symbol": _display_symbol(bundle),
                "provider_symbol": bundle.response.symbol,
                "interval": bundle.response.interval,
                "current_price": float(bundle.response.current_price),
                "primary_model": bundle.response.primary_model,
                "model_summaries": model_summaries,
                "latest_context_points": (context_payload or {}).get("context_points", [])[-3:],
                "latest_news": (context_payload or {}).get("news", [])[-8:],
                "forecast_policy": "LLM is context/explanation only; numeric prices come from the model forecast path.",
            },
            ensure_ascii=False,
        )
    )
    try:
        raw = (
            _google_model_commentary(current_settings, prompt)
            if current_settings.llm_context_mode == "google_generative"
            or "generativelanguage.googleapis.com" in current_settings.llm_api_base
            or current_settings.llm_model.lower().startswith("gemma-")
            else _openai_compatible_model_commentary(current_settings, prompt)
        )
        return {
            "mode": "llm_assistant",
            **_symbol_payload(bundle),
            "answer": str(raw.get("answer") or fallback["answer"]),
            "warnings": [*context_warnings, *[str(item) for item in raw.get("warnings") or []]],
            "llm_used": True,
            "question": request.question,
        }
    except (KeyError, RuntimeError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {
            **fallback,
            "warnings": [*fallback["warnings"], f"External LLM chat fallback: {exc}"],
        }
