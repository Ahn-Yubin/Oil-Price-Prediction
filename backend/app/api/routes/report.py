from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.api.dependencies import service_error
from market_ai.config import get_settings
from market_ai.data.providers.yfinance_provider import MarketDataUnavailable
from market_ai.forecasting.service import ForecastUnavailable, build_forecast
from market_ai.modeling.forecasters.neural_npz import PretrainedModelNotFoundError
from market_ai.modeling.model_catalog import InvalidModelRequest


router = APIRouter()


class ReportSection(BaseModel):
    title: str
    body: str
    bullets: list[str] = Field(default_factory=list)


class ForecastReport(BaseModel):
    generated_at: datetime
    symbol: str
    interval: str
    horizon: int
    mode: str = "deterministic_report"
    llm_used: bool = False
    source_note: str | None = None
    title: str
    executive_summary: str
    recommendation_note: str
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    sections: list[ReportSection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    markdown: str


def _pct(start: float, end: float) -> float:
    if start == 0:
        return 0.0
    return ((end / start) - 1.0) * 100.0


def _fmt_price(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.2f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.2f}%"


def _local_date(value: datetime | int | None) -> str:
    if value is None:
        return "-"
    timestamp = pd.Timestamp(value, unit="s", tz="UTC") if isinstance(value, int) else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("Asia/Seoul").strftime("%Y-%m-%d")


def _local_datetime(value: datetime | int | None) -> str:
    if value is None:
        return "-"
    timestamp = pd.Timestamp(value, unit="s", tz="UTC") if isinstance(value, int) else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S KST")


def _horizon_label(horizon: int, interval: str, language: str) -> str:
    interval_unit = str(interval).lower()
    if language == "ko":
        unit = "일" if interval_unit == "1d" else "시간" if interval_unit == "1h" else f"{interval} 구간"
        return f"{horizon}{unit} 뒤"
    unit = "day" if interval_unit == "1d" else "hour" if interval_unit == "1h" else f"{interval} step"
    plural = "" if horizon == 1 else "s"
    return f"in {horizon} {unit}{plural}"


def _period_label(first_time: int | None, last_time: int | None, language: str) -> str:
    start = _local_date(first_time)
    end = _local_date(last_time)
    return f"{start} ~ {end}" if language == "ko" else f"{start} to {end}"


def _dominant_regime(regime: Any) -> str:
    values = regime.model_dump() if hasattr(regime, "model_dump") else dict(regime or {})
    candidates = {key: val for key, val in values.items() if key != "confidence"}
    if not candidates:
        return "unknown"
    return max(candidates, key=lambda key: float(candidates.get(key) or 0.0))


def _label_direction(direction: str, language: str) -> str:
    if language != "ko":
        return direction
    return {
        "upside": "상방",
        "downside": "하방",
        "range-bound": "횡보",
        "trend_up": "상승 추세",
        "trend_down": "하락 추세",
        "range": "횡보",
        "high_volatility": "고변동성",
        "event_driven": "이벤트 주도",
        "unknown": "알 수 없음",
    }.get(direction, direction)


def _model_path_summary(paths: list[dict[str, Any]], current_price: float, language: str) -> list[str]:
    bullets: list[str] = []
    for path in paths[:6]:
        points = path.get("points") or []
        if len(points) < 2:
            continue
        label = path.get("label") or path.get("id") or "model"
        start = float(points[0].get("value") or current_price)
        end = float(points[-1].get("value") or start)
        if language == "ko":
            bullets.append(f"{label}: {_fmt_price(start)}에서 {_fmt_price(end)}로 변화 ({_fmt_pct(_pct(start, end))})")
        else:
            bullets.append(f"{label}: {_fmt_price(start)} -> {_fmt_price(end)} ({_fmt_pct(_pct(start, end))})")
    return bullets


def _public_adapter_read(path_adapter: dict[str, Any] | None, language: str) -> str:
    adapter = str((path_adapter or {}).get("adapter") or "")
    if language == "ko":
        return {
            "geopolitical_supply_shock": "지정학적 긴장과 공급 차질 가능성이 유가에 프리미엄을 더한 것으로 해석됩니다.",
            "bullish_event_breakout": "뉴스 분위기와 최근 가격 흐름이 상방 돌파 가능성을 키운 것으로 해석됩니다.",
            "event_risk_premium": "에너지와 지정학 관련 뉴스가 이어지면서 하방보다 상방 위험을 더 크게 반영했습니다.",
            "overextended_mean_reversion": "최근 가격 흐름이 빠르게 올라 단기 되돌림 가능성도 함께 반영했습니다.",
            "pattern_residual_detemplate": "최근 차트의 고점과 저점 흐름을 반영해 단순 직선 예측을 피했습니다.",
        }.get(adapter, "")
    return {
        "geopolitical_supply_shock": "Geopolitical tension and possible supply disruption appear to be adding a crude-risk premium.",
        "bullish_event_breakout": "News tone and recent price action point to a possible upside breakout setup.",
        "event_risk_premium": "Energy and geopolitical headlines are adding more upside risk than downside risk.",
        "overextended_mean_reversion": "Recent price action looks stretched, so the path also allows for short-term cooling.",
        "pattern_residual_detemplate": "The path reflects recent peaks and troughs instead of a straight-line forecast.",
    }.get(adapter, "")


def _markdown(report: ForecastReport, language: str) -> str:
    labels = (
        {
            "generated": "작성 시각",
            "symbol": "종목",
            "interval": "주기",
            "horizon": "예측 길이",
            "summary": "핵심 요약",
            "metrics": "핵심 지표",
            "sections": "상세 내용",
            "warnings": "주의 사항",
            "note": "참고",
        }
        if language == "ko"
        else {
            "generated": "Generated",
            "symbol": "Symbol",
            "interval": "Interval",
            "horizon": "Horizon",
            "summary": "Executive Summary",
            "metrics": "Key Metrics",
            "sections": "Sections",
            "warnings": "Warnings",
            "note": "Note",
        }
    )
    lines = [
        f"# {report.title}",
        "",
        f"- {labels['generated']}: {report.generated_at.isoformat()}",
        f"- {labels['symbol']}: {report.symbol}",
        f"- {labels['interval']}: {report.interval}",
        f"- {labels['horizon']}: {report.horizon}",
        "",
        f"## {labels['summary']}",
        report.executive_summary,
        "",
        f"## {labels['metrics']}",
    ]
    for key, value in report.key_metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", f"## {labels['sections']}"])
    for section in report.sections:
        lines.extend(["", f"### {section.title}", section.body])
        lines.extend(bullet for bullet in section.bullets)
    if report.warnings:
        lines.extend(["", f"## {labels['warnings']}"])
        lines.extend(f"- {warning}" for warning in report.warnings)
    if report.recommendation_note:
        lines.extend(["", f"## {labels['note']}", report.recommendation_note])
    return "\n".join(lines)


@router.get("/api/report", response_model=ForecastReport)
def forecast_report(
    symbol: str = Query(default=None),
    interval: str = Query(default=None),
    horizon: int | None = Query(default=None, ge=1),
    models: str | None = Query(default=None),
    language: str = Query(default="ko"),
):
    language = "en" if language == "en" else "ko"
    settings = get_settings()
    try:
        bundle = build_forecast(
            symbol=symbol or settings.default_symbol,
            interval=interval or settings.default_interval,
            horizon=horizon,
            models=models,
            include_scenarios=True,
            settings=settings,
        )
    except (MarketDataUnavailable, PretrainedModelNotFoundError, ForecastUnavailable) as exc:
        raise service_error(exc) from exc
    except InvalidModelRequest as exc:
        raise HTTPException(status_code=400, detail=exc.as_detail()) from exc

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
    regime_label = _label_direction(regime, language)

    generated_at = datetime.now(timezone.utc)
    horizon = len(response.forecast)
    horizon_text = _horizon_label(horizon, response.interval, language)
    period_text = _period_label(first.time if first else None, last.time if last else None, language)
    generated_date = _local_date(generated_at)
    generated_datetime = _local_datetime(generated_at)
    title = (
        f"{response.symbol} {response.interval.upper()} 예측 리포트"
        if language == "ko"
        else f"{response.symbol} {response.interval.upper()} Forecast Report"
    )
    if language == "ko":
        executive = (
            f"작성일 {generated_date} 기준 예측기간은 {period_text}입니다. "
            f"현재 유가는 {_fmt_price(current)}이며, 모델은 {horizon_text}까지 "
            f"{_label_direction(direction, language)} 흐름에 가까운 시나리오를 우선 보고 있습니다. "
            f"{horizon_text} 중앙값은 {_fmt_price(median_end)}({_fmt_pct(median_change)})이며, "
            f"예상 변동 범위는 대략 {_fmt_price(p10_end)}에서 {_fmt_price(p90_end)}입니다. "
            f"현재 시장 흐름은 {regime_label}에 가깝습니다."
        )
    else:
        direction_article = "an" if direction[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        executive = (
            f"As of {generated_date}, the forecast period is {period_text}. "
            f"Crude is at {_fmt_price(current)}, and the model points to {direction_article} {direction} scenario {horizon_text}. "
            f"Current price is {_fmt_price(current)}, the {horizon_text} median estimate is "
            f"{_fmt_price(median_end)} ({_fmt_pct(median_change)}), with an expected range near "
            f"{_fmt_price(p10_end)} to {_fmt_price(p90_end)}. The dominant detected regime is {regime}."
        )

    primary_adapter = {}
    if response.primary_model:
        primary_adapter = (response.deep_model_info.get(response.primary_model) or {}).get("path_adapter") or {}
    adapter_read = _public_adapter_read(primary_adapter, language)
    direction_read = (
        "상방 압력이 우세하지만, 실제 경로는 뉴스 변화와 단기 수급에 따라 흔들릴 수 있습니다."
        if language == "ko" and direction == "upside"
        else "하방 압력이 우세하지만, 공급 뉴스가 바뀌면 반등이 빨라질 수 있습니다."
        if language == "ko" and direction == "downside"
        else "뚜렷한 한쪽 방향보다 박스권 안에서의 등락 가능성이 더 크게 반영됐습니다."
        if language == "ko"
        else "Upside pressure is dominant, but the actual path can still swing with news and short-term supply-demand changes."
        if direction == "upside"
        else "Downside pressure is dominant, but supply headlines can still trigger a quick rebound."
        if direction == "downside"
        else "The model sees more range-bound movement than a decisive directional break."
    )
    sections = [
        ReportSection(
            title="핵심 전망" if language == "ko" else "Core View",
            body=(
                f"{horizon_text} 기준 중앙 전망은 {_fmt_price(median_end)}이며 현재가 대비 {_fmt_pct(median_change)}입니다. "
                f"예상 범위는 {_fmt_price(p10_end)}에서 {_fmt_price(p90_end)} 사이로 넓게 열려 있어, 방향성은 있지만 변동성도 함께 고려해야 합니다."
                if language == "ko"
                else f"The {horizon_text} median view is {_fmt_price(median_end)}, or {_fmt_pct(median_change)} from the current price. "
                f"The expected range is broad, from {_fmt_price(p10_end)} to {_fmt_price(p90_end)}, so the directional view should be read together with volatility."
            ),
            bullets=[
                direction_read,
            ],
        ),
        ReportSection(
            title="시황 해석" if language == "ko" else "Market Read",
            body=(
                (
                    f"현재 시장은 {regime_label} 흐름에 가까우며, 최근 차트와 뉴스 흐름을 함께 보면 {adapter_read or direction_read}"
                )
                if language == "ko"
                else (
                    f"The market currently looks closest to a {regime} regime. Taken together with recent chart and news flow, "
                    f"{adapter_read or direction_read}"
                )
            ),
            bullets=[
                (
                    "특히 원유 재고, OPEC 관련 발언, 중동과 러시아를 둘러싼 공급 뉴스, 달러와 금리 흐름이 앞으로의 방향성을 좌우할 가능성이 큽니다."
                    if language == "ko"
                    else "Inventories, OPEC communication, Middle East and Russia supply headlines, and dollar/rate moves are likely to matter most for the next leg."
                )
            ],
        ),
        ReportSection(
            title="확인할 변수" if language == "ko" else "What To Watch",
            body=(
                "상승 시나리오가 이어지려면 공급 차질 우려가 유지되거나 수요 둔화 우려가 완화되어야 합니다. 반대로 달러 강세, 금리 상승, 재고 증가, 지정학 긴장 완화는 예측 경로를 낮출 수 있습니다."
                if language == "ko"
                else "For the upside scenario to persist, supply-disruption concerns need to remain in place or demand worries need to ease. A stronger dollar, higher rates, rising inventories, or easing geopolitical tension could pull the path lower."
            ),
            bullets=[
                (
                    "따라서 이 리포트는 하나의 숫자보다 가격 흐름, 뉴스 흐름, 수급 변화를 함께 읽는 자료로 보는 편이 적절합니다."
                    if language == "ko"
                    else "This report is best read as a combined view of price action, news flow, and supply-demand changes rather than as a single number."
                )
            ],
        ),
    ]

    warnings = list(response.warnings or [])
    warnings.extend(w.message for w in response.warning_objects if getattr(w, "message", None))
    if response.calibration_status.get("calibration_status") != "calibrated":
        warnings = []

    report = ForecastReport(
        generated_at=generated_at,
        symbol=response.symbol,
        interval=response.interval,
        horizon=horizon,
        mode="deterministic_report",
        llm_used=False,
        source_note=(
            "이 리포트는 외부 LLM fallback이 아니라 모델 예측값과 규칙 기반 문장 템플릿으로 작성됩니다."
            if language == "ko"
            else "This report is generated from model outputs and rule-based prose templates, not an external LLM fallback."
        ),
        title=title,
        executive_summary=executive,
        recommendation_note="",
        key_metrics={
            ("작성일" if language == "ko" else "report_date"): generated_date,
            ("예측기간" if language == "ko" else "forecast_period"): period_text,
            ("현재가" if language == "ko" else "current_price"): _fmt_price(current),
            (f"{horizon_text}_중앙값" if language == "ko" else f"{horizon_text}_median"): _fmt_price(median_end),
            ("중앙값_변화율" if language == "ko" else "median_change"): _fmt_pct(median_change),
            ("예상_변동_범위" if language == "ko" else "expected_range"): f"{_fmt_price(p10_end)} - {_fmt_price(p90_end)}",
            ("시장_흐름" if language == "ko" else "market_flow"): regime_label if language == "ko" else regime,
            ("작성_시각" if language == "ko" else "generated_at_local"): generated_datetime,
        },
        sections=sections,
        warnings=warnings,
        markdown="",
    )
    return report.model_copy(update={"markdown": _markdown(report, language)})
