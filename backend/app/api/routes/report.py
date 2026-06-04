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


def _market_context_summary(summary: dict[str, Any], language: str) -> list[str]:
    bullets: list[str] = []
    if not summary:
        return bullets
    bias = summary.get("overall_bias")
    impact = summary.get("impact_score")
    uncertainty = summary.get("uncertainty")
    explanation = summary.get("explanation")
    if bias:
        bullets.append(f"컨텍스트 방향성: {bias}" if language == "ko" else f"Context bias: {bias}")
    if impact is not None:
        bullets.append(f"컨텍스트 영향 점수: {float(impact):.2f}" if language == "ko" else f"Context impact score: {float(impact):.2f}")
    if uncertainty is not None:
        bullets.append(f"컨텍스트 불확실성: {float(uncertainty):.2f}" if language == "ko" else f"Context uncertainty: {float(uncertainty):.2f}")
    if explanation:
        bullets.append(str(explanation))
    return bullets


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
        lines.extend(f"- {bullet}" for bullet in section.bullets)
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
    confidence = float(first.confidence) if first else None

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
            f"현재 예측 모델은 {horizon_text}까지 중앙 경로가 "
            f"{_label_direction(direction, language)} 흐름에 가깝다고 봅니다. 현재가는 {_fmt_price(current)}, "
            f"{horizon_text} 중앙값은 {_fmt_price(median_end)}({_fmt_pct(median_change)})이며, "
            f"{horizon_text} P10-P90 범위는 대략 {_fmt_price(p10_end)}에서 {_fmt_price(p90_end)}입니다. "
            f"감지된 주요 시장 국면은 {regime_label}입니다."
        )
    else:
        direction_article = "an" if direction[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        executive = (
            f"As of {generated_date}, the forecast period is {period_text}. "
            f"The forecast model points to {direction_article} {direction} median path {horizon_text}. "
            f"Current price is {_fmt_price(current)}, the {horizon_text} median estimate is "
            f"{_fmt_price(median_end)} ({_fmt_pct(median_change)}), with an end-band range near "
            f"{_fmt_price(p10_end)} to {_fmt_price(p90_end)}. The dominant detected regime is {regime}."
        )

    sections = [
        ReportSection(
            title="예측 경로" if language == "ko" else "Forecast Path",
            body=(
                "중앙값과 분위수 경로는 모델 파이프라인의 산출물이며, 매매 지시가 아닙니다."
                if language == "ko"
                else "Median and quantile paths are produced by the model forecast pipeline; they are not trading instructions."
            ),
            bullets=[
                f"작성일: {generated_date}" if language == "ko" else f"Report date: {generated_date}",
                f"예측기간: {period_text}" if language == "ko" else f"Forecast period: {period_text}",
                f"현재가: {_fmt_price(current)}" if language == "ko" else f"Current price: {_fmt_price(current)}",
                (
                    f"{horizon_text} 중앙값: {_fmt_price(median_end)} ({_fmt_pct(median_change)})"
                    if language == "ko"
                    else f"{horizon_text.title()} median estimate: {_fmt_price(median_end)} ({_fmt_pct(median_change)})"
                ),
                (
                    f"{horizon_text} P10-P90 범위: {_fmt_price(p10_end)} - {_fmt_price(p90_end)}"
                    if language == "ko"
                    else f"{horizon_text.title()} P10-P90 range: {_fmt_price(p10_end)} - {_fmt_price(p90_end)}"
                ),
            ],
        ),
        ReportSection(
            title="리스크와 컨텍스트" if language == "ko" else "Risk And Context",
            body=(
                "시장 국면, 밴드 보정 상태, 컨텍스트 신호는 예측 신뢰도를 해석하는 보조 정보입니다."
                if language == "ko"
                else "Regime, calibration, and context signals help explain forecast reliability."
            ),
            bullets=[
                f"주요 시장 국면: {regime_label}" if language == "ko" else f"Dominant regime: {regime}",
                (
                    f"예측 신뢰도: {round(confidence * 100)}%"
                    if language == "ko" and confidence is not None
                    else "예측 신뢰도: -"
                    if language == "ko"
                    else f"Forecast confidence: {round(confidence * 100)}%"
                    if confidence is not None
                    else "Forecast confidence: -"
                ),
                f"데이터 상태: {response.data_status.status}" if language == "ko" else f"Data status: {response.data_status.status}",
                (
                    f"밴드 상태: {response.calibration_status.get('calibration_status', 'unknown')}"
                    if language == "ko"
                    else f"Band status: {response.calibration_status.get('calibration_status', 'unknown')}"
                ),
                *_market_context_summary(response.llm_context_summary, language),
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
        title=title,
        executive_summary=executive,
        recommendation_note="",
        key_metrics={
            ("작성일" if language == "ko" else "report_date"): generated_date,
            ("예측기간" if language == "ko" else "forecast_period"): period_text,
            ("현재가" if language == "ko" else "current_price"): _fmt_price(current),
            (f"{horizon_text}_중앙값" if language == "ko" else f"{horizon_text}_median"): _fmt_price(median_end),
            ("중앙값_변화율" if language == "ko" else "median_change"): _fmt_pct(median_change),
            (f"{horizon_text}_P10" if language == "ko" else f"{horizon_text}_p10"): _fmt_price(p10_end),
            (f"{horizon_text}_P90" if language == "ko" else f"{horizon_text}_p90"): _fmt_price(p90_end),
            ("주요_시장_국면" if language == "ko" else "dominant_regime"): regime_label if language == "ko" else regime,
            ("신뢰도" if language == "ko" else "confidence"): f"{round(confidence * 100)}%" if confidence is not None else "-",
            ("작성_시각" if language == "ko" else "generated_at_local"): generated_datetime,
        },
        sections=sections,
        warnings=warnings,
        markdown="",
    )
    return report.model_copy(update={"markdown": _markdown(report, language)})
