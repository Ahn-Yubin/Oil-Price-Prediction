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
    title = (
        f"{response.symbol} {response.interval.upper()} 예측 리포트"
        if language == "ko"
        else f"{response.symbol} {response.interval.upper()} Forecast Report"
    )
    if language == "ko":
        executive = (
            f"선택된 예측 모델 묶음은 향후 {len(response.forecast)}개 {response.interval} 구간에서 "
            f"중앙 경로가 {_label_direction(direction, language)} 흐름에 가깝다고 봅니다. 현재가는 {_fmt_price(current)}, "
            f"예측 마지막 시점의 중앙값은 {_fmt_price(median_end)}({_fmt_pct(median_change)})이며, "
            f"마지막 시점의 P10-P90 범위는 대략 {_fmt_price(p10_end)}에서 {_fmt_price(p90_end)}입니다. "
            f"감지된 주요 시장 국면은 {regime_label}입니다."
        )
    else:
        executive = (
            f"The selected forecast stack points to a {direction} median path over {len(response.forecast)} "
            f"{response.interval} steps. Current price is {_fmt_price(current)}, median terminal estimate is "
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
                f"현재가: {_fmt_price(current)}" if language == "ko" else f"Current price: {_fmt_price(current)}",
                (
                    f"마지막 시점 중앙값: {_fmt_price(median_end)} ({_fmt_pct(median_change)})"
                    if language == "ko"
                    else f"Median terminal estimate: {_fmt_price(median_end)} ({_fmt_pct(median_change)})"
                ),
                (
                    f"마지막 시점 P10-P90 범위: {_fmt_price(p10_end)} - {_fmt_price(p90_end)}"
                    if language == "ko"
                    else f"Terminal P10-P90 range: {_fmt_price(p10_end)} - {_fmt_price(p90_end)}"
                ),
                (
                    f"주요 모델: {response.primary_model or response.model_version or '선택된 예측 모델 묶음'}"
                    if language == "ko"
                    else f"Primary model: {response.primary_model or response.model_version or 'selected forecast stack'}"
                ),
            ],
        ),
        ReportSection(
            title="모델 비교" if language == "ko" else "Model Comparison",
            body=(
                "표시 가능한 모델 경로를 시작점 대비 마지막 값 변화로 요약했습니다."
                if language == "ko"
                else "Visible model paths are summarized by terminal movement from their anchor point."
            ),
            bullets=_model_path_summary(bundle.forecast_models or response.model_paths, current, language)
            or (["모델 경로 비교 데이터가 없습니다."] if language == "ko" else ["No model path comparison was available."]),
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
        warnings.append(
            "예측 밴드는 변동성 기반이거나 아직 보정되지 않았을 수 있으므로, 활용 전 커버리지 검증이 필요합니다."
            if language == "ko"
            else "Forecast bands may be volatility-derived or uncalibrated; validate coverage before relying on them."
        )

    report = ForecastReport(
        generated_at=generated_at,
        symbol=response.symbol,
        interval=response.interval,
        horizon=len(response.forecast),
        title=title,
        executive_summary=executive,
        recommendation_note=(
            "이 리포트는 연구와 모니터링을 위한 모델 출력 요약이며, 금융 조언이 아닙니다."
            if language == "ko"
            else "This report summarizes model outputs for research and monitoring only. It is not financial advice."
        ),
        key_metrics={
            ("현재가" if language == "ko" else "current_price"): _fmt_price(current),
            ("마지막_중앙값" if language == "ko" else "median_terminal"): _fmt_price(median_end),
            ("중앙값_변화율" if language == "ko" else "median_change"): _fmt_pct(median_change),
            ("마지막_P10" if language == "ko" else "p10_terminal"): _fmt_price(p10_end),
            ("마지막_P90" if language == "ko" else "p90_terminal"): _fmt_price(p90_end),
            ("주요_시장_국면" if language == "ko" else "dominant_regime"): regime_label if language == "ko" else regime,
            ("신뢰도" if language == "ko" else "confidence"): f"{round(confidence * 100)}%" if confidence is not None else "-",
            ("작성_시각" if language == "ko" else "generated_at_local"): pd.Timestamp(generated_at).tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S KST"),
        },
        sections=sections,
        warnings=warnings,
        markdown="",
    )
    return report.model_copy(update={"markdown": _markdown(report, language)})
