from datetime import datetime, timezone

import pytest
import pandas as pd

httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import backend.app.main as main
from backend.app.api.routes import backtests as backtests_route
from backend.app.api.routes import chart as chart_route
from backend.app.api.routes import data_status as data_status_route
from backend.app.api.routes import explanation as explanation_route
from backend.app.api.routes import features as features_route
from backend.app.api.routes import forecast as forecast_route
from backend.app.api.routes import market_context as market_context_route
from backend.app.api.routes import models as models_route
from backend.app.api.routes import report as report_route
from backend.app.api.routes import scenarios as scenarios_route
from market_ai.config import Settings
from market_ai.modeling.deep.availability import DeepArtifactAvailability
from market_ai.schemas.market import (
    AssetClass,
    AssetMetadata,
    Candle,
    DataStatus,
    ForecastPoint,
    ForecastResponse,
    MarketDataWindow,
    MarketSymbol,
    RegimeProbabilities,
    ScenarioForecastResponse,
    ScenarioPoint,
    Timeframe,
)
from market_ai.forecasting.service import ForecastBundle


def _data_status() -> DataStatus:
    return DataStatus(
        status="real",
        source="unit-test",
        symbol_requested="CL=F",
        symbol_resolved="CL=F",
        interval_requested="1d",
        interval_resolved="1d",
        last_bar_time=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        is_stale=False,
        warnings=[],
    )


def _market_window() -> MarketDataWindow:
    return MarketDataWindow(
        symbol=MarketSymbol(
            requested="CL=F",
            normalized="CL=F",
            provider_symbol="CL=F",
            asset_class=AssetClass.futures,
        ),
        timeframe=Timeframe(
            requested="1d",
            normalized="1d",
            provider_interval="1d",
            provider_period="10y",
            seconds=86400,
        ),
        candles=[
            Candle(time=1_700_000_000, open=80.0, high=81.0, low=79.0, close=80.5, volume=1.0),
            Candle(time=1_700_086_400, open=80.5, high=82.0, low=80.0, close=81.0, volume=1.0),
        ],
        data_status=_data_status(),
    )


def _forecast_bundle() -> ForecastBundle:
    status = _data_status()
    response = ForecastResponse(
        symbol="CL=F",
        asset_metadata=AssetMetadata(symbol="CL=F", provider_symbol="CL=F", asset_class=AssetClass.futures),
        interval="1d",
        generated_at=datetime.now(timezone.utc),
        current_price=81.0,
        model_version="test",
        training_cutoff="2026-01-01T00:00:00+00:00",
        data_status=status,
        forecast=[
            ForecastPoint(
                time=1_700_172_800,
                p05=78.0,
                p10=79.0,
                p25=80.0,
                p50=82.0,
                p75=83.0,
                p90=84.0,
                p95=85.0,
                expected_return=0.01,
                expected_volatility=0.02,
                prob_up=0.6,
                confidence=0.7,
            )
        ],
        regime=RegimeProbabilities().normalized(),
        models=[],
        warnings=[],
    )
    return ForecastBundle(
        response=response,
        market_data=_market_window(),
        forecast_models=[],
        metrics={"mae": None, "rmse": None, "mape": None, "model": "unit"},
        model_info={},
        horizon=1,
    )


def _enable_fake_model_commentary_llm(monkeypatch, payload=None, captured=None):
    payload = payload or {
        "summary": "현재 예측은 공급 뉴스와 최근 차트 흐름을 함께 반영해 상방 쪽으로 읽힙니다.",
        "model_interpretation": "차트 흐름과 뉴스 근거가 같은 방향으로 작용했습니다.",
        "risk_notes": ["재고와 OPEC 관련 뉴스가 바뀌면 방향이 달라질 수 있습니다."],
        "warnings": [],
    }
    monkeypatch.setattr(
        market_context_route,
        "get_settings",
        lambda: Settings(enable_external_llm_calls=True, llm_api_key="unit-key", llm_context_mode="openai"),
    )
    monkeypatch.setattr(market_context_route, "_reserve_llm_call", lambda: True)

    def fake_llm(_settings, prompt):
        if captured is not None:
            captured["prompt"] = prompt
        return payload

    monkeypatch.setattr(market_context_route, "_openai_compatible_model_commentary", fake_llm)


def test_health_and_models_endpoints():
    client = TestClient(main.app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/models").status_code == 200


def test_models_endpoint_uses_requested_interval_horizon(monkeypatch):
    class FakeRegistry:
        def __init__(self, settings):
            self.settings = settings

        def scan(self):
            return []

        def resolve(self, *, model_name, interval=None, horizon=None, asset_class=None):
            raise models_route.ModelArtifactNotFound("missing")

    seen = []

    def fake_deep_availability(*, settings, model_name, interval, horizon):
        seen.append((model_name, interval, horizon))
        return DeepArtifactAvailability(
            model_name=model_name,
            interval=interval,
            horizon=horizon,
            status="artifact_missing",
            expected_artifact_file=f"{model_name}_{interval}_h{horizon}.pt",
            expected_metadata_file=f"{model_name}_{interval}_h{horizon}.json",
            artifact_path=settings.model_dir / f"{model_name}_{interval}_h{horizon}.pt",
            metadata_path=settings.metadata_dir / f"{model_name}_{interval}_h{horizon}.json",
            training_command="train",
            metadata={},
            reason="missing",
        )

    monkeypatch.setattr(models_route, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(models_route, "deep_artifact_availability", fake_deep_availability)

    client = TestClient(main.app)
    response = client.get("/api/models?interval=1h&horizon=7")
    body = response.json()

    assert response.status_code == 200
    assert ("oil_context_fusion", "1h", 30) in seen
    oil_model = next(item for item in body["user_facing_models"] if item["id"] == "oil_context_fusion")
    assert oil_model["status"] == "artifact_missing"
    assert "oil_context_fusion_1h_h30.pt" in oil_model["expected_artifact_file"]
    assert body["deep_artifact_policy"]["display_horizon"] == 7
    assert body["deep_artifact_policy"]["artifact_horizon"] == 30


def test_models_endpoint_uses_single_1d_artifact_horizon_for_display_slices(monkeypatch):
    class FakeRegistry:
        def __init__(self, settings):
            self.settings = settings

        def scan(self):
            return []

        def resolve(self, *, model_name, interval=None, horizon=None, asset_class=None):
            raise models_route.ModelArtifactNotFound("missing")

    seen = []

    def fake_deep_availability(*, settings, model_name, interval, horizon):
        seen.append((model_name, interval, horizon))
        return DeepArtifactAvailability(
            model_name=model_name,
            interval=interval,
            horizon=horizon,
            status="artifact_missing",
            expected_artifact_file=f"{model_name}_{interval}_h{horizon}.pt",
            expected_metadata_file=f"{model_name}_{interval}_h{horizon}.json",
            artifact_path=settings.model_dir / f"{model_name}_{interval}_h{horizon}.pt",
            metadata_path=settings.metadata_dir / f"{model_name}_{interval}_h{horizon}.json",
            training_command="train",
            metadata={},
            reason="missing",
        )

    monkeypatch.setattr(models_route, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(models_route, "deep_artifact_availability", fake_deep_availability)

    client = TestClient(main.app)
    body_7 = client.get("/api/models?interval=1d&horizon=7").json()
    body_14 = client.get("/api/models?interval=1d&horizon=14").json()

    assert ("oil_context_fusion", "1d", 30) in seen
    assert body_7["deep_artifact_policy"]["artifact_horizon"] == 30
    assert body_7["deep_artifact_policy"]["display_horizon"] == 7
    assert body_14["deep_artifact_policy"]["artifact_horizon"] == 30
    assert body_14["deep_artifact_policy"]["display_horizon"] == 14


def test_data_status_endpoint(monkeypatch):
    monkeypatch.setattr(data_status_route, "load_market_data_window", lambda *args, **kwargs: _market_window())
    client = TestClient(main.app)
    response = client.get("/api/data-status?symbol=CL=F&interval=1d")
    assert response.status_code == 200
    assert response.json()["status"] == "real"


def test_forecast_schema_endpoint(monkeypatch):
    monkeypatch.setattr(forecast_route, "build_forecast", lambda **kwargs: _forecast_bundle())
    client = TestClient(main.app)
    response = client.get("/api/forecast?symbol=CL=F&interval=1d")
    body = response.json()
    assert response.status_code == 200
    assert body["forecast"][0]["p05"] <= body["forecast"][0]["p50"] <= body["forecast"][0]["p95"]
    assert body["data_status"]["status"] == "real"


def test_scenario_forecast_endpoint_returns_event_context_path(monkeypatch):
    captured = {}

    def fake_build_scenario_forecast(**kwargs):
        captured.update(kwargs)
        return ScenarioForecastResponse(
            scenario_id="scenario-unit",
            title=kwargs["title"],
            content=kwargs["content"],
            symbol="CL=F",
            interval="1d",
            generated_at=datetime.now(timezone.utc),
            event_time=kwargs.get("event_time"),
            current_price=81.0,
            points=[
                ScenarioPoint(time=1_700_086_400, value=81.0),
                ScenarioPoint(time=1_700_172_800, value=86.0),
            ],
            forecast=_forecast_bundle().response.forecast,
            data_status=_data_status(),
            primary_model="oil_context_fusion",
            llm_context_summary={
                "role": "context/event encoder only",
                "event_context_source": "scenario_override",
                "overall_bias": "bullish",
            },
            llm_context={"overall_bias": "bullish", "events": []},
            warnings=[],
        )

    monkeypatch.setattr(scenarios_route, "build_scenario_forecast", fake_build_scenario_forecast)
    client = TestClient(main.app)
    response = client.post(
        "/api/scenarios/forecast",
        json={
            "title": "Iran shock",
            "content": "Hypothetical supply disruption in Iran.",
            "event_time": "2026-06-14T00:00:00Z",
            "events": [
                {
                    "title": "Hormuz disruption",
                    "content": "Shipping disruption raises crude supply risk.",
                    "event_time": "2026-06-21T00:00:00Z",
                }
            ],
            "symbol": "CL=F",
            "interval": "1d",
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["scenario_id"] == "scenario-unit"
    assert body["points"][0]["value"] == 81.0
    assert body["llm_context_summary"]["event_context_source"] == "scenario_override"
    assert captured["events"][0].title == "Hormuz disruption"
    assert "target_price" not in str(body).lower()


def test_chart_backward_compatibility_endpoint(monkeypatch):
    monkeypatch.setattr(chart_route, "build_forecast", lambda **kwargs: _forecast_bundle())
    client = TestClient(main.app)
    response = client.get("/api/chart?symbol=CL=F&interval=1d")
    body = response.json()
    assert response.status_code == 200
    for key in ["candles", "predicted", "predicted_lower", "predicted_upper", "metrics", "updated_at"]:
        assert key in body
    assert body["data_status"]["status"] == "real"
    assert "band_explanation" in body


def test_features_endpoint_schema(monkeypatch):
    monkeypatch.setattr(features_route, "load_market_data_window", lambda *args, **kwargs: _market_window())
    client = TestClient(main.app)
    response = client.get("/api/features?symbol=CL=F&interval=1d")
    body = response.json()
    assert response.status_code == 200
    assert body["feature_set_version"] == "price_v1"
    assert body["data_status"]["status"] == "real"
    assert "log_return" in body["summary"]


def test_explanation_endpoint_uses_context_encoder_not_price_forecaster(monkeypatch):
    monkeypatch.setattr(explanation_route, "build_forecast", lambda **kwargs: _forecast_bundle())
    client = TestClient(main.app)
    response = client.get("/api/explanation?symbol=CL=F&interval=1d")
    body = response.json()
    assert response.status_code == 200
    assert body["symbol"] == "CL=F"
    assert "main_drivers" in body
    assert "target_price" not in str(body).lower()


def test_market_context_endpoint_returns_overlay_payload(monkeypatch):
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: _forecast_bundle())
    monkeypatch.setattr(market_context_route, "_news_items", lambda **kwargs: [{"time": 1_700_000_000, "headline": "OPEC supply cut", "symbol": "CL=F"}])
    monkeypatch.setattr(
        market_context_route,
        "_context_points",
        lambda **kwargs: [{"time": 1_700_000_000, "overall_bias": "bullish", "impact_score": 0.5, "event_count": 1}],
    )
    client = TestClient(main.app)
    response = client.get("/api/market-context?symbol=CL=F&interval=1d")
    body = response.json()
    assert response.status_code == 200
    assert body["news"][0]["headline"] == "OPEC supply cut"
    assert body["context_points"][0]["overall_bias"] == "bullish"
    assert body["chart_context_points"][0]["news_items"][0]["headline"] == "OPEC supply cut"
    assert "scenario_commentary" in body


def test_market_context_chart_markers_sample_full_origin_window(monkeypatch):
    monkeypatch.setattr(market_context_route, "_forecast_bundle_for_commentary", lambda **kwargs: _forecast_bundle())
    news = [
        {"time": 1_700_000_000 + idx * 86_400, "headline": f"Oil headline {idx}", "symbol": "CL=F"}
        for idx in range(24)
    ]
    context_points = [
        {
            "time": item["time"],
            "overall_bias": "bullish" if idx % 2 else "bearish",
            "impact_score": 0.5,
            "event_count": 1,
            "explanation": item["headline"],
        }
        for idx, item in enumerate(news)
    ]
    monkeypatch.setattr(market_context_route, "_news_items", lambda **kwargs: news)
    monkeypatch.setattr(market_context_route, "_context_points", lambda **kwargs: context_points)

    client = TestClient(main.app)
    response = client.get("/api/market-context?symbol=CL=F&interval=1d&origin_time=2026-01-30T09:00:00%2B09:00&limit=3")
    body = response.json()

    assert response.status_code == 200
    assert [item["headline"] for item in body["news"]] == ["Oil headline 21", "Oil headline 22", "Oil headline 23"]
    marker_times = [point["time"] for point in body["chart_context_points"]]
    assert marker_times[0] == news[0]["time"]
    assert marker_times[-1] == news[-1]["time"]
    assert len(marker_times) == 6


def test_market_context_chart_markers_use_spread_news_pool(monkeypatch):
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: _forecast_bundle())
    latest_news = [
        {"time": 1_780_000_000 + idx * 60, "headline": f"latest oil news {idx}", "symbol": "CL=F"}
        for idx in range(5)
    ]
    spread_news = [
        {"time": 1_700_000_000 + idx * 2_592_000, "headline": f"128d context news {idx}", "symbol": "CL=F"}
        for idx in range(4)
    ]
    samplings = []

    def fake_news_items(**kwargs):
        samplings.append(kwargs.get("sampling", "tail"))
        return spread_news if kwargs.get("sampling") == "spread" else latest_news

    monkeypatch.setattr(market_context_route, "_news_items", fake_news_items)
    monkeypatch.setattr(market_context_route, "_context_points", lambda **kwargs: [])

    client = TestClient(main.app)
    response = client.get("/api/market-context?symbol=CL=F&interval=1d&limit=3")
    body = response.json()

    assert response.status_code == 200
    assert [item["headline"] for item in body["news"]] == ["latest oil news 2", "latest oil news 3", "latest oil news 4"]
    marker_times = [point["time"] for point in body["chart_context_points"]]
    assert marker_times == [item["time"] for item in spread_news]
    assert "tail" in samplings
    assert "spread" in samplings


def test_market_context_chart_markers_coalesce_adjacent_news(monkeypatch):
    monkeypatch.setattr(market_context_route, "_forecast_bundle_for_commentary", lambda **kwargs: _forecast_bundle())
    adjacent_news = [
        {"time": 1_700_000_000 + idx * 43_200, "headline": f"Adjacent oil news {idx}", "symbol": "CL=F"}
        for idx in range(4)
    ]
    monkeypatch.setattr(market_context_route, "_news_items", lambda **kwargs: adjacent_news)
    monkeypatch.setattr(market_context_route, "_context_points", lambda **kwargs: [])

    client = TestClient(main.app)
    response = client.get("/api/market-context?symbol=CL=F&interval=1d&origin_time=2026-01-30T09:00:00%2B09:00")
    body = response.json()

    assert response.status_code == 200
    assert len(body["chart_context_points"]) == 1
    assert body["chart_context_points"][0]["event_count"] == 4
    assert len(body["chart_context_points"][0]["news_items"]) == 4


def test_market_context_origin_time_uses_point_in_time_cache(monkeypatch):
    monkeypatch.setattr(market_context_route, "_forecast_bundle_for_commentary", lambda **kwargs: _forecast_bundle())
    news_calls = []

    def fake_news_items(**kwargs):
        news_calls.append(kwargs)
        return [{"time": 1_700_000_000, "headline": "OPEC supply cut", "symbol": "CL=F"}]

    monkeypatch.setattr(market_context_route, "_news_items", fake_news_items)
    monkeypatch.setattr(
        market_context_route,
        "_context_points",
        lambda **kwargs: [{"time": 1_700_000_000, "overall_bias": "bullish", "impact_score": 0.5, "event_count": 1}],
    )
    client = TestClient(main.app)
    response = client.get("/api/market-context?symbol=CL=F&interval=1d&live=1&origin_time=2026-01-30T09:00:00%2B09:00")
    body = response.json()

    assert response.status_code == 200
    assert body["news_source"] == "point_in_time_news_cache"
    assert body["news"][0]["headline"] == "OPEC supply cut"
    assert body["context_points"][0]["overall_bias"] == "bullish"
    assert news_calls
    assert (news_calls[0]["end_ts"] - news_calls[0]["start_ts"]).days == 128


def test_model_commentary_origin_time_does_not_use_live_context(monkeypatch):
    monkeypatch.setattr(market_context_route, "_forecast_bundle_for_commentary", lambda **kwargs: _forecast_bundle())
    monkeypatch.setattr(market_context_route, "_news_items", lambda **kwargs: [{"time": 1_700_000_000, "headline": "OPEC supply cut", "symbol": "CL=F"}])
    monkeypatch.setattr(
        market_context_route,
        "_context_points",
        lambda **kwargs: [{"time": 1_700_000_000, "overall_bias": "bullish", "impact_score": 0.5, "event_count": 1}],
    )

    def fail_live_context(**_kwargs):
        raise AssertionError("origin commentary must not request live context")

    monkeypatch.setattr(market_context_route, "_load_live_context_payload", fail_live_context)
    _enable_fake_model_commentary_llm(monkeypatch)
    market_context_route._MODEL_COMMENTARY_CACHE.clear()
    client = TestClient(main.app)
    response = client.get("/api/model-commentary?symbol=CL=F&interval=1d&language=en&origin_time=2026-01-30T09:00:00%2B09:00")
    body = response.json()

    assert response.status_code == 200
    assert body["market_context"]["source"] == "point_in_time_news_cache"


def test_forecast_report_endpoint_returns_user_summary(monkeypatch):
    bundle = _forecast_bundle()
    bundle.forecast_models.append(
        {
            "id": "unit_model",
            "label": "Unit Model",
            "points": [
                {"time": 1_700_086_400, "value": 81.0},
                {"time": 1_700_172_800, "value": 82.0},
            ],
        }
    )
    monkeypatch.setattr(report_route, "build_forecast", lambda **kwargs: bundle)
    client = TestClient(main.app)
    response = client.get("/api/report?symbol=CL=F&interval=1d&horizon=1")
    body = response.json()

    assert response.status_code == 200
    assert body["symbol"] == "CL=F"
    assert body["mode"] == "deterministic_report"
    assert body["llm_used"] is False
    assert "규칙 기반 문장 템플릿" in body["source_note"]
    assert "executive_summary" in body
    assert body["sections"][0]["title"] == "핵심 전망"
    assert "Unit Model" not in str(body)
    assert "모델 비교" not in str(body)
    assert "매매 지시" not in str(body)
    assert "데이터 상태" not in str(body)
    assert "밴드 상태" not in str(body)
    assert "컨텍스트 영향 점수" not in str(body)
    assert body["recommendation_note"] == ""
    assert "작성일" in body["key_metrics"]
    assert "예측기간" in body["key_metrics"]
    assert "마지막" not in str(body["key_metrics"])
    assert "# CL=F 1D 예측 리포트" in body["markdown"]

    english = client.get("/api/report?symbol=CL=F&interval=1d&horizon=1&language=en").json()
    assert english["mode"] == "deterministic_report"
    assert english["llm_used"] is False
    assert "not an external LLM fallback" in english["source_note"]
    assert english["sections"][0]["title"] == "Core View"
    assert "Model Comparison" not in str(english)
    assert "scenario in 1 day" in english["executive_summary"]
    assert english["recommendation_note"] == ""
    assert "forecast_period" in english["key_metrics"]


def test_dashboard_analysis_splits_panels_into_three_external_llm_calls(monkeypatch):
    bundle = _forecast_bundle()
    bundle.response.primary_model = "oil_context_fusion"
    bundle.forecast_models.append(
        {
            "id": "oil_context_fusion",
            "label": "Oil Context Fusion",
            "points": [
                {"time": 1_700_086_400, "value": 81.0},
                {"time": 1_700_172_800, "value": 82.0},
            ],
        }
    )
    captured = {"prompts": []}
    reserve_calls = {"count": 0}

    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: bundle)
    monkeypatch.setattr(
        market_context_route,
        "get_settings",
        lambda: Settings(enable_external_llm_calls=True, llm_api_key="unit-key", llm_context_mode="openai"),
    )

    def reserve_calls_for_panel(count=1):
        reserve_calls["count"] += count
        return True

    monkeypatch.setattr(market_context_route, "_reserve_llm_calls", reserve_calls_for_panel)

    def fake_live_context(**kwargs):
        assert kwargs["settings"].enable_external_llm_calls is False
        return {
            "news": pd.DataFrame(
                [
                    {
                        "published_at": "2026-06-07T00:00:00+00:00",
                        "symbol": "CL=F",
                        "headline": "OPEC plans fourth oil quota hike amid Hormuz closure tensions",
                        "source": "unit",
                        "url": "",
                    }
                ]
            ),
            "context_frame": pd.DataFrame(
                [
                    {
                        "timestamp": "2026-06-07T00:00:00+00:00",
                        "symbol": "CL=F",
                        "llm_mode": "local_rules",
                        "overall_bias": "bullish",
                        "impact_score": 0.7,
                        "uncertainty": 0.3,
                        "event_count": 1,
                        "explanation": "Deterministic local event context encoder produced structured context only.",
                        "warnings": "",
                    }
                ]
            ),
            "warnings": [],
            "source": "live_public_news",
        }

    monkeypatch.setattr(market_context_route, "build_live_event_context", fake_live_context)

    def fake_llm(_settings, prompt):
        captured["prompts"].append(prompt)
        if '"target_panel": "commentary"' in prompt:
            return {
                "summary": "공급 차질 우려가 현재 예측의 핵심 배경입니다.",
                "model_interpretation": "최근 뉴스와 차트 반등이 함께 상방 압력을 만들고 있습니다.",
                "risk_notes": ["OPEC 증산 뉴스가 강해지면 흐름이 약해질 수 있습니다."],
                "warnings": [],
            }
        if '"target_panel": "news_context"' in prompt:
            return {
                "summary": "뉴스는 호르무즈 긴장과 OPEC 공급 대응 사이의 균형을 보여줍니다.",
                "translated_news": [
                    {
                        "source_index": 0,
                        "headline": "호르무즈 긴장 속 OPEC 추가 증산 가능성",
                    }
                ],
                "context_points": [
                    {
                        "source_index": 0,
                        "overall_bias": "bullish",
                        "explanation": "공급 차질 우려가 증산 기대보다 더 크게 반영됐습니다.",
                    }
                ],
                "warnings": [],
            }
        if '"target_panel": "report"' in prompt:
            return {
                "title": "CL=F 1D 예측 리포트",
                "executive_summary": "현재 예측은 완만한 상방 흐름을 우선 반영합니다.",
                "sections": [
                    {
                        "title": "핵심 전망",
                        "body": "중앙 경로는 현재가 대비 소폭 상승을 가리킵니다.",
                        "bullets": ["숫자는 제공된 모델 출력만 사용했습니다."],
                    }
                ],
                "recommendation_note": "",
                "warnings": [],
            }
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(market_context_route, "_openai_compatible_model_commentary", fake_llm)
    market_context_route._DASHBOARD_ANALYSIS_CACHE.clear()
    market_context_route._MARKET_CONTEXT_CACHE.clear()
    client = TestClient(main.app)
    response = client.get("/api/dashboard-analysis?symbol=CL=F&interval=1d&horizon=1&language=ko")
    body = response.json()

    assert response.status_code == 200
    assert reserve_calls["count"] == 3
    assert len(captured["prompts"]) == 3
    assert body["mode"] == "llm_dashboard_analysis"
    assert body["llm_used"] is True
    assert body["commentary"]["mode"] == "llm_dashboard_commentary"
    assert body["market_context"]["scenario_commentary"]["summary"].startswith("뉴스는")
    assert body["market_context"]["news"][0]["headline"] == "호르무즈 긴장 속 OPEC 추가 증산 가능성"
    assert body["market_context"]["context_points"][0]["explanation"].startswith("공급 차질")
    assert "Deterministic local event" not in str(body)
    assert body["report"]["mode"] == "llm_dashboard_report"
    assert body["report"]["llm_used"] is True
    assert all("shared_voice" in prompt for prompt in captured["prompts"])
    assert all("말투, 용어, 기준 시점 표현을 일관되게 유지" in prompt for prompt in captured["prompts"])
    assert any('"target_panel": "commentary"' in prompt for prompt in captured["prompts"])
    assert any('"target_panel": "news_context"' in prompt for prompt in captured["prompts"])
    assert any('"target_panel": "report"' in prompt for prompt in captured["prompts"])
    assert any("sections는 정확히 4개" in prompt for prompt in captured["prompts"])
    assert any("model_interpretation은 3~5문장" in prompt for prompt in captured["prompts"])
    assert all("checkpoints" in prompt for prompt in captured["prompts"])
    assert any("source_index" in prompt for prompt in captured["prompts"])
    assert "target_price" not in str(body).lower()

    monkeypatch.setattr(market_context_route, "_forecast_bundle_for_commentary", lambda **kwargs: bundle)
    market_context_route._DASHBOARD_ANALYSIS_CACHE.clear()
    captured["prompts"] = []
    response = client.get(
        "/api/dashboard-analysis?symbol=CL=F&interval=1d&horizon=1&language=ko&origin_time=1700000000"
    )
    origin_body = response.json()
    assert response.status_code == 200
    assert reserve_calls["count"] == 6
    assert len(captured["prompts"]) == 3
    assert "기준가" in origin_body["report"]["key_metrics"]
    assert "현재 예측은" not in str(origin_body)
    assert all("\"analysis_mode\": \"point_in_time_backtest\"" in prompt for prompt in captured["prompts"])
    assert all("reference_time_label" in prompt for prompt in captured["prompts"])
    assert all("현재, 최근, 지금, 금일" in prompt for prompt in captured["prompts"])
    assert all("존댓말" in prompt for prompt in captured["prompts"])
    assert all("first sentence must explicitly include reference_time_label" in prompt for prompt in captured["prompts"])


def test_dashboard_llm_json_parser_ignores_trailing_text():
    parsed = market_context_route._extract_commentary_json('{"commentary": {"summary": "요약"}}\n\n추가 설명')

    assert parsed == {"commentary": {"summary": "요약"}}


def test_llm_provider_uses_configured_timeout(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"candidates":[{"content":{"parts":[{"text":"{\\"summary\\":\\"ok\\"}"}]}}]}'

    def fake_urlopen(_request, *, timeout, context):
        captured["timeout"] = timeout
        captured["context"] = context
        return FakeResponse()

    monkeypatch.setattr(market_context_route.urllib.request, "urlopen", fake_urlopen)
    settings = Settings(
        llm_api_key="unit-key",
        llm_context_mode="google_generative",
        llm_api_base="https://generativelanguage.googleapis.com/v1beta",
        llm_model="unit-model",
        llm_request_timeout_seconds=77,
    )

    assert market_context_route._google_model_commentary(settings, "prompt") == {"summary": "ok"}
    assert captured["timeout"] == 77
    assert captured["context"] is not None


def test_dashboard_report_missing_sections_is_not_rule_based_fallback():
    with pytest.raises(ValueError, match="missing sections"):
        market_context_route._dashboard_report_from_llm(
            bundle=_forecast_bundle(),
            raw_report={"executive_summary": "LLM summary only."},
            forecast_facts={"horizon": 1, "key_metrics": {}},
            generated_at=datetime.now(timezone.utc),
            language="ko",
        )


def test_live_market_context_does_not_silently_use_cached_news(monkeypatch):
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: _forecast_bundle())
    monkeypatch.setattr(market_context_route, "_news_items", lambda **kwargs: [{"time": 1_700_000_000, "headline": "cached old news", "symbol": "CL=F"}])
    monkeypatch.setattr(market_context_route, "_context_points", lambda **kwargs: [{"time": 1_700_000_000, "overall_bias": "bullish", "impact_score": 0.5, "event_count": 1}])

    def fail_live_context(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(market_context_route, "build_live_event_context", fail_live_context)
    market_context_route._MARKET_CONTEXT_CACHE.clear()
    client = TestClient(main.app)
    response = client.get("/api/market-context?symbol=CL=F&interval=1d&live=1")
    body = response.json()

    assert response.status_code == 200
    assert body["news"] == []
    assert body["context_points"] == []
    assert body["news_source"] == "live_public_news_unavailable"
    assert body["offline_cache_available"]["news_count"] == 1
    assert body["chart_context_points"][0]["news_items"][0]["headline"] == "cached old news"


def test_live_market_context_chart_markers_use_forecast_context_window(monkeypatch):
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: _forecast_bundle())
    cached_news = [
        {"time": 1_700_000_000 + idx * 86_400, "headline": f"forecast context news {idx}", "symbol": "CL=F"}
        for idx in range(24)
    ]
    monkeypatch.setattr(market_context_route, "_news_items", lambda **kwargs: cached_news)
    monkeypatch.setattr(
        market_context_route,
        "_context_points",
        lambda **kwargs: [
            {"time": item["time"], "overall_bias": "neutral", "impact_score": 0.4, "event_count": 1}
            for item in cached_news
        ],
    )

    def live_context(**kwargs):
        return {
            "news": pd.DataFrame(
                [
                    {
                        "published_at": datetime.fromtimestamp(1_700_900_000, tz=timezone.utc),
                        "symbol": "CL=F",
                        "headline": "live-only latest headline",
                        "source": "unit",
                    }
                ]
            ),
            "context_frame": pd.DataFrame(),
            "warnings": [],
            "source": "live_public_news",
        }

    monkeypatch.setattr(market_context_route, "build_live_event_context", live_context)
    market_context_route._MARKET_CONTEXT_CACHE.clear()
    client = TestClient(main.app)
    response = client.get("/api/market-context?symbol=CL=F&interval=1d&live=1&limit=3")
    body = response.json()

    assert response.status_code == 200
    assert body["news"][0]["headline"] == "live-only latest headline"
    marker_times = [point["time"] for point in body["chart_context_points"]]
    assert marker_times[0] == cached_news[0]["time"]
    assert marker_times[-1] == cached_news[-1]["time"]
    assert len(marker_times) == 6


def test_live_market_context_caches_live_encoder_payload(monkeypatch):
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: _forecast_bundle())
    monkeypatch.setattr(market_context_route, "_news_items", lambda **kwargs: [])
    monkeypatch.setattr(market_context_route, "_context_points", lambda **kwargs: [])
    calls = {"count": 0}

    def live_context(**kwargs):
        calls["count"] += 1
        return {
            "news": pd.DataFrame(
                [
                    {
                        "published_at": datetime.fromtimestamp(1_700_000_000, tz=timezone.utc),
                        "symbol": "CL=F",
                        "headline": "live oil headline",
                        "source": "unit",
                        "url": "https://example.com",
                    }
                ]
            ),
            "context_frame": pd.DataFrame(
                [
                    {
                        "timestamp": datetime.fromtimestamp(1_700_000_000, tz=timezone.utc),
                        "symbol": "CL=F",
                        "overall_bias": "bullish",
                        "impact_score": 0.8,
                        "uncertainty": 0.2,
                        "event_count": 1,
                        "explanation": "live context",
                        "warnings": "",
                    }
                ]
            ),
            "warnings": [],
            "source": "live_public_news",
        }

    monkeypatch.setattr(market_context_route, "build_live_event_context", live_context)
    market_context_route._MARKET_CONTEXT_CACHE.clear()
    client = TestClient(main.app)

    first = client.get("/api/market-context?symbol=CL=F&interval=1d&live=1")
    second = client.get("/api/market-context?symbol=CL=F&interval=1d&live=1")

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls["count"] == 1
    assert first.json()["news_source"] == "live_public_news"
    assert second.json()["news_source"] == "live_public_news_cached"


def test_model_commentary_endpoint_explains_model_outputs_without_price_forecasting(monkeypatch):
    bundle = _forecast_bundle()
    bundle.forecast_models.append(
        {
            "id": "unit_model",
            "label": "Unit Model",
            "points": [
                {"time": 1_700_086_400, "value": 81.0},
                {"time": 1_700_172_800, "value": 82.0},
            ],
        }
    )
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: bundle)
    captured = {}
    _enable_fake_model_commentary_llm(
        monkeypatch,
        payload={
            "summary": "현재 예측은 공급 뉴스와 최근 차트 흐름을 함께 반영해 상방 쪽으로 읽힙니다.",
            "model_interpretation": "차트 흐름과 뉴스 근거가 같은 방향으로 작용했습니다.",
            "risk_notes": "재고와 OPEC 관련 뉴스가 바뀌면 방향이 달라질 수 있습니다.",
            "warnings": "지정학적 리스크에 따른 변동성에 유의해야 합니다.",
        },
        captured=captured,
    )
    market_context_route._MODEL_COMMENTARY_CACHE.clear()
    client = TestClient(main.app)
    response = client.get("/api/model-commentary?symbol=CL=F&interval=1d")
    body = response.json()
    assert response.status_code == 200
    assert body["mode"] == "llm_model_commentary"
    assert body["llm_used"] is True
    assert body["model_summaries"][0]["id"] == "unit_model"
    assert "model_interpretation" in body
    assert body["risk_notes"] == ["재고와 OPEC 관련 뉴스가 바뀌면 방향이 달라질 수 있습니다."]
    assert body["warnings"] == ["지정학적 리스크에 따른 변동성에 유의해야 합니다."]
    assert "새로운 가격 목표" in captured["prompt"]
    assert "모델 간" not in str(body)
    assert "target_price" not in str(body).lower()


def test_model_commentary_prompt_tells_llm_not_to_quote_english_headlines(monkeypatch):
    bundle = _forecast_bundle()
    bundle.forecast_models.append(
        {
            "id": "unit_model",
            "label": "Unit Model",
            "points": [
                {"time": 1_700_086_400, "value": 81.0},
                {"time": 1_700_172_800, "value": 88.0},
            ],
        }
    )
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: bundle)
    monkeypatch.setattr(
        market_context_route,
        "_commentary_market_context",
        lambda *args, **kwargs: {
            "news": [
                {"headline": "Global Stocks Possibly Trading Below Their Estimated Value"},
                {"headline": "CTA Funds Thrive Amid Volatile Oil Prices"},
            ],
            "context_points": [],
        },
    )
    captured = {}
    _enable_fake_model_commentary_llm(monkeypatch, captured=captured)
    market_context_route._MODEL_COMMENTARY_CACHE.clear()
    client = TestClient(main.app)
    response = client.get("/api/model-commentary?symbol=CL=F&interval=1d&language=ko")
    body = response.json()

    assert response.status_code == 200
    assert body["llm_used"] is True
    assert "영어 뉴스 제목을 원문 그대로 인용하지 말고" in captured["prompt"]
    assert "Global Stocks Possibly Trading Below Their Estimated Value" in captured["prompt"]


def test_model_commentary_mentions_geopolitical_supply_shock_adapter(monkeypatch):
    bundle = _forecast_bundle()
    bundle.response.primary_model = "oil_context_fusion"
    bundle.response.deep_model_info = {
        "oil_context_fusion": {
            "path_adapter": {
                "adapter": "geopolitical_supply_shock",
                "geopolitical_supply_shock_score": 0.78,
            }
        }
    }
    bundle.forecast_models.append(
        {
            "id": "oil_context_fusion",
            "label": "Oil Context Fusion",
            "points": [
                {"time": 1_700_086_400, "value": 81.0},
                {"time": 1_700_172_800, "value": 88.0},
            ],
        }
    )
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: bundle)
    _enable_fake_model_commentary_llm(
        monkeypatch,
        payload={
            "summary": "현재 예측은 공급 리스크가 유가에 프리미엄을 더한 것으로 읽힙니다.",
            "model_interpretation": "지정학적 긴장과 공급 리스크가 차트 반등과 함께 반영됐습니다.",
            "risk_notes": ["공급 뉴스가 완화되면 방향이 달라질 수 있습니다."],
            "warnings": [],
        },
    )
    market_context_route._MODEL_COMMENTARY_CACHE.clear()
    client = TestClient(main.app)

    response = client.get("/api/model-commentary?symbol=CL=F&interval=1d&language=ko")
    body = response.json()

    assert response.status_code == 200
    assert body["llm_used"] is True
    assert body["model_summaries"][0]["path_adapter"]["adapter"] == "geopolitical_supply_shock"
    assert "공급 리스크" in body["model_interpretation"]
    assert "target_price" not in str(body).lower()


def test_model_commentary_language_query_changes_external_llm_prompt(monkeypatch):
    bundle = _forecast_bundle()
    bundle.forecast_models.append(
        {
            "id": "unit_model",
            "label": "Unit Model",
            "points": [
                {"time": 1_700_086_400, "value": 81.0},
                {"time": 1_700_172_800, "value": 82.0},
            ],
        }
    )
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: bundle)
    captured = {}
    _enable_fake_model_commentary_llm(
        monkeypatch,
        payload={
            "summary": "CL=F leans sideways in the current view.",
            "model_interpretation": "The chart and news context are mixed.",
            "risk_notes": ["Inventories can change the read."],
            "warnings": [],
        },
        captured=captured,
    )
    market_context_route._MODEL_COMMENTARY_CACHE.clear()
    client = TestClient(main.app)

    response = client.get("/api/model-commentary?symbol=CL=F&interval=1d&language=en")
    body = response.json()

    assert response.status_code == 200
    assert body["llm_used"] is True
    assert "WTI" in body["summary"] or "CL=F" in body["summary"]
    assert "market_context" in body
    assert "price_action" in body
    assert "English" in captured["prompt"]
    assert "기울기" not in body["summary"]


def test_model_commentary_requires_external_llm_without_fallback(monkeypatch):
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: _forecast_bundle())
    monkeypatch.setattr(market_context_route, "get_settings", lambda: Settings(enable_external_llm_calls=False))
    market_context_route._MODEL_COMMENTARY_CACHE.clear()
    client = TestClient(main.app)

    response = client.get("/api/model-commentary?symbol=CL=F&interval=1d&language=ko")
    body = response.json()

    assert response.status_code == 503
    assert "인공지능 해설가가 응답하지 않아요" in body["detail"]["message"]
    assert "deterministic_model_commentary" not in str(body)


def test_model_commentary_provider_error_does_not_fallback(monkeypatch):
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: _forecast_bundle())
    monkeypatch.setattr(
        market_context_route,
        "get_settings",
        lambda: Settings(enable_external_llm_calls=True, llm_api_key="unit-key", llm_context_mode="openai"),
    )
    monkeypatch.setattr(market_context_route, "_reserve_llm_call", lambda: True)

    def provider_error(*_args, **_kwargs):
        raise OSError("HTTP 429")

    monkeypatch.setattr(market_context_route, "_openai_compatible_model_commentary", provider_error)
    market_context_route._MODEL_COMMENTARY_CACHE.clear()
    client = TestClient(main.app)

    response = client.get("/api/model-commentary?symbol=CL=F&interval=1d&language=ko")
    body = response.json()

    assert response.status_code == 503
    assert "인공지능 해설가가 응답하지 않아요" in body["detail"]["message"]
    assert "HTTP 429" in body["detail"]["message"]
    assert "deterministic_model_commentary" not in str(body)


def test_assistant_chat_requires_external_llm_without_fallback(monkeypatch):
    bundle = _forecast_bundle()
    bundle.forecast_models.append(
        {
            "id": "unit_model",
            "label": "Unit Model",
            "points": [
                {"time": 1_700_086_400, "value": 81.0},
                {"time": 1_700_172_800, "value": 82.0},
            ],
        }
    )
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: bundle)
    monkeypatch.setattr(market_context_route, "get_settings", lambda: Settings(enable_external_llm_calls=False))
    monkeypatch.setattr(
        market_context_route,
        "_load_live_context_payload",
        lambda **kwargs: {
            "context_points": [{"overall_bias": "bullish", "impact_score": 0.7, "event_count": 2}],
            "news": [],
            "warnings": [],
        },
    )
    client = TestClient(main.app)

    response = client.post(
        "/api/assistant-chat",
        json={"question": "근거가 뭐야?", "symbol": "CL=F", "interval": "1d", "language": "ko"},
    )
    body = response.json()

    assert response.status_code == 503
    assert "인공지능 해설가가 응답하지 않아요" in body["detail"]["message"]
    assert "deterministic_assistant" not in str(body)
    assert "target_price" not in str(body).lower()


def test_assistant_chat_external_prompt_uses_public_context(monkeypatch):
    bundle = _forecast_bundle()
    bundle.response.primary_model = "oil_context_fusion"
    bundle.forecast_models.append(
        {
            "id": "oil_context_fusion",
            "label": "Oil Context Fusion",
            "points": [
                {"time": 1_700_086_400, "value": 81.0},
                {"time": 1_700_172_800, "value": 82.0},
            ],
        }
    )
    captured = {}

    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: bundle)
    monkeypatch.setattr(
        market_context_route,
        "get_settings",
        lambda: Settings(enable_external_llm_calls=True, llm_api_key="unit-key", llm_context_mode="openai"),
    )
    monkeypatch.setattr(
        market_context_route,
        "_load_live_context_payload",
        lambda **kwargs: {
            "context_points": [{"overall_bias": "bullish", "impact_score": 0.7, "event_count": 2}],
            "news": [{"headline": "OPEC supply cut", "source": "unit"}],
            "warnings": [],
        },
    )
    monkeypatch.setattr(market_context_route, "_reserve_llm_call", lambda: True)

    def fake_llm(_settings, prompt):
        captured["prompt"] = prompt
        return {
            "answer": "현재 화면은 공급 관련 뉴스와 최근 가격 흐름이 함께 반영되어 상방 쪽으로 읽힙니다.",
            "warnings": "본 답변은 매매 지시가 아닙니다.",
        }

    monkeypatch.setattr(market_context_route, "_openai_compatible_model_commentary", fake_llm)

    client = TestClient(main.app)
    response = client.post(
        "/api/assistant-chat",
        json={"question": "상승 근거가 뭐야?", "symbol": "CL=F", "interval": "1d", "language": "ko"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["mode"] == "llm_assistant"
    assert "oil_context_fusion" not in captured["prompt"]
    assert "model_summaries" not in captured["prompt"]
    assert "latest_context_points" not in captured["prompt"]
    assert "impact_score" not in captured["prompt"]
    assert "existing_forecast" in captured["prompt"]
    assert "chart_read" in captured["prompt"]
    assert "news_read" in captured["prompt"]
    assert "반드시 사용자의 질문에 직접 답하라" in captured["prompt"]
    assert "제목을 그대로 나열하지 말고" in captured["prompt"]
    assert body["answer"].startswith("현재 화면")
    assert body["warnings"] == ["본 답변은 매매 지시가 아닙니다."]


def test_assistant_chat_rejects_external_technical_answer(monkeypatch):
    bundle = _forecast_bundle()
    bundle.response.primary_model = "oil_context_fusion"
    bundle.forecast_models.append(
        {
            "id": "oil_context_fusion",
            "label": "Oil Context Fusion",
            "points": [
                {"time": 1_700_086_400, "value": 81.0},
                {"time": 1_700_172_800, "value": 82.0},
            ],
        }
    )
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: bundle)
    monkeypatch.setattr(
        market_context_route,
        "get_settings",
        lambda: Settings(enable_external_llm_calls=True, llm_api_key="unit-key", llm_context_mode="openai"),
    )
    monkeypatch.setattr(
        market_context_route,
        "_load_live_context_payload",
        lambda **kwargs: {
            "context_points": [{"overall_bias": "bullish", "impact_score": 0.7, "event_count": 2}],
            "news": [],
            "warnings": [],
        },
    )
    monkeypatch.setattr(market_context_route, "_reserve_llm_call", lambda: True)
    monkeypatch.setattr(
        market_context_route,
        "_openai_compatible_model_commentary",
        lambda *_args, **_kwargs: {
            "answer": "Oil Context Fusion has a model-calculated event score of 0.7.",
            "warnings": [],
        },
    )

    client = TestClient(main.app)
    response = client.post(
        "/api/assistant-chat",
        json={"question": "상승 근거가 뭐야?", "symbol": "CL=F", "interval": "1d", "language": "ko"},
    )
    body = response.json()

    assert response.status_code == 502
    assert "표시하지 않았습니다" in body["detail"]["message"]
    assert "deterministic_assistant" not in str(body)


def test_assistant_chat_provider_error_does_not_fallback(monkeypatch):
    bundle = _forecast_bundle()
    bundle.response.primary_model = "oil_context_fusion"
    bundle.forecast_models.append(
        {
            "id": "oil_context_fusion",
            "label": "Oil Context Fusion",
            "points": [
                {"time": 1_700_086_400, "value": 81.0},
                {"time": 1_700_172_800, "value": 82.0},
            ],
        }
    )
    calls = {"count": 0}
    monkeypatch.setattr(market_context_route, "build_forecast", lambda **kwargs: bundle)
    monkeypatch.setattr(
        market_context_route,
        "get_settings",
        lambda: Settings(enable_external_llm_calls=True, llm_api_key="unit-key", llm_context_mode="openai"),
    )
    monkeypatch.setattr(
        market_context_route,
        "_load_live_context_payload",
        lambda **kwargs: {"context_points": [], "news": [], "warnings": []},
    )
    monkeypatch.setattr(market_context_route, "_reserve_llm_call", lambda: True)

    def provider_error(*_args, **_kwargs):
        calls["count"] += 1
        raise OSError("HTTP 503 Service Unavailable")

    monkeypatch.setattr(market_context_route, "_openai_compatible_model_commentary", provider_error)

    client = TestClient(main.app)
    response = client.post(
        "/api/assistant-chat",
        json={"question": "뉴스 근거가 뭐야?", "symbol": "CL=F", "interval": "1d", "language": "ko"},
    )
    body = response.json()

    assert response.status_code == 503
    assert calls["count"] == 1
    assert "인공지능 해설가가 응답하지 않아요" in body["detail"]["message"]
    assert "HTTP 503" in body["detail"]["message"]
    assert "deterministic_assistant" not in str(body)


def test_backtests_endpoint_missing_is_graceful(tmp_path, monkeypatch):
    monkeypatch.setattr(backtests_route, "PROJECT_DIR", tmp_path)
    client = TestClient(main.app)
    response = client.get("/api/backtests?symbol=CL=F&interval=1d")
    assert response.status_code == 200
    assert response.json()["status"] == "missing"


def test_backtest_visualization_endpoint_returns_point_in_time_payload(monkeypatch):
    market = _market_window()

    def fake_build_forecast(**kwargs):
        point_in_time_market = kwargs["market_override"]
        assert [candle.time for candle in point_in_time_market.candles] == [market.candles[0].time]
        bundle = _forecast_bundle()
        response = bundle.response.model_copy(
            update={
                "current_price": point_in_time_market.candles[-1].close,
                "data_status": point_in_time_market.data_status,
                "candles": point_in_time_market.candles,
            }
        )
        return ForecastBundle(
            response=response,
            market_data=point_in_time_market,
            forecast_models=bundle.forecast_models,
            metrics=bundle.metrics,
            model_info={"training_cutoff": "2023-01-01T00:00:00+00:00"},
            horizon=1,
        )

    monkeypatch.setattr(backtests_route, "load_market_data_window", lambda *args, **kwargs: market)
    monkeypatch.setattr(backtests_route, "build_forecast", fake_build_forecast)
    client = TestClient(main.app)
    response = client.get(f"/api/backtests/visualization?symbol=CL=F&interval=1d&origin_time={market.candles[0].time}")
    body = response.json()
    assert response.status_code == 200
    assert body["mode"] == "backtest_visualization"
    assert body["origin_time"] == market.candles[0].time
    assert len(body["candles"]) == 1
    assert body["actual_future_candles"][0]["time"] == market.candles[1].time
    assert body["backtest"]["history_rows"] == 1
    assert body["metrics"]["metric_samples"] == 1
    assert "final_ape_pct" in body["metrics"]
    assert "shape_score" in body["metrics"]
    assert body["backtest"]["leakage_audit_status"] == "post_artifact_cutoff"
    assert body["backtest"]["is_post_training_cutoff"] is True
    assert body["backtest"]["model_training_cutoff"] == "2023-01-01T00:00:00+00:00"
    assert body["leakage_audit_status"] == "post_artifact_cutoff"
