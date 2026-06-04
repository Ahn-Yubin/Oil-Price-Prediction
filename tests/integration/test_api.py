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
            provider_period="2y",
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


def test_models_endpoint_uses_nearest_1d_artifact_horizon(monkeypatch):
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

    assert ("oil_context_fusion", "1d", 8) in seen
    assert ("oil_context_fusion", "1d", 14) in seen
    assert body_7["deep_artifact_policy"]["artifact_horizon"] == 8
    assert body_7["deep_artifact_policy"]["display_horizon"] == 7
    assert body_14["deep_artifact_policy"]["artifact_horizon"] == 14
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
    assert "scenario_commentary" in body


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
    assert "executive_summary" in body
    assert body["sections"][0]["title"] == "예측 경로"
    assert "Unit Model" not in str(body)
    assert "모델 비교" not in str(body)
    assert body["recommendation_note"] == ""
    assert "작성일" in body["key_metrics"]
    assert "예측기간" in body["key_metrics"]
    assert "마지막" not in str(body["key_metrics"])
    assert "# CL=F 1D 예측 리포트" in body["markdown"]

    english = client.get("/api/report?symbol=CL=F&interval=1d&horizon=1&language=en").json()
    assert english["sections"][0]["title"] == "Forecast Path"
    assert "Model Comparison" not in str(english)
    assert "median path in 1 day" in english["executive_summary"]
    assert english["recommendation_note"] == ""
    assert "forecast_period" in english["key_metrics"]


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
    monkeypatch.setattr(market_context_route, "get_settings", lambda: Settings(enable_external_llm_calls=False))
    market_context_route._MODEL_COMMENTARY_CACHE.clear()
    client = TestClient(main.app)
    response = client.get("/api/model-commentary?symbol=CL=F&interval=1d")
    body = response.json()
    assert response.status_code == 200
    assert body["mode"] == "deterministic_model_commentary"
    assert body["model_summaries"][0]["id"] == "unit_model"
    assert "model_interpretation" in body
    assert "모델 간" not in str(body)
    assert "target_price" not in str(body).lower()


def test_model_commentary_language_query_changes_deterministic_text(monkeypatch):
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
    market_context_route._MODEL_COMMENTARY_CACHE.clear()
    client = TestClient(main.app)

    response = client.get("/api/model-commentary?symbol=CL=F&interval=1d&language=en")
    body = response.json()

    assert response.status_code == 200
    assert "WTI" in body["summary"] or "CL=F" in body["summary"]
    assert "market_context" in body
    assert "price_action" in body
    assert "기울기" not in body["summary"]


def test_assistant_chat_uses_deterministic_fallback_without_price_target(monkeypatch):
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

    assert response.status_code == 200
    assert body["mode"] == "deterministic_assistant"
    assert "단일 운영 모델" in body["answer"]
    assert "target_price" not in str(body).lower()


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
            model_info=bundle.model_info,
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
