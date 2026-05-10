from datetime import datetime, timezone

import pytest

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


def test_backtests_endpoint_missing_is_graceful(tmp_path, monkeypatch):
    monkeypatch.setattr(backtests_route, "PROJECT_DIR", tmp_path)
    client = TestClient(main.app)
    response = client.get("/api/backtests?symbol=CL=F&interval=1d")
    assert response.status_code == 200
    assert response.json()["status"] == "missing"
