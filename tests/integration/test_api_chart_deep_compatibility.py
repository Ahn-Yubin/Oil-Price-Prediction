from fastapi.testclient import TestClient

import backend.app.main as main
from tests.integration.test_forecast_models_query import _comparison, _market_window


def test_api_chart_deep_request_keeps_legacy_schema(monkeypatch):
    monkeypatch.setattr("market_ai.forecasting.service.load_market_data_window", lambda *args, **kwargs: _market_window())
    monkeypatch.setattr("market_ai.forecasting.service.forecast_model_comparison", _comparison)
    client = TestClient(main.app)
    response = client.get("/api/chart?symbol=CL=F&interval=1d&models=deep_lstm_tcn_fusion")
    assert response.status_code == 200
    body = response.json()
    for key in ["candles", "predicted", "predicted_lower", "predicted_upper", "forecast_models", "warnings"]:
        assert key in body
    assert "artifact_status" in body
