from fastapi.testclient import TestClient

import backend.app.main as main
from backend.app.api.routes import forecast as forecast_route
from market_ai.modeling.forecasters.deep_fusion import DeepModelUnavailable
from tests.integration.test_forecast_models_query import _comparison, _market_window, _missing_deep_availability


def test_api_forecast_deep_model_request_gracefully_falls_back(monkeypatch):
    monkeypatch.setattr("market_ai.forecasting.service.load_market_data_window", lambda *args, **kwargs: _market_window())
    monkeypatch.setattr("market_ai.forecasting.service.forecast_model_comparison", _comparison)
    monkeypatch.setattr("market_ai.forecasting.service._deep_availability_by_model", _missing_deep_availability)
    monkeypatch.setattr(
        "market_ai.forecasting.service.forecast_with_deep_model",
        lambda **kwargs: (_ for _ in ()).throw(DeepModelUnavailable("missing test artifact")),
    )
    client = TestClient(main.app)
    response = client.get("/api/forecast?symbol=CL=F&interval=1d&models=deep_lstm_tcn_fusion")
    assert response.status_code == 200
    body = response.json()
    assert "deep_lstm_tcn_fusion" in body["selected_models"]
    assert body["artifact_status"]["deep_lstm_tcn_fusion"] == "artifact_missing"
    assert any(item["code"] == "deep_artifact_unavailable" for item in body["warning_objects"])
    assert body["forecast"]


def test_api_forecast_unknown_model_is_400():
    client = TestClient(main.app)
    response = client.get("/api/forecast?symbol=CL=F&interval=1d&models=unknown_model")
    assert response.status_code == 400
    assert "unknown_model" in response.text
