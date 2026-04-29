from tests.integration.test_api import _forecast_bundle

from fastapi.testclient import TestClient

import backend.app.main as main
from backend.app.api.routes import chart as chart_route


def test_chart_accepts_models_query_for_legacy_clients(monkeypatch):
    monkeypatch.setattr(chart_route, "build_forecast", lambda **kwargs: _forecast_bundle())
    client = TestClient(main.app)
    response = client.get("/api/chart?symbol=CL=F&interval=1d&models=cycle")
    assert response.status_code == 200
    body = response.json()
    assert "predicted" in body
    assert "forecast_models" in body
