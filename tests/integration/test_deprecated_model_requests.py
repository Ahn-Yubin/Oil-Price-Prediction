from fastapi.testclient import TestClient

import backend.app.main as main


def test_removed_model_request_returns_400():
    client = TestClient(main.app)
    response = client.get("/api/forecast?symbol=CL=F&interval=1d&models=lstm")
    assert response.status_code == 400
    body = response.json()["detail"]
    assert body["removed_models"] == ["lstm"]
    assert "oil_context_fusion" in body["replacement_models"]["lstm"]
