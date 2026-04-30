from fastapi.testclient import TestClient

import backend.app.main as main


def test_api_models_after_cleanup_excludes_removed_from_default_list():
    client = TestClient(main.app)
    body = client.get("/api/models").json()
    assert {"cycle", "lstm", "tcn", "ensemble"}.isdisjoint(body["logical_models"])
    assert {"deep_lstm_tcn_fusion", "llm_context_seq_moe"}.issubset(set(body["logical_models"]))
    deep_models = {item["id"]: item for item in body["user_facing_models"]}
    assert deep_models["deep_lstm_tcn_fusion"]["status"] in {"available", "artifact_missing"}
    assert deep_models["deep_lstm_tcn_fusion"]["expected_artifact_file"].endswith("_1d_h45.pt")
    assert "scripts/train/train_deep_fusion_models.py" in deep_models["deep_lstm_tcn_fusion"]["training_command"]
    removed_ids = {item["id"] for item in body["removed_models"]}
    assert {"cycle", "lstm", "tcn", "ensemble"}.issubset(removed_ids)
