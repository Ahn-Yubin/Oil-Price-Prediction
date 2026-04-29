from pathlib import Path

from market_ai.config import PROJECT_DIR, Settings


def test_settings_defaults_are_portable():
    settings = Settings.from_env({})
    assert settings.app_env == "development"
    assert settings.default_symbol == "NYMEX:CL1!"
    assert settings.default_interval == "1d"
    assert settings.model_dir == PROJECT_DIR / "artifacts" / "models"
    assert settings.metadata_dir == PROJECT_DIR / "artifacts" / "metadata"
    assert settings.mock_data_enabled is True


def test_settings_env_override(tmp_path: Path):
    settings = Settings.from_env(
        {
            "APP_ENV": "production",
            "ALLOW_MOCK_DATA": "false",
            "MODEL_DIR": str(tmp_path / "models"),
            "DATA_DIR": str(tmp_path / "data"),
            "DEFAULT_SYMBOL": "BTC-USD",
            "DEFAULT_INTERVAL": "1h",
            "DATA_STALE_THRESHOLD_SECONDS": "60",
            "ENABLE_LLM_CONTEXT": "true",
        }
    )
    assert settings.app_env == "production"
    assert settings.mock_data_enabled is False
    assert settings.model_dir == tmp_path / "models"
    assert settings.baseline_predictions_path == tmp_path / "data" / "predictions.csv"
    assert settings.default_symbol == "BTC-USD"
    assert settings.default_interval == "1h"
    assert settings.data_stale_threshold_seconds == 60
    assert settings.enable_llm_context is True
