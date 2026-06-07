from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, Field, field_validator

from market_ai.env import load_project_env


PROJECT_DIR = Path(__file__).resolve().parent.parent


def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _path_from_env(value: str | None, default: Path) -> Path:
    if value is None or not value.strip():
        return default
    return Path(value).expanduser()


class Settings(BaseModel):
    app_env: str = Field(default="development")
    default_symbol: str = Field(default="CL=F")
    default_interval: str = Field(default="1d")
    model_dir: Path = Field(default_factory=lambda: PROJECT_DIR / "artifacts" / "models")
    metadata_dir: Path = Field(default_factory=lambda: PROJECT_DIR / "artifacts" / "metadata")
    data_dir: Path = Field(default_factory=lambda: PROJECT_DIR / "data")
    baseline_predictions_path: Path = Field(default_factory=lambda: PROJECT_DIR / "data" / "predictions.csv")
    baseline_ohlc_path: Path = Field(default_factory=lambda: PROJECT_DIR / "data" / "ohlc.csv")
    allow_mock_data: bool = Field(default=False)
    data_stale_threshold_seconds: int = Field(default=86_400)
    enable_llm_context: bool = Field(default=False)
    enable_external_llm_calls: bool = Field(default=False)
    llm_api_key: str | None = Field(default=None)
    llm_model: str = Field(default="context-encoder-placeholder")
    llm_api_base: str = Field(default="https://api.openai.com/v1/chat/completions")
    local_llm_api_base: str = Field(default="http://localhost:11434/api/chat")
    local_llm_model: str = Field(default="local-context-encoder")
    llm_context_mode: str = Field(default="google_generative")
    enable_external_features: bool = Field(default=False)
    enable_cross_asset_features: bool = Field(default=False)
    enable_online_residual_calibration: bool = Field(default=False)
    app_version: str = Field(default="0.2.0")

    @field_validator("app_env")
    @classmethod
    def normalize_app_env(cls, value: str) -> str:
        normalized = (value or "development").strip().lower()
        return normalized or "development"

    @field_validator("data_stale_threshold_seconds")
    @classmethod
    def positive_stale_threshold(cls, value: int) -> int:
        return max(1, int(value))

    @property
    def is_development(self) -> bool:
        return self.app_env in {"development", "dev", "local", "test", "testing"}

    @property
    def mock_data_enabled(self) -> bool:
        return self.is_development or self.allow_mock_data

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        if env is None:
            load_project_env()
        source = env if env is not None else os.environ
        data_dir = _path_from_env(source.get("DATA_DIR"), PROJECT_DIR / "data")
        return cls(
            app_env=source.get("APP_ENV", "development"),
            default_symbol=source.get("DEFAULT_SYMBOL", "CL=F"),
            default_interval=source.get("DEFAULT_INTERVAL", "1d"),
            model_dir=_path_from_env(source.get("MODEL_DIR"), PROJECT_DIR / "artifacts" / "models"),
            metadata_dir=_path_from_env(source.get("METADATA_DIR"), PROJECT_DIR / "artifacts" / "metadata"),
            data_dir=data_dir,
            baseline_predictions_path=_path_from_env(
                source.get("BASELINE_PREDICTIONS_PATH"),
                data_dir / "predictions.csv",
            ),
            baseline_ohlc_path=_path_from_env(
                source.get("BASELINE_OHLC_PATH"),
                data_dir / "ohlc.csv",
            ),
            allow_mock_data=_parse_bool(source.get("ALLOW_MOCK_DATA"), False),
            data_stale_threshold_seconds=int(source.get("DATA_STALE_THRESHOLD_SECONDS", "86400")),
            enable_llm_context=_parse_bool(source.get("ENABLE_LLM_CONTEXT"), False),
            enable_external_llm_calls=_parse_bool(source.get("ENABLE_EXTERNAL_LLM_CALLS"), False),
            llm_api_key=source.get("LLM_API_KEY") or None,
            llm_model=source.get("LLM_MODEL", "context-encoder-placeholder"),
            llm_api_base=source.get("LLM_API_BASE", "https://api.openai.com/v1/chat/completions"),
            local_llm_api_base=source.get("LOCAL_LLM_API_BASE", "http://localhost:11434/api/chat"),
            local_llm_model=source.get("LOCAL_LLM_MODEL", "local-context-encoder"),
            llm_context_mode=source.get("LLM_CONTEXT_MODE", "google_generative"),
            enable_external_features=_parse_bool(source.get("ENABLE_EXTERNAL_FEATURES"), False),
            enable_cross_asset_features=_parse_bool(source.get("ENABLE_CROSS_ASSET_FEATURES"), False),
            enable_online_residual_calibration=_parse_bool(source.get("ENABLE_ONLINE_RESIDUAL_CALIBRATION"), False),
            app_version=source.get("APP_VERSION", "0.2.0"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
