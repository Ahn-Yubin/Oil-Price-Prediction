from __future__ import annotations

import os
from pathlib import Path

from market_ai.env import load_project_env


def test_load_project_env_reads_simple_values(tmp_path: Path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "ENABLE_LLM_CONTEXT=true",
                "export LLM_MODEL=\"gemma-3-27b-it\"",
                "LLM_API_BASE=https://example.test/chat/completions",
            ]
        ),
        encoding="utf-8",
    )
    for key in ["ENABLE_LLM_CONTEXT", "LLM_MODEL", "LLM_API_BASE"]:
        monkeypatch.delenv(key, raising=False)

    loaded = load_project_env(env_path)

    assert loaded["ENABLE_LLM_CONTEXT"] == "true"
    assert os.environ["LLM_MODEL"] == "gemma-3-27b-it"
    assert os.environ["LLM_API_BASE"] == "https://example.test/chat/completions"


def test_load_project_env_does_not_override_existing_value(tmp_path: Path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_MODEL=from_file\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "from_shell")

    loaded = load_project_env(env_path)

    assert "LLM_MODEL" not in loaded
    assert os.environ["LLM_MODEL"] == "from_shell"
