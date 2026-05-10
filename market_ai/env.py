from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent


def _parse_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_project_env(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load a simple KEY=VALUE .env file into os.environ."""

    env_path = Path(path).expanduser() if path else PROJECT_DIR / ".env"
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        if not override and key in os.environ:
            continue
        value = _parse_env_value(raw_value)
        os.environ[key] = value
        loaded[key] = value
    return loaded
