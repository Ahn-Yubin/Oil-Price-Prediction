from __future__ import annotations

from pathlib import Path

from market_ai.config import PROJECT_DIR


DEFAULT_UNIVERSE_PATH = PROJECT_DIR / "configs" / "symbol_universe.yaml"


def load_symbol_universe(path: Path | None = None) -> dict[str, list[str]]:
    source = path or DEFAULT_UNIVERSE_PATH
    if not source.exists():
        return {}
    universes: dict[str, list[str]] = {}
    current: str | None = None
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current = line[:-1].strip()
            universes[current] = []
            continue
        if current and line.strip().startswith("- "):
            universes[current].append(line.strip()[2:].strip())
    return universes


def resolve_universe(name: str, path: Path | None = None) -> list[str]:
    universes = load_symbol_universe(path)
    if name not in universes:
        raise KeyError(f"Unknown symbol universe: {name}")
    return universes[name]
