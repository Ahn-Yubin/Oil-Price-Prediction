from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from market_ai.config import PROJECT_DIR


DATA_ROOT = PROJECT_DIR / "data"


DATA_LAKE_DIRS: tuple[Path, ...] = (
    DATA_ROOT / "raw" / "market",
    DATA_ROOT / "raw" / "eia",
    DATA_ROOT / "raw" / "cftc",
    DATA_ROOT / "raw" / "cme",
    DATA_ROOT / "raw" / "events",
    DATA_ROOT / "raw" / "news",
    DATA_ROOT / "interim" / "market",
    DATA_ROOT / "interim" / "fundamentals",
    DATA_ROOT / "interim" / "events",
    DATA_ROOT / "processed" / "market_panel",
    DATA_ROOT / "processed" / "oil_fundamentals",
    DATA_ROOT / "processed" / "event_context",
    DATA_ROOT / "features" / "deep_training",
    DATA_ROOT / "manifests",
)


@dataclass(frozen=True)
class WriteResult:
    requested_path: Path
    path: Path
    format: str
    fallback_used: bool = False


def ensure_data_lake(root: Path = DATA_ROOT) -> None:
    for path in DATA_LAKE_DIRS:
        if path.is_relative_to(DATA_ROOT):
            resolved = root / path.relative_to(DATA_ROOT)
        else:
            resolved = path
        resolved.mkdir(parents=True, exist_ok=True)


def project_relative(path: str | Path) -> str:
    resolved = Path(path)
    try:
        return str(resolved.relative_to(PROJECT_DIR))
    except ValueError:
        return str(resolved)


def resolve_data_path(path: str | Path, *, root: Path = PROJECT_DIR) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return root / value


def _can_write_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401

            return True
        except Exception:
            return False


def write_table(frame: pd.DataFrame, path: str | Path, *, index: bool = False) -> WriteResult:
    requested = Path(path)
    requested.parent.mkdir(parents=True, exist_ok=True)
    suffix = requested.suffix.lower()
    if suffix == ".parquet":
        if _can_write_parquet():
            frame.to_parquet(requested, index=index)
            return WriteResult(requested_path=requested, path=requested, format="parquet")
        fallback = requested.with_suffix(".csv")
        frame.to_csv(fallback, index=index)
        return WriteResult(requested_path=requested, path=fallback, format="csv", fallback_used=True)
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            frame.to_json(requested, orient="records", lines=True, force_ascii=False)
        else:
            frame.to_json(requested, orient="records", force_ascii=False, indent=2)
        return WriteResult(requested_path=requested, path=requested, format=suffix.lstrip("."))
    frame.to_csv(requested, index=index)
    return WriteResult(requested_path=requested, path=requested, format="csv")


def read_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists() and resolved.suffix.lower() == ".parquet":
        fallback = resolved.with_suffix(".csv")
        if fallback.exists():
            resolved = fallback
    suffix = resolved.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(resolved)
    if suffix == ".jsonl":
        return pd.read_json(resolved, lines=True)
    if suffix == ".json":
        return pd.read_json(resolved)
    return pd.read_csv(resolved)


def safe_symbol(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("=", "_")
        .replace("^", "_")
        .replace(" ", "_")
    )


def touch_gitkeep(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        marker = path / ".gitkeep"
        if not marker.exists():
            marker.write_text("", encoding="utf-8")
