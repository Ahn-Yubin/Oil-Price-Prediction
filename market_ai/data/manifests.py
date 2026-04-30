from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from market_ai.config import PROJECT_DIR
from market_ai.data.storage import project_relative, read_table


MANIFEST_PATH = PROJECT_DIR / "data" / "manifests" / "data_inventory.json"
LATEST_SNAPSHOT_PATH = PROJECT_DIR / "data" / "manifests" / "latest_snapshot.json"


class DatasetManifestEntry(BaseModel):
    dataset_name: str
    source: str
    path: str
    symbol_or_series: str | None = None
    frequency: str | None = None
    start: str | None = None
    end: str | None = None
    rows: int
    columns: list[str] = Field(default_factory=list)
    generated_at: str
    source_url_or_provider: str | None = None
    point_in_time_safe: bool = False
    notes: str | None = None


def manifest_schema() -> dict[str, Any]:
    return DatasetManifestEntry.model_json_schema()


def _timestamp_bounds(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    candidates = [
        "timestamp",
        "date",
        "report_date",
        "release_time",
        "as_of_time",
        "feature_available_at",
        "published_at",
        "retrieved_at",
    ]
    for col in candidates:
        if col not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[col], errors="coerce", utc=True).dropna()
        if not parsed.empty:
            return parsed.min().isoformat(), parsed.max().isoformat()
    return None, None


def _infer_frequency(frame: pd.DataFrame) -> str | None:
    for col in ("timestamp", "date", "report_date", "release_time", "as_of_time", "feature_available_at"):
        if col not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[col], errors="coerce", utc=True).dropna().sort_values()
        if len(parsed) < 3:
            continue
        deltas = parsed.diff().dropna().dt.total_seconds()
        median = float(deltas.median()) if not deltas.empty else 0.0
        if median <= 0:
            continue
        if median <= 60 * 60:
            return "intraday"
        if median <= 36 * 60 * 60:
            return "daily"
        if median <= 10 * 24 * 60 * 60:
            return "weekly"
        if median <= 40 * 24 * 60 * 60:
            return "monthly"
        return "irregular"
    return None


def _infer_symbol_or_series(frame: pd.DataFrame, fallback: str | None = None) -> str | None:
    values: list[str] = []
    for col in ("symbol", "series_id", "contract", "dataset_name"):
        if col not in frame.columns:
            continue
        unique = [str(v) for v in frame[col].dropna().astype(str).unique()[:8]]
        if unique:
            values.extend(unique)
    if values:
        return ",".join(dict.fromkeys(values))
    return fallback


def entry_from_file(
    path: str | Path,
    *,
    dataset_name: str | None = None,
    source: str | None = None,
    source_url_or_provider: str | None = None,
    point_in_time_safe: bool = False,
    notes: str | None = None,
) -> DatasetManifestEntry:
    resolved = Path(path)
    frame = read_table(resolved)
    start, end = _timestamp_bounds(frame)
    return DatasetManifestEntry(
        dataset_name=dataset_name or resolved.stem,
        source=source or (resolved.parts[-3] if len(resolved.parts) >= 3 else "unknown"),
        path=project_relative(resolved),
        symbol_or_series=_infer_symbol_or_series(frame),
        frequency=_infer_frequency(frame),
        start=start,
        end=end,
        rows=int(len(frame)),
        columns=[str(col) for col in frame.columns],
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_url_or_provider=source_url_or_provider,
        point_in_time_safe=bool(point_in_time_safe),
        notes=notes,
    )


def load_inventory(path: str | Path = MANIFEST_PATH) -> list[DatasetManifestEntry]:
    resolved = Path(path)
    if not resolved.exists():
        return []
    data = json.loads(resolved.read_text(encoding="utf-8"))
    rows = data.get("datasets", data if isinstance(data, list) else [])
    return [DatasetManifestEntry.model_validate(row) for row in rows]


def save_inventory(entries: list[DatasetManifestEntry], path: str | Path = MANIFEST_PATH) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema": "market_ai.data_inventory.v1",
        "datasets": [entry.model_dump() for entry in sorted(entries, key=lambda item: item.path)],
    }
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_inventory_entries(entries: list[DatasetManifestEntry], path: str | Path = MANIFEST_PATH) -> list[DatasetManifestEntry]:
    current = load_inventory(path)
    by_key = {(entry.dataset_name, entry.path): entry for entry in current}
    for entry in entries:
        by_key[(entry.dataset_name, entry.path)] = entry
    out = list(by_key.values())
    save_inventory(out, path)
    write_latest_snapshot(out)
    return out


def write_latest_snapshot(entries: list[DatasetManifestEntry], path: str | Path = LATEST_SNAPSHOT_PATH) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    latest_by_dataset: dict[str, DatasetManifestEntry] = {}
    for entry in entries:
        prev = latest_by_dataset.get(entry.dataset_name)
        if prev is None or (entry.end or "") >= (prev.end or ""):
            latest_by_dataset[entry.dataset_name] = entry
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {name: entry.model_dump() for name, entry in sorted(latest_by_dataset.items())},
    }
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
