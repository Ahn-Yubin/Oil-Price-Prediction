#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime, timezone

from market_ai.data.manifests import (
    MANIFEST_PATH,
    DatasetManifestEntry,
    entry_from_file,
    manifest_schema,
    save_inventory,
    write_latest_snapshot,
)
from market_ai.data.storage import DATA_ROOT, ensure_data_lake


DEFAULT_PATTERNS = ("*.csv", "*.json", "*.jsonl", "*.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a point-in-time data inventory manifest.")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--output", default=str(MANIFEST_PATH))
    parser.add_argument("--schema", action="store_true", help="Print the JSON schema and exit.")
    return parser.parse_args()


def _source_from_path(path: Path, data_root: Path) -> str:
    try:
        rel = path.relative_to(data_root)
    except ValueError:
        return "unknown"
    parts = rel.parts
    if len(parts) >= 2:
        return parts[1] if parts[0] in {"raw", "interim", "processed", "features"} else parts[0]
    return parts[0] if parts else "unknown"


def _dataset_name(path: Path, data_root: Path) -> str:
    try:
        rel = path.relative_to(data_root).with_suffix("")
        return "_".join(rel.parts)
    except ValueError:
        return path.stem


def main() -> None:
    args = parse_args()
    if args.schema:
        import json

        print(json.dumps(manifest_schema(), ensure_ascii=False, indent=2))
        return
    data_root = Path(args.data_root)
    ensure_data_lake(data_root)
    entries = []
    for pattern in DEFAULT_PATTERNS:
        for path in sorted(data_root.rglob(pattern)):
            if "manifests" in path.parts or path.name.startswith("."):
                continue
            try:
                entries.append(
                    entry_from_file(
                        path,
                        dataset_name=_dataset_name(path, data_root),
                        source=_source_from_path(path, data_root),
                        point_in_time_safe="processed" in path.parts or "features" in path.parts,
                    )
                )
            except Exception as exc:
                entries.append(
                    DatasetManifestEntry(
                        dataset_name=_dataset_name(path, data_root),
                        source=_source_from_path(path, data_root),
                        path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                        rows=0,
                        columns=[],
                        generated_at=datetime.now(timezone.utc).isoformat(),
                        point_in_time_safe=False,
                        notes=f"Inventory read warning: {exc}",
                    )
                )
    save_inventory(entries, args.output)
    write_latest_snapshot(entries)
    print(f"Wrote {len(entries)} dataset entries to {args.output}")


if __name__ == "__main__":
    main()
