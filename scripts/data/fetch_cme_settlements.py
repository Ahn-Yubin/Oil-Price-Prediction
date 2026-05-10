#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.data.manifests import entry_from_file, upsert_inventory_entries
from market_ai.data.providers.cme_provider import fetch_cme_csv, load_cme_manual_csv
from market_ai.data.storage import DATA_ROOT, ensure_data_lake, write_table


CME_MANUAL_CSV_HINT = (
    "trade_date/date plus settle/settlement; contract/contract_month is recommended "
    "so the curve can build m1/m2/m3/m6 spreads"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch or import licensed/manual CME settlements.")
    parser.add_argument("--manual-csv", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    return parser.parse_args()


def _require_manual_csv(path: str, *, dataset_name: str, column_hint: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise SystemExit(
            f"{dataset_name} manual CSV not found: {path}\n"
            "Replace the documentation placeholder with a real local CSV path. "
            f"Expected columns: {column_hint}."
        )
    if not resolved.is_file():
        raise SystemExit(f"{dataset_name} manual CSV path is not a file: {path}")
    return resolved


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    ensure_data_lake(data_root)
    if args.manual_csv:
        manual_path = _require_manual_csv(args.manual_csv, dataset_name="CME settlements", column_hint=CME_MANUAL_CSV_HINT)
        processed = load_cme_manual_csv(manual_path)
        raw_path = data_root / "raw" / "cme" / manual_path.name
        source = "manual_csv"
    elif args.url:
        processed = fetch_cme_csv(args.url)
        raw_path = data_root / "raw" / "cme" / "cme_settlements.csv"
        source = args.url
    else:
        raise SystemExit("CME ingest requires --manual-csv or licensed --url; no fake scraping or synthetic fallback is used.")
    write_table(processed, raw_path)
    processed_path = data_root / "processed" / "oil_fundamentals" / "cme_curve_daily.csv"
    write_table(processed, processed_path)
    upsert_inventory_entries(
        [
            entry_from_file(raw_path, dataset_name="raw_cme_settlements", source="cme", source_url_or_provider=source, point_in_time_safe=True),
            entry_from_file(processed_path, dataset_name="processed_cme_curve_daily", source="cme", source_url_or_provider=source, point_in_time_safe=True),
        ]
    )
    print(f"Wrote CME curve rows={len(processed)} to {processed_path}")


if __name__ == "__main__":
    main()
