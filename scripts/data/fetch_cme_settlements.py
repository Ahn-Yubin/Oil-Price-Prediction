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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch or import licensed/manual CME settlements.")
    parser.add_argument("--manual-csv", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    ensure_data_lake(data_root)
    if args.manual_csv:
        processed = load_cme_manual_csv(args.manual_csv)
        raw_path = data_root / "raw" / "cme" / Path(args.manual_csv).name
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
