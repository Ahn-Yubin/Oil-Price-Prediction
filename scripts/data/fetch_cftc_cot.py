#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.data.manifests import entry_from_file, upsert_inventory_entries
from market_ai.data.providers.cftc_provider import cot_weekly_to_daily_point_in_time, fetch_cftc_csv, load_cftc_manual_csv
from market_ai.data.storage import DATA_ROOT, ensure_data_lake, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch or import CFTC COT positioning data.")
    parser.add_argument("--manual-csv", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--end", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    ensure_data_lake(data_root)
    if args.manual_csv:
        weekly = load_cftc_manual_csv(args.manual_csv)
        raw_path = data_root / "raw" / "cftc" / Path(args.manual_csv).name
        source = "manual_csv"
    elif args.url:
        weekly = fetch_cftc_csv(args.url)
        raw_path = data_root / "raw" / "cftc" / "cftc_cot_weekly.csv"
        source = args.url
    else:
        raise SystemExit("CFTC ingest requires --manual-csv or --url; no synthetic fallback is used.")
    write_table(weekly, raw_path)
    processed = cot_weekly_to_daily_point_in_time(weekly, end=args.end or None)
    processed_path = data_root / "processed" / "oil_fundamentals" / "cftc_cot_weekly.csv"
    write_table(processed, processed_path)
    upsert_inventory_entries(
        [
            entry_from_file(raw_path, dataset_name="raw_cftc_cot_weekly", source="cftc", source_url_or_provider=source, point_in_time_safe=True),
            entry_from_file(processed_path, dataset_name="processed_cftc_cot_daily_pit", source="cftc", source_url_or_provider=source, point_in_time_safe=True),
        ]
    )
    print(f"Wrote CFTC processed rows={len(processed)} to {processed_path}")


if __name__ == "__main__":
    main()
