#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.data.manifests import entry_from_file, upsert_inventory_entries
from market_ai.data.providers.eia_provider import EIAFetchConfig, fetch_eia_series, load_eia_manual_csv, weekly_to_daily_point_in_time
from market_ai.data.storage import DATA_ROOT, ensure_data_lake, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch or import EIA petroleum fundamentals.")
    parser.add_argument("--manual-csv", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--end", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    ensure_data_lake(data_root)
    if args.manual_csv:
        weekly = load_eia_manual_csv(args.manual_csv)
        raw_path = data_root / "raw" / "eia" / Path(args.manual_csv).name
        write_table(weekly, raw_path)
        source = "manual_csv"
    else:
        try:
            weekly = fetch_eia_series(EIAFetchConfig(api_key=args.api_key or None))
        except Exception as exc:
            raise SystemExit(f"EIA ingest failed: {exc}. Provide --manual-csv or EIA_API_KEY.") from exc
        raw_path = data_root / "raw" / "eia" / "eia_petroleum_weekly.csv"
        write_table(weekly, raw_path)
        source = "eia_api"
    processed = weekly_to_daily_point_in_time(weekly, end=args.end or None)
    processed_path = data_root / "processed" / "oil_fundamentals" / "eia_weekly.csv"
    write_table(processed, processed_path)
    upsert_inventory_entries(
        [
            entry_from_file(raw_path, dataset_name="raw_eia_petroleum_weekly", source="eia", source_url_or_provider=source, point_in_time_safe=True),
            entry_from_file(processed_path, dataset_name="processed_eia_petroleum_daily_pit", source="eia", source_url_or_provider=source, point_in_time_safe=True),
        ]
    )
    print(f"Wrote EIA processed rows={len(processed)} to {processed_path}")


if __name__ == "__main__":
    main()
