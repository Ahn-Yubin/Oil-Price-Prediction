#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.data.manifests import entry_from_file, upsert_inventory_entries
from market_ai.data.providers.cftc_provider import cot_weekly_to_daily_point_in_time, fetch_cftc_csv, load_cftc_manual_csv
from market_ai.data.storage import DATA_ROOT, ensure_data_lake, write_table


CFTC_MANUAL_CSV_HINT = (
    "report_date/date plus open_interest, managed_money_long, managed_money_short, "
    "commercial_long, commercial_short; official CFTC column aliases are also accepted"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch or import CFTC COT positioning data.")
    parser.add_argument("--manual-csv", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--end", default="")
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


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    ensure_data_lake(data_root)
    if args.manual_csv:
        manual_path = _require_manual_csv(args.manual_csv, dataset_name="CFTC COT", column_hint=CFTC_MANUAL_CSV_HINT)
        weekly = load_cftc_manual_csv(manual_path)
        raw_path = data_root / "raw" / "cftc" / manual_path.name
        source = "manual_csv"
    elif args.url:
        frames = [fetch_cftc_csv(url) for url in _split(args.url)]
        weekly = pd.concat(frames, ignore_index=True).sort_values("report_date").drop_duplicates(subset=["report_date"], keep="last")
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
