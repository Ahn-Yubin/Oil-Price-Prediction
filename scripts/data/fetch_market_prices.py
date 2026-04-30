#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.config import PROJECT_DIR
from market_ai.data.manifests import entry_from_file, upsert_inventory_entries
from market_ai.data.market_panel import build_missing_bars_report, save_market_panel
from market_ai.data.providers.market_price_provider import (
    CsvMarketPriceProvider,
    YFinanceMarketPriceProvider,
    combine_market_frames,
    write_market_cache,
)
from market_ai.data.storage import DATA_ROOT, ensure_data_lake, write_table
from market_ai.data.symbol_universe import load_symbol_universe


UNIVERSE_PATH = PROJECT_DIR / "configs" / "symbol_universe.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch reproducible market price panels.")
    parser.add_argument("--universe", default="oil_core")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--interval", choices=["1d", "1h", "30m", "15m"], default="1d")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--provider", choices=["yfinance", "csv"], default="yfinance")
    parser.add_argument("--csv-cache", default="")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    return parser.parse_args()


def resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [item.strip() for item in args.symbols.split(",") if item.strip()]
    universe = load_symbol_universe(UNIVERSE_PATH)
    return universe.get(args.universe, universe.get("oil_core", ["CL=F"]))


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    ensure_data_lake(data_root)
    symbols = resolve_symbols(args)
    provider = CsvMarketPriceProvider(args.csv_cache) if args.provider == "csv" else YFinanceMarketPriceProvider()
    frames = []
    failures = []
    raw_paths = []
    for symbol in symbols:
        try:
            frame = provider.fetch(symbol, interval=args.interval, period=args.period)
            path = write_market_cache(frame, root=data_root, provider=provider.provider_name, interval=args.interval, symbol=symbol)
            raw_paths.append(path)
            frames.append(frame)
            print(f"{symbol}: wrote {len(frame)} rows to {path}")
        except Exception as exc:
            failures.append({"symbol": symbol, "interval": args.interval, "provider": provider.provider_name, "error": str(exc)})
            print(f"{symbol}: fetch failed: {exc}", file=sys.stderr)
    if not frames:
        raise SystemExit("No market price data was fetched; refusing to create synthetic fallback.")
    panel = combine_market_frames(frames)
    panel_path = save_market_panel(panel, interval=args.interval, data_root=data_root)
    missing = build_missing_bars_report(panel, interval=args.interval)
    missing_path = data_root / "interim" / "market" / f"missing_bars_{args.universe}_{args.interval}.csv"
    write_table(missing, missing_path)
    if failures:
        write_table(
            pd.DataFrame(failures),
            data_root / "interim" / "market" / f"fetch_failures_{args.universe}_{args.interval}.csv",
        )
    entries = [
        entry_from_file(panel_path, dataset_name=f"processed_market_panel_{args.interval}", source="market", source_url_or_provider=provider.provider_name, point_in_time_safe=True),
        entry_from_file(missing_path, dataset_name=f"missing_bars_{args.universe}_{args.interval}", source="market", point_in_time_safe=True),
    ]
    for path in raw_paths:
        entries.append(entry_from_file(path, dataset_name=f"raw_market_{path.stem}_{args.interval}", source="market", source_url_or_provider=provider.provider_name, point_in_time_safe=True))
    upsert_inventory_entries(entries)
    print(f"Panel rows={len(panel)} path={panel_path}")
    print(f"Missing bar report rows={len(missing)} path={missing_path}")


if __name__ == "__main__":
    main()
