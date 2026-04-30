from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_ai.data.providers.market_price_provider import combine_market_frames, load_market_cache, missing_bars_report
from market_ai.data.storage import DATA_ROOT, read_table, safe_symbol, write_table


def build_market_panel_from_files(paths: list[str | Path]) -> pd.DataFrame:
    frames = [load_market_cache(path) for path in paths]
    return combine_market_frames(frames)


def build_market_panel_from_raw(*, provider: str, interval: str, symbols: list[str], data_root: Path = DATA_ROOT) -> pd.DataFrame:
    paths = [
        data_root / "raw" / "market" / provider / interval / f"{safe_symbol(symbol)}.csv"
        for symbol in symbols
    ]
    return build_market_panel_from_files([path for path in paths if path.exists()])


def save_market_panel(panel: pd.DataFrame, *, interval: str, data_root: Path = DATA_ROOT) -> Path:
    target = data_root / "processed" / "market_panel" / interval / "panel.parquet"
    return write_table(panel, target).path


def load_market_panel(path: str | Path) -> pd.DataFrame:
    frame = read_table(path)
    if "timestamp" not in frame.columns:
        raise ValueError("market panel requires timestamp column")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame.dropna(subset=["timestamp", "symbol"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def build_missing_bars_report(panel: pd.DataFrame, *, interval: str) -> pd.DataFrame:
    reports = []
    for symbol, group in panel.groupby("symbol"):
        reports.append(missing_bars_report(group, interval=interval, symbol=str(symbol)))
    if not reports:
        return pd.DataFrame(columns=["symbol", "interval", "missing_timestamp"])
    return pd.concat(reports, ignore_index=True)
