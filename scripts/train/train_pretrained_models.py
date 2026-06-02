#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.config import get_settings
from market_ai.data.storage import read_table
from market_ai.modeling.forecasters.neural_npz import train_and_save_pretrained_model, train_and_save_pretrained_model_from_series
from market_ai.modeling.registry import metadata_for_artifact, metadata_sidecar_path


DEFAULT_HORIZON = {
    "1d": 30,
    "1h": 30,
    "30m": 30,
    "15m": 30,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline pretraining for global DL forecast models")
    p.add_argument("--interval", choices=["1d", "1h", "30m", "15m"], default="")
    p.add_argument("--horizon", type=int, default=0)
    p.add_argument("--symbols", type=str, default="")
    p.add_argument("--market-panel", type=str, default="", help="Optional processed market panel CSV/parquet for offline local training")
    p.add_argument("--force", action="store_true")
    p.add_argument("--metadata-only", action="store_true", help="Write metadata JSON sidecars for existing artifacts")
    return p.parse_args()


def _series_from_market_panel(path: str, symbols: list[str] | None = None) -> tuple[list, list[str]]:
    frame = read_table(Path(path))
    required = {"timestamp", "symbol", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Market panel missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "symbol", "close"])
    frame = frame[frame["close"] > 0.0].sort_values(["symbol", "timestamp"])
    if symbols:
        allowed = set(symbols)
        frame = frame[frame["symbol"].astype(str).isin(allowed)]
    series = []
    used = []
    for symbol, group in frame.groupby("symbol", sort=True):
        close = group["close"].to_numpy(dtype=float)
        if len(close) < 200:
            continue
        series.append(close)
        used.append(str(symbol))
    if not series:
        raise RuntimeError("No usable market panel series were found")
    return series, used


def main() -> None:
    args = parse_args()
    if args.interval:
        jobs = [(args.interval, args.horizon or DEFAULT_HORIZON[args.interval])]
    else:
        jobs = [(k, v) for k, v in DEFAULT_HORIZON.items()]

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    results = []
    settings = get_settings()
    for interval, horizon in jobs:
        model_path = settings.model_dir / f"global_dl_{interval}_h{horizon}.npz"
        if args.metadata_only:
            if not model_path.exists():
                results.append({"interval": interval, "horizon": horizon, "path": str(model_path), "status": "missing"})
                continue
            metadata = metadata_for_artifact(model_path)
            metadata_sidecar_path(model_path).write_text(
                json.dumps(metadata.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            results.append(
                {
                    "interval": interval,
                    "horizon": horizon,
                    "path": str(model_path),
                    "metadata_path": str(metadata_sidecar_path(model_path)),
                    "status": "metadata_written",
                }
            )
            continue

        if args.market_panel:
            series, used_symbols = _series_from_market_panel(args.market_panel, symbols=symbols)
            model = train_and_save_pretrained_model_from_series(
                interval=interval,
                horizon=horizon,
                force=args.force,
                series=series,
                symbols=used_symbols,
            )
        else:
            model = train_and_save_pretrained_model(
                interval=interval,
                horizon=horizon,
                force=args.force,
                symbols=symbols,
            )
        meta = model["meta"]
        results.append(
            {
                "interval": interval,
                "horizon": horizon,
                "path": str(model_path),
                "trained_at": meta.get("trained_at"),
                "n_train": meta.get("n_train"),
                "n_val": meta.get("n_val"),
                "symbols": meta.get("symbols", []),
            }
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
