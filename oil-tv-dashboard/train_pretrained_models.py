#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.global_dl_model import train_and_save_pretrained_model


DEFAULT_HORIZON = {
    "1d": 45,
    "1h": 72,
    "30m": 120,
    "15m": 192,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline pretraining for global DL forecast models")
    p.add_argument("--interval", choices=["1d", "1h", "30m", "15m"], default="")
    p.add_argument("--horizon", type=int, default=0)
    p.add_argument("--symbols", type=str, default="")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval:
        jobs = [(args.interval, args.horizon or DEFAULT_HORIZON[args.interval])]
    else:
        jobs = [(k, v) for k, v in DEFAULT_HORIZON.items()]

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    results = []
    for interval, horizon in jobs:
        model = train_and_save_pretrained_model(
            interval=interval,
            horizon=horizon,
            force=args.force,
            symbols=symbols,
        )
        meta = model["meta"]
        model_path = (
            Path(__file__).resolve().parent
            / "app"
            / "models"
            / f"global_dl_{interval}_h{horizon}.npz"
        )
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
