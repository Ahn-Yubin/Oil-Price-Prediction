#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.config import get_settings
from market_ai.modeling.forecasters.neural_npz import train_and_save_pretrained_model
from market_ai.modeling.registry import metadata_for_artifact, metadata_sidecar_path


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
    p.add_argument("--metadata-only", action="store_true", help="Write metadata JSON sidecars for existing artifacts")
    return p.parse_args()


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
