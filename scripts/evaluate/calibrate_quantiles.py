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
from market_ai.modeling.calibration.conformal import compute_conformal_adjustment, save_calibration_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create conformal quantile calibration artifacts from rolling backtest details.")
    parser.add_argument("--details", default="")
    parser.add_argument("--symbol", default="CL=F")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details_path = Path(args.details) if args.details else PROJECT_DIR / "outputs" / "backtests" / f"{args.symbol.replace('=', '_')}_{args.interval}_details.csv"
    if not details_path.exists():
        raise SystemExit(f"Backtest details not found: {details_path}")
    details = pd.read_csv(details_path)
    artifact = compute_conformal_adjustment(details, model=args.model, symbol=args.symbol, interval=args.interval)
    path = save_calibration_artifact(artifact)
    print(f"Wrote calibration artifact status={artifact.calibration_status} n_origins={artifact.n_origins} path={path}")


if __name__ == "__main__":
    main()
