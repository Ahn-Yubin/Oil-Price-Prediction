#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.backtesting.runner import download_candles, run_rolling_backtest, write_backtest_outputs
from market_ai.config import PROJECT_DIR
from market_ai.constants import INTERVAL_TO_HORIZON


DEFAULT_MODELS = "oil_context_fusion,random_walk,drift,motif,pattern_mlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-symbol rolling model leaderboard.")
    parser.add_argument("--symbols", default="CL=F")
    parser.add_argument("--interval", default="1d", choices=["1d", "1h", "30m", "15m"])
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--max-origins", type=int, default=50)
    parser.add_argument("--lookback", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--output-root", default=str(PROJECT_DIR / "outputs" / "backtests" / "leaderboards"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_root) / created
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    all_frames: dict[str, list[pd.DataFrame]] = {
        "leaderboard": [],
        "horizon_metrics": [],
        "probabilistic_metrics": [],
        "regime_metrics": [],
        "model_availability": [],
    }
    failures = []
    data_sources = {}
    for symbol in symbols:
        try:
            candles = download_candles(symbol, args.interval)
            close = candles["close"].to_numpy(dtype=float)
            horizon = args.horizon or INTERVAL_TO_HORIZON[args.interval]
            lookback = args.lookback or {"1d": 260, "1h": 420, "30m": 420, "15m": 560}.get(args.interval, 420)
            step = args.step or max(1, horizon // 4)
            outputs = run_rolling_backtest(
                close,
                args.interval,
                models,
                symbol=symbol,
                candles=candles,
                lookback=lookback,
                horizon=horizon,
                step=step,
                max_origins=args.max_origins,
                rolling=True,
                expanding=False,
                include_regime_breakdown=True,
            )
            data_sources[symbol] = {
                "source": candles.attrs.get("source", "yfinance"),
                "path": candles.attrs.get("source_path"),
                "rows": int(len(candles)),
            }
            prefix = f"{symbol.replace('=', '_')}_{args.interval}"
            write_backtest_outputs(outputs, output_dir, prefix)
            for key in all_frames:
                frame = outputs[key].copy()
                if not frame.empty:
                    frame.insert(0, "symbol", symbol)
                    frame.insert(1, "interval", args.interval)
                    all_frames[key].append(frame)
        except Exception as exc:
            failures.append({"symbol": symbol, "interval": args.interval, "error": str(exc)})
    for key, frames in all_frames.items():
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combined.to_csv(output_dir / f"{key}.csv", index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "interval": args.interval,
        "models": models,
        "max_origins": args.max_origins,
        "failures": failures,
        "data_sources": data_sources,
        "output_dir": str(output_dir.relative_to(PROJECT_DIR)),
    }
    (output_dir / "summary.md").write_text(_summary_md(summary, all_frames), encoding="utf-8")
    (output_dir / "run_meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = Path(args.output_root) / "latest.json"
    latest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _summary_md(summary: dict, frames: dict[str, list[pd.DataFrame]]) -> str:
    leaderboard = pd.concat(frames["leaderboard"], ignore_index=True) if frames["leaderboard"] else pd.DataFrame()
    lines = [
        "# Model Leaderboard Summary",
        "",
        f"- created_at: {summary['created_at']}",
        f"- interval: {summary['interval']}",
        f"- symbols: {', '.join(summary['symbols'])}",
        f"- max_origins: {summary['max_origins']}",
        f"- data_sources: {summary.get('data_sources', {})}",
        "",
    ]
    if leaderboard.empty:
        lines.append("No successful leaderboard rows were produced.")
    else:
        lines.extend(["Top rows:", "", "```", leaderboard.head(12).to_csv(index=False).strip(), "```"])
    if summary["failures"]:
        lines.extend(["", "Failures:", ""])
        for item in summary["failures"]:
            lines.append(f"- {item['symbol']}: {item['error']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
