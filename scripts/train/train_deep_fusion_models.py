#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_ai.config import PROJECT_DIR
from market_ai.data.deep_dataset import DeepDataset, _time_split_indices, build_deep_dataset_from_frame, build_synthetic_deep_dataset, combine_auxiliary_feature_frames
from market_ai.data.event_providers import EVENT_FILE_ENV_VARS, FileEventProvider
from market_ai.data.market_panel import load_market_panel
from market_ai.data.storage import read_table
from market_ai.data.symbol_universe import load_symbol_universe
from market_ai.modeling.deep.artifacts import deep_artifact_name, deep_metadata_name, save_deep_artifact, write_deep_metadata
from market_ai.modeling.deep.training import train_deep_model
from market_ai.schemas.deep_learning import DeepDatasetConfig


DEFAULT_HORIZON = {"1d": 45, "1h": 72, "30m": 120, "15m": 192}
DEFAULT_LOOKBACK = {"1d": 128, "1h": 192, "30m": 240, "15m": 288}
UNIVERSE_PATH = PROJECT_DIR / "configs" / "symbol_universe.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train artifact-based deep sequence forecast models")
    parser.add_argument("--model", choices=["deep_lstm_tcn_fusion", "llm_context_seq_moe", "both"], default="both")
    parser.add_argument("--interval", choices=["1d", "1h", "30m", "15m"], default="1d")
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--lookback", type=int, default=0)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--universe", default="oil_core")
    parser.add_argument("--related-assets", dest="related_assets", action="store_true", default=True)
    parser.add_argument("--no-related-assets", dest="related_assets", action="store_false")
    parser.add_argument("--llm-context", dest="llm_context", action="store_true", default=True)
    parser.add_argument("--no-llm-context", dest="llm_context", action="store_false")
    parser.add_argument("--events-path", default="")
    parser.add_argument("--use-processed-data", action="store_true")
    parser.add_argument("--market-panel", default="")
    parser.add_argument("--oil-fundamentals", default="")
    parser.add_argument("--cot", default="")
    parser.add_argument("--cme-curve", default="")
    parser.add_argument("--event-context", default="")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--allow-synthetic-fallback", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [item.strip() for item in args.symbols.split(",") if item.strip()]
    universe = load_symbol_universe(UNIVERSE_PATH)
    return universe.get(args.universe, universe.get("oil_core", ["CL=F"]))


def _download_frame(symbol: str, interval: str) -> pd.DataFrame | None:
    period = {"1d": "10y", "1h": "730d", "30m": "60d", "15m": "60d"}.get(interval, "5y")
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if data.empty:
        return None
    frame = data.reset_index()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in frame.columns]
    date_col = "Date" if "Date" in frame.columns else "Datetime" if "Datetime" in frame.columns else None
    if not date_col:
        return None
    frame = frame.rename(
        columns={
            date_col: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return frame[["date", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)


def _event_paths_from_args(args: argparse.Namespace) -> list[str]:
    if args.events_path:
        return [item.strip() for item in args.events_path.split(",") if item.strip()]
    return [os.environ.get(name, "").strip() for name in EVENT_FILE_ENV_VARS if os.environ.get(name, "").strip()]


def _event_provider_from_args(args: argparse.Namespace, config: DeepDatasetConfig) -> FileEventProvider | None:
    if not config.event_context_enabled:
        return None
    return FileEventProvider(paths=_event_paths_from_args(args))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _sample_time_bounds(dataset) -> tuple[str | None, str | None]:
    values = [sample.as_of_time.isoformat() for sample in dataset.samples]
    if not values:
        return None, None
    return min(values), max(values)


def build_dataset(args: argparse.Namespace, symbols: list[str], config: DeepDatasetConfig):
    event_paths = _event_paths_from_args(args)
    event_provider = _event_provider_from_args(args, config)
    if args.synthetic or args.quick_test:
        return build_synthetic_deep_dataset(config), {
            "source": "synthetic",
            "symbols_used": symbols,
            "warnings": [],
            "synthetic_used": True,
            "events_path": event_paths,
        }
    if getattr(args, "use_processed_data", False):
        if not getattr(args, "market_panel", ""):
            raise RuntimeError("--use-processed-data requires --market-panel")
        panel = load_market_panel(args.market_panel)
        auxiliary = combine_auxiliary_feature_frames(
            oil_fundamentals=read_table(args.oil_fundamentals) if getattr(args, "oil_fundamentals", "") else None,
            cot=read_table(args.cot) if getattr(args, "cot", "") else None,
            cme_curve=read_table(args.cme_curve) if getattr(args, "cme_curve", "") else None,
        )
        event_context_frame = read_table(args.event_context) if getattr(args, "event_context", "") else None
        samples = []
        warnings: list[str] = []
        for symbol in symbols:
            symbol_panel = panel[panel["symbol"].astype(str) == symbol].copy()
            if symbol_panel.empty:
                warnings.append(f"No processed market panel rows for {symbol}")
                continue
            symbol_panel = symbol_panel.rename(columns={"timestamp": "date"})
            try:
                ds = build_deep_dataset_from_frame(
                    symbol=symbol,
                    interval=args.interval,
                    candles=symbol_panel,
                    config=config,
                    event_provider=event_provider,
                    auxiliary_frame=auxiliary,
                    market_panel=panel,
                    event_context_frame=event_context_frame,
                )
                samples.extend(ds.samples)
            except Exception as exc:
                warnings.append(f"{symbol}: {exc}")
        if not samples:
            raise RuntimeError(f"No usable processed samples were produced. Warnings: {warnings}")
        train_idx, val_idx, test_idx = _time_split_indices(len(samples), config.validation_ratio, config.test_ratio)
        return DeepDataset(samples=samples, train_indices=train_idx, validation_indices=val_idx, test_indices=test_idx), {
            "source": "processed",
            "symbols_used": symbols,
            "warnings": warnings,
            "synthetic_used": False,
            "events_path": event_paths,
            "data_inputs": {
                "market_panel": getattr(args, "market_panel", ""),
                "oil_fundamentals": getattr(args, "oil_fundamentals", "") or None,
                "cot": getattr(args, "cot", "") or None,
                "cme_curve": getattr(args, "cme_curve", "") or None,
                "event_context": getattr(args, "event_context", "") or None,
            },
        }
    samples = []
    warnings: list[str] = []
    for symbol in symbols:
        try:
            frame = _download_frame(symbol, args.interval)
            if frame is None:
                warnings.append(f"No usable yfinance data for {symbol}")
                continue
            ds = build_deep_dataset_from_frame(
                symbol=symbol,
                interval=args.interval,
                candles=frame,
                config=config,
                event_provider=event_provider,
            )
            samples.extend(ds.samples)
        except Exception as exc:
            warnings.append(f"{symbol}: {exc}")
    if not samples:
        if not args.allow_synthetic_fallback:
            raise RuntimeError(
                "No usable yfinance samples were produced. Production training does not use synthetic fallback by default. "
                "Use --synthetic for explicit synthetic smoke data, --quick-test for smoke training, or "
                "--allow-synthetic-fallback to permit fallback after yfinance failure."
            )
        fallback = build_synthetic_deep_dataset(config)
        return fallback, {
            "source": "synthetic_fallback",
            "symbols_used": symbols,
            "warnings": warnings,
            "synthetic_used": True,
            "events_path": event_paths,
        }
    from market_ai.data.deep_dataset import DeepDataset, _time_split_indices

    train_idx, val_idx, test_idx = _time_split_indices(len(samples), config.validation_ratio, config.test_ratio)
    return DeepDataset(samples=samples, train_indices=train_idx, validation_indices=val_idx, test_indices=test_idx), {
        "source": "yfinance",
        "symbols_used": symbols,
        "warnings": warnings,
        "synthetic_used": False,
        "events_path": event_paths,
        "data_inputs": {
            "market_panel": None,
            "oil_fundamentals": None,
            "cot": None,
            "cme_curve": None,
            "event_context": None,
        },
    }


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def train_one(model_name: str, args: argparse.Namespace, dataset, config: DeepDatasetConfig, data_report: dict[str, Any]) -> dict[str, Any]:
    if args.quick_test:
        artifact_dir = PROJECT_DIR / "artifacts" / "smoke" / "models"
        metadata_dir = PROJECT_DIR / "artifacts" / "smoke" / "metadata"
    else:
        artifact_dir = PROJECT_DIR / "artifacts" / "models"
        metadata_dir = PROJECT_DIR / "artifacts" / "metadata"
    output_dir = PROJECT_DIR / "outputs" / "reports"
    artifact_path = artifact_dir / deep_artifact_name(model_name, args.interval, config.horizon)
    metadata_path = metadata_dir / deep_metadata_name(model_name, args.interval, config.horizon)
    if artifact_path.exists() and not args.force and not args.quick_test:
        raise RuntimeError(f"Artifact already exists: {artifact_path}. Use --force to overwrite.")
    train_start, train_end = _sample_time_bounds(dataset)
    synthetic_used = bool(data_report.get("synthetic_used"))
    status = "smoke_only" if args.quick_test else "synthetic_only" if synthetic_used else "available"
    if args.metadata_only:
        status = "failed"
    metadata: dict[str, Any] = {
        "model_name": model_name,
        "artifact_file": artifact_path.name,
        "interval": args.interval,
        "horizon": config.horizon,
        "lookback": config.lookback,
        "target": "volatility_scaled_cumulative_log_return_distribution",
        "feature_set": dataset.feature_version,
        "asset_universe": config.symbols,
        "supported_intervals": [args.interval],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_start": train_start,
        "train_end": train_end,
        "training_cutoff": train_end,
        "metrics": {},
        "n_train": int(len(dataset.train_indices)),
        "n_val": int(len(dataset.validation_indices)),
        "n_test": int(len(dataset.test_indices)),
        "data_source": data_report.get("source"),
        "synthetic_used": synthetic_used,
        "event_context_enabled": bool(config.event_context_enabled),
        "events_path": list(data_report.get("events_path") or []),
        "related_assets_enabled": bool(config.related_assets_enabled),
        "git_commit": _git_commit(),
        "data_report": data_report,
        "data_inputs": data_report.get("data_inputs", {}),
        "status": status,
        "notes": "Quick-test smoke artifact; not for production inference." if args.quick_test else None,
    }
    if args.metadata_only:
        write_deep_metadata(metadata_path, model_name=model_name, artifact_path=artifact_path, metadata=metadata)
        return metadata
    result = train_deep_model(
        model_name,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=_resolve_device(args.device),
        seed=args.seed,
    )
    metadata["metrics"] = {
        "train_loss": result.train_loss,
        "validation_loss": result.validation_loss,
        "epochs_ran": result.epochs_ran,
        "n_train": metadata["n_train"],
        "n_val": metadata["n_val"],
        "n_test": metadata["n_test"],
    }
    metadata["deep_config"] = result.model.config_dict()
    save_deep_artifact(result.model, artifact_path, model_name=model_name, metadata=metadata)
    write_deep_metadata(metadata_path, model_name=model_name, artifact_path=artifact_path, metadata=metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / f"deep_training_{model_name}_{args.interval}.json"
    report_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md = output_dir / f"deep_training_{model_name}_{args.interval}.md"
    report_md.write_text(
        "\n".join(
            [
                f"# Deep Training Report: {model_name}",
                "",
                f"- interval: {args.interval}",
                f"- horizon: {config.horizon}",
                f"- lookback: {config.lookback}",
                f"- train_loss: {result.train_loss}",
                f"- validation_loss: {result.validation_loss}",
                f"- artifact: {artifact_path.name}",
            ]
        ),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    args = parse_args()
    symbols = resolve_symbols(args)
    horizon = args.horizon or DEFAULT_HORIZON[args.interval]
    lookback = args.lookback or DEFAULT_LOOKBACK[args.interval]
    if args.quick_test:
        horizon = min(horizon, 8)
        lookback = min(lookback, 32)
        args.epochs = max(1, args.epochs)
        args.max_samples = min(args.max_samples or 128, 128)
    config = DeepDatasetConfig(
        interval=args.interval,
        lookback=lookback,
        horizon=horizon,
        symbols=symbols,
        related_assets_enabled=args.related_assets,
        llm_context_enabled=args.llm_context,
        event_context_enabled=args.llm_context,
        max_samples=args.max_samples,
        min_history=lookback,
        seed=args.seed,
    )
    dataset, data_report = build_dataset(args, symbols, config)
    model_names = ["deep_lstm_tcn_fusion", "llm_context_seq_moe"] if args.model == "both" else [args.model]
    reports = [train_one(model_name, args, dataset, config, data_report) for model_name in model_names]
    print(json.dumps({"trained": model_names, "reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
