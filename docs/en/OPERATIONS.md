# Operations

Operations docs collect repeated commands and environment settings for developers and operators.

## Server

```bash
uvicorn backend.app.main:app --reload --port 8000
```

The old `app.main:app` is a compatibility wrapper. New docs and operations commands should use `backend.app.main:app`.

## Training

Legacy `.npz` fallback:

```bash
python scripts/train/train_pretrained_models.py --interval 1d
```

Deep models:

```bash
python scripts/train/train_deep_fusion_models.py --model both --interval 1d --universe oil_core --epochs 10 --batch-size 64
python scripts/train/train_deep_fusion_models.py --model deep_lstm_tcn_fusion --interval 1d --quick-test --epochs 1 --max-samples 256
```

Artifacts are saved in `artifacts/models`, and metadata JSON is saved in `artifacts/metadata`.

Real processed-data training:

```bash
python scripts/train/train_deep_fusion_models.py \
  --model both \
  --interval 1d \
  --universe oil_core \
  --use-processed-data \
  --market-panel data/processed/market_panel/1d/panel.parquet \
  --oil-fundamentals data/processed/oil_fundamentals/eia_weekly.csv \
  --cot data/processed/oil_fundamentals/cftc_cot_weekly.csv \
  --cme-curve data/processed/oil_fundamentals/cme_curve_daily.csv \
  --event-context data/processed/event_context/event_context_daily.csv \
  --epochs 30 \
  --batch-size 64 \
  --force
```

If `panel.parquet` is absent and `panel.csv` exists, the loader reads the CSV fallback. Production training does not use synthetic fallback unless `--synthetic`, `--quick-test`, or `--allow-synthetic-fallback` is explicit.

## Data Build

```bash
python scripts/data/fetch_market_prices.py --universe oil_core --interval 1d --period 10y
python scripts/data/fetch_eia_petroleum.py --manual-csv path/to/eia.csv
python scripts/data/fetch_cftc_cot.py --manual-csv path/to/cftc.csv
python scripts/data/fetch_cme_settlements.py --manual-csv path/to/cme.csv
python scripts/data/build_event_context.py --events-path data/external/events/sample_market_events.csv --mode local_rules
python scripts/data/build_data_inventory.py
```

API keys and secrets must be injected through `.env` or shell environment variables and must not be committed.

## Backtesting

```bash
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 10 --models random_walk,drift,motif,pattern_mlp,deep_lstm_tcn_fusion,llm_context_seq_moe --no-plots
```

If a deep artifact is missing, that model is recorded as unavailable and the overall backtest continues.

Leaderboard and calibration:

```bash
python scripts/evaluate/run_model_leaderboard.py --symbols CL=F,BZ=F,NG=F --interval 1d --max-origins 50
python scripts/evaluate/calibrate_quantiles.py --model motif --symbol CL=F --interval 1d
```

## API and Chart

Pass model selection with `/api/forecast?symbol=CL=F&interval=1d&models=deep_lstm_tcn_fusion`. The frontend model selector sends the same query. The `/api/chart` schema remains backward compatible.

## Maintenance

```bash
python scripts/maintenance/check_docs_i18n.py --check-legacy
python -m compileall backend market_ai scripts
python scripts/maintenance/smoke_test_api.py
python -m pytest
```

If `python` is unavailable in the local shell, use `.venv/bin/python`.

## Default Paths

Unless overridden, use `MODEL_DIR=artifacts/models`, `METADATA_DIR=artifacts/metadata`, and `DATA_DIR=data`.
