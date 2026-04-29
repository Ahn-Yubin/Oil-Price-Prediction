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

## Backtesting

```bash
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 10 --models random_walk,drift,motif,pattern_mlp,deep_lstm_tcn_fusion,llm_context_seq_moe --no-plots
```

If a deep artifact is missing, that model is recorded as unavailable and the overall backtest continues.

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
