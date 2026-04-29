# Operations

Operations docs collect commands and environment settings that developers and operators run repeatedly.

## Server

```bash
uvicorn backend.app.main:app --reload --port 8000
```

The old `app.main:app` is a compatibility wrapper. New docs and operational commands should use `backend.app.main:app`.

## Training

```bash
python scripts/train/train_pretrained_models.py --interval 1d
```

## Backtesting

```bash
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 5 --models random_walk,drift,flat --no-plots
```

## Maintenance

```bash
python scripts/maintenance/check_docs_i18n.py
python scripts/maintenance/audit_unused_files.py
python scripts/maintenance/smoke_test_api.py
```

## Default Paths

Unless explicitly overridden, use `MODEL_DIR=artifacts/models`, `METADATA_DIR=artifacts/metadata`, and `DATA_DIR=data`.
