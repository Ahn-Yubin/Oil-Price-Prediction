# Oil Price Forecasting Dashboard

This repository is an oil-price-only forecasting system built with FastAPI, `market_ai`, and a frontend chart overlay. The goal is to combine the WTI crude oil (`CL=F`) chart, energy/macro indicators, news/LLM context, one unified deep forecasting model, and chart overlays into one operational workflow.

Korean documentation is primary. See [README.md](README.md). New contributors should start with [Project Status](docs/en/PROJECT_STATUS.md).

## Core Principles

- Production must not silently use mock/synthetic data.
- `/api/chart` backward compatibility must be preserved.
- The LLM is a context/event encoder, not a numeric price forecaster.
- The forecast target is a volatility-scaled cumulative log return distribution.
- Forecast prices are reconstructed with `price_t+h = current_price * exp(predicted_cumulative_log_return_h)`.
- The only user-facing forecast model is `oil_context_fusion`.
- `.pt`/`.npz` artifacts live in `artifacts/models`; metadata JSON lives in `artifacts/metadata`.
- Korean docs and English mirror docs must keep the same relative path structure.

## Quick Start

```bash
.venv/bin/python -m uvicorn backend.app.main:app --reload --port 8000
```

Main APIs:

- `GET /api/health`
- `GET /api/models`
- `GET /api/data-status?symbol=CL=F&interval=1d`
- `GET /api/forecast?symbol=CL=F&interval=1d`
- `GET /api/chart?symbol=CL=F&interval=1d`
- `GET /api/dashboard-analysis?symbol=CL=F&interval=1d&horizon=30`
- `GET /api/market-context?symbol=CL=F&interval=1d`
- `GET /api/explanation`
- `GET /api/backtests`
- `GET /api/backtests/visualization`

The operating dashboard shows a fixed `CL=F / 1D / 30-day` path and marks the 1W/2W/1M endpoints on the same h30 output. AI market commentary, news interpretation, and the forecast report are rendered from one external LLM response returned by `/api/dashboard-analysis`.

## Current Data

The currently usable training data includes:

- `data/processed/market_panel/{interval}/panel.csv`: `1d`, `1h`, `30m`, and `15m` market panels
- `data/processed/oil_fundamentals/eia_weekly.csv`: EIA weekly petroleum data
- `data/processed/oil_fundamentals/cftc_cot_weekly.csv`: CFTC COT weekly positioning data
- `data/processed/macro_panel/fred_daily_wide.csv`: FRED macro data such as rates, FX, dollar, and VIX series
- `data/raw/news/public_market_news.csv`: public news text
- `data/processed/event_context/event_context_daily.csv`: news/event context vectors
- `data/manifests/data_inventory.json`: data inventory

Current limitations are long-history CME futures curve data, longer news history, and sufficient calibration residuals. See [Data Pipeline](docs/en/DATA_PIPELINE.md).

## LLM Context

The LLM flow is:

```text
news/events -> LLM context encoder -> event context vector -> deep model input -> time-series model produces numeric forecast
```

Example for the Google hosted Gemma API:

```bash
export ENABLE_LLM_CONTEXT=true
export ENABLE_EXTERNAL_LLM_CALLS=true
export LLM_CONTEXT_MODE=google_generative
export LLM_API_KEY="YOUR_GOOGLE_API_KEY"
export LLM_API_BASE="https://generativelanguage.googleapis.com/v1beta"
export LLM_MODEL="gemma-3-27b-it"
export LLM_REQUEST_TIMEOUT_SECONDS=45
```

`export` only persists in the current shell. It disappears when the terminal closes. Use `.env` for project-specific persistent settings. The main server/script entrypoints now auto-load the project-root `.env`.

```bash
cp .env.example .env
```

Shell commands such as `echo "$LLM_MODEL"` do not automatically show `.env` values. Check the settings read by the project with Python:

```bash
.venv/bin/python - <<'PY'
from market_ai.config import get_settings
s = get_settings()
print("enable_llm_context:", s.enable_llm_context)
print("enable_external_llm_calls:", s.enable_external_llm_calls)
print("llm_model:", s.llm_model)
print("llm_api_base:", s.llm_api_base)
print("llm_request_timeout_seconds:", s.llm_request_timeout_seconds)
print("llm_api_key_set:", bool(s.llm_api_key))
PY
```

Validation:

```bash
.venv/bin/python scripts/llm/test_llm_context.py --mode openai_compatible --dry-run
.venv/bin/python scripts/llm/test_llm_context.py --mode google_generative --live
```

See [LLM Context](docs/en/LLM_CONTEXT.md).

## Data Build

```bash
.venv/bin/python scripts/data/fetch_market_prices.py --universe oil_core --interval 1d --period 10y
.venv/bin/python scripts/data/fetch_eia_petroleum.py
.venv/bin/python scripts/data/fetch_cftc_cot.py
.venv/bin/python scripts/data/build_event_context.py --news-path data/raw/news/public_market_news.csv --symbols CL=F,BZ=F,NG=F --mode local_rules
.venv/bin/python scripts/data/build_data_inventory.py
```

Public-data orchestration:

```bash
.venv/bin/python scripts/data/build_real_dataset.py \
  --universe oil_core \
  --interval 1d \
  --period 10y \
  --news-timespan 3m \
  --news-maxrecords 30
```

## Training

Training the single unified oil model with current processed data:

```bash
.venv/bin/python scripts/train/train_deep_fusion_models.py \
  --model oil_context_fusion \
  --interval 1d \
  --horizon 30 \
  --lookback 128 \
  --universe oil_core \
  --use-processed-data \
  --market-panel data/processed/market_panel/1d/panel.csv \
  --oil-fundamentals data/processed/oil_fundamentals/eia_weekly.csv \
  --cot data/processed/oil_fundamentals/cftc_cot_weekly.csv \
  --macro-panel data/processed/macro_panel/fred_daily_wide.csv \
  --event-context data/processed/event_context/event_context_daily.csv \
  --max-samples 0 \
  --epochs 5 \
  --batch-size 64 \
  --device mps \
  --force
```

`oil_context_fusion` merges the LSTM/TCN price encoders from `deep_lstm_tcn_fusion` with the context-gated expert structure from `llm_context_seq_moe`, and includes attention/pattern/motif expert branches. It consumes price features, related energy/macro markets, EIA/CFTC fundamentals, FRED macro data, and news/event context vectors.

## Validation

```bash
.venv/bin/python -m pytest tests/integration/test_api.py tests/unit/test_real_data_pipeline.py tests/unit/test_deep_dataset.py tests/unit/test_train_deep_fusion_cli_policy.py
.venv/bin/python -m compileall backend market_ai scripts
.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy
```

## Documentation

- [Project Status](docs/en/PROJECT_STATUS.md)
- [Architecture](docs/en/ARCHITECTURE.md)
- [API](docs/en/API.md)
- [Data Pipeline](docs/en/DATA_PIPELINE.md)
- [Model Design](docs/en/MODEL_DESIGN.md)
- [LLM Context](docs/en/LLM_CONTEXT.md)
- [Operations](docs/en/OPERATIONS.md)
- [Frontend](docs/en/FRONTEND.md)
- [Backtesting](docs/en/BACKTESTING.md)
- [Roadmap](docs/en/ROADMAP.md)
