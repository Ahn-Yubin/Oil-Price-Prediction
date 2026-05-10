# Operations

This document covers repeated commands, environment variables, data builds, training, validation, and server operation. If the local shell has no `python`, use `.venv/bin/python` for all commands.

## Server

```bash
.venv/bin/python -m uvicorn backend.app.main:app --reload --port 8000
```

The old `app.main:app` remains a compatibility wrapper. New operational commands should use `backend.app.main:app`.

Environment changes are not automatically applied to an already running server. Restart the server after changing LLM keys, model names, or data paths.

## export And .env

`export` only affects the current terminal session and child processes started from that session. It disappears when the terminal closes and is not shared with other terminals.

Check the current shell:

```bash
echo "$ENABLE_LLM_CONTEXT"
echo "$ENABLE_EXTERNAL_LLM_CALLS"
echo "$LLM_CONTEXT_MODE"
echo "$LLM_API_BASE"
echo "$LLM_MODEL"
test -n "$LLM_API_KEY" && echo "LLM_API_KEY is set" || echo "LLM_API_KEY is missing"
```

There are two persistence options.

1. Put exports in `~/.zshrc` for personal shell defaults.
2. Create a project `.env`. The main server/script entrypoints auto-load the project-root `.env`.

```bash
cp .env.example .env
# edit LLM_API_KEY and LLM_MODEL inside .env
```

`.env` and `.env.*` are ignored by Git. Still, never print or commit API keys.

Shell commands such as `echo "$LLM_MODEL"` do not automatically show `.env` values. Check the settings read by the project with Python:

```bash
.venv/bin/python - <<'PY'
from market_ai.config import get_settings
s = get_settings()
print("enable_llm_context:", s.enable_llm_context)
print("enable_external_llm_calls:", s.enable_external_llm_calls)
print("llm_model:", s.llm_model)
print("llm_api_base:", s.llm_api_base)
print("llm_api_key_set:", bool(s.llm_api_key))
PY
```

## Google Gemma/Gemini LLM Context

Example for Google's OpenAI-compatible endpoint:

```bash
export ENABLE_LLM_CONTEXT=true
export ENABLE_EXTERNAL_LLM_CALLS=true
export LLM_CONTEXT_MODE=google_generative
export LLM_API_KEY="YOUR_GOOGLE_API_KEY"
export LLM_API_BASE="https://generativelanguage.googleapis.com/v1beta"
export LLM_MODEL="gemma-3-27b-it"
```

Dry-run:

```bash
.venv/bin/python scripts/llm/test_llm_context.py --mode openai_compatible --dry-run
```

Live call:

```bash
.venv/bin/python scripts/llm/test_llm_context.py --mode google_generative --live
```

Success means `safety_check_passed=true` and no `External LLM fallback` warning.

## Data Build

Market panel:

```bash
.venv/bin/python scripts/data/fetch_market_prices.py --universe research_core --interval 1d --period 10y
.venv/bin/python scripts/data/fetch_market_prices.py --universe research_core --interval 1h --period 730d
.venv/bin/python scripts/data/fetch_market_prices.py --universe research_core --interval 30m --period 60d
.venv/bin/python scripts/data/fetch_market_prices.py --universe research_core --interval 15m --period 60d
```

Oil fundamentals and positioning:

```bash
.venv/bin/python scripts/data/fetch_eia_petroleum.py
.venv/bin/python scripts/data/fetch_cftc_cot.py
.venv/bin/python scripts/data/fetch_cme_settlements.py --manual-csv data/external/fundamentals/cme_settlements.csv
```

Event context:

```bash
.venv/bin/python scripts/data/build_event_context.py \
  --news-path data/raw/news/public_market_news.csv \
  --symbols CL=F,BZ=F,NG=F,RB=F,HO=F,GC=F,SI=F,HG=F,DX-Y.NYB,EURUSD=X,USDKRW=X,JPY=X,SPY,QQQ,^GSPC,^VIX,XLE,USO \
  --mode google_generative \
  --live
```

Manifest:

```bash
.venv/bin/python scripts/data/build_data_inventory.py
```

## Training

One-day training with current processed data:

```bash
.venv/bin/python scripts/train/train_deep_fusion_models.py \
  --model both \
  --interval 1d \
  --horizon 8 \
  --lookback 128 \
  --universe research_core \
  --use-processed-data \
  --market-panel data/processed/market_panel/1d/panel.csv \
  --oil-fundamentals data/processed/oil_fundamentals/eia_weekly.csv \
  --cot data/processed/oil_fundamentals/cftc_cot_weekly.csv \
  --event-context data/processed/event_context/event_context_daily.csv \
  --max-samples 512 \
  --epochs 3 \
  --batch-size 64 \
  --device mps \
  --force
```

Longer horizon artifact:

```bash
.venv/bin/python scripts/train/train_deep_fusion_models.py \
  --model both \
  --interval 1d \
  --horizon 45 \
  --lookback 128 \
  --universe research_core \
  --use-processed-data \
  --market-panel data/processed/market_panel/1d/panel.csv \
  --oil-fundamentals data/processed/oil_fundamentals/eia_weekly.csv \
  --cot data/processed/oil_fundamentals/cftc_cot_weekly.csv \
  --event-context data/processed/event_context/event_context_daily.csv \
  --max-samples 512 \
  --epochs 3 \
  --batch-size 64 \
  --device mps \
  --force
```

Production training does not use synthetic fallback unless `--synthetic`, `--quick-test`, or `--allow-synthetic-fallback` is explicit. Artifacts are stored in `artifacts/models`; metadata JSON is stored in `artifacts/metadata`.

## Backtest And Calibration

```bash
.venv/bin/python scripts/backtest/run_backtest.py \
  --symbol CL=F \
  --interval 1d \
  --max-origins 10 \
  --models random_walk,drift,motif,pattern_mlp,deep_lstm_tcn_fusion,llm_context_seq_moe \
  --no-plots
```

Leaderboard and calibration:

```bash
.venv/bin/python scripts/evaluate/run_model_leaderboard.py --symbols CL=F,BZ=F,NG=F --interval 1d --max-origins 50
.venv/bin/python scripts/evaluate/calibrate_quantiles.py --model motif --symbol CL=F --interval 1d
```

Until calibration artifacts are sufficiently validated, forecast bands are residual-volatility adapters, not validated confidence intervals.

## API And Chart

- `/api/forecast`: new forecast contract
- `/api/chart`: legacy chart compatibility contract
- `/api/market-context`: news, context markers, and model scenario commentary

Example:

```bash
curl "http://127.0.0.1:8000/api/market-context?symbol=NYMEX:CL1%21&interval=1d&models=llm_context_seq_moe"
```

## Validation

```bash
.venv/bin/python -m pytest tests/integration/test_api.py tests/unit/test_real_data_pipeline.py tests/unit/test_deep_dataset.py tests/unit/test_train_deep_fusion_cli_policy.py
.venv/bin/python -m compileall backend market_ai scripts
.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy
```

Frontend JS syntax:

```bash
node --check frontend/src/main.js
```

If `npm` is not in PATH, Vite build cannot be run. If the Node runtime is located elsewhere, run `--check` with that `node` binary.
