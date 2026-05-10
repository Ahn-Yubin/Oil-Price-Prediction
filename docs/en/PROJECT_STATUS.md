# Project Status

This document is the current canonical status map for the repository. Older audit/work reports have been folded into the active docs. Operational decisions should use this document plus `DATA_PIPELINE`, `LLM_CONTEXT`, and `OPERATIONS`.

## One-Line Summary

The project has moved from an oil-only dashboard toward a universal market forecasting platform. FastAPI backend, `market_ai` domain logic, frontend chart overlay, data CLIs, deep learning training CLIs, and model artifacts/metadata are separated.

The currently usable training data includes price panels, EIA weekly petroleum data, CFTC COT data, public news, and event context. Long-history CME futures curve data, longer news history, and calibration residuals still need work.

## Implementation Scope

| Area | Status |
| --- | --- |
| Backend | FastAPI app at `backend.app.main:app`; provides `/api/forecast`, `/api/chart`, and `/api/market-context` |
| Frontend | Forecast overlay, context markers, news/context panel, and scenario commentary |
| Market data | yfinance-based `research_core` panels can be built |
| Fundamentals | EIA bulk and CFTC ZIP/manual CSV ingestion available |
| CME | Manual CSV ingestion available; licensed CSV still needed |
| News/context | Yahoo RSS/GDELT public news collection and local_rules/external LLM context generation |
| Deep learning | `deep_lstm_tcn_fusion` and `llm_context_seq_moe` training and metadata storage |
| Backtest/calibration | Rolling backtest and calibration scripts exist; sufficient coverage validation must still be run |
| Docs | Korean/English mirror structure maintained |

## Directory Roles

| Path | Role |
| --- | --- |
| `backend/` | FastAPI routes, static frontend serving, service adapters |
| `market_ai/` | Core data, feature, forecasting, modeling, calibration, regime, backtesting, and LLM context logic |
| `frontend/` | Chart overlay UI, controls, and panels |
| `scripts/` | Human-run data/train/evaluate/maintenance CLIs |
| `artifacts/models/` | `.npz` and `.pt` model artifacts |
| `artifacts/metadata/` | model metadata JSON |
| `data/` | raw/interim/processed/external/features/manifests |
| `outputs/` | generated outputs; stale report Markdown is folded into canonical docs and removed |
| `docs/ko`, `docs/en` | Korean primary docs and English mirror docs |
| `tests/` | unit/integration tests |

## API Status

| Endpoint | Role |
| --- | --- |
| `GET /api/health` | settings, model artifacts, provider status |
| `GET /api/models` | model registry and artifact availability |
| `GET /api/data-status` | data state for a symbol/interval |
| `GET /api/forecast` | new forecast contract |
| `GET /api/chart` | legacy chart compatibility contract; do not remove |
| `GET /api/market-context` | news, context points, and scenario commentary |
| `GET /api/explanation` | forecast and optional LLM context explanation |
| `GET /api/backtests` | backtest output lookup |

New fields must be additive. `/api/chart` remains a backward-compatibility target.

## Forecasting Policy

Numeric forecasts are produced by time-series models and baselines, not by the LLM.

```text
target = future cumulative log return / recent realized volatility
price_t+h = current_price * exp(predicted_cumulative_log_return_h)
```

The LLM converts news/events into a context vector and can indirectly affect gating/confidence/uncertainty in `llm_context_seq_moe`. LLM output must not create or overwrite prices, target prices, p50/p90, or future return paths.

## Model Status

| Model | Class | Status |
| --- | --- | --- |
| `motif` | Classical | Available |
| `pattern_mlp` | Deep `.npz` | Uses interval artifacts |
| `deep_lstm_tcn_fusion` | Deep `.pt` | Can train on processed data |
| `llm_context_seq_moe` | Deep `.pt` + event context | Consumes LLM/event context input |
| `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive` | Baseline | Available |
| `flat`, `simple_moving_average_path`, `regime_ensemble` | Backtest-only | Not default operational forecast models |
| `cycle`, `lstm`, `tcn`, `ensemble` | Removed/deprecated | Not active models |

Deep artifacts are stored in `artifacts/models` and `artifacts/metadata`. Smoke/quick artifacts do not imply production performance.

## Current Data And Size

Exact row counts and date ranges are tracked in `data/manifests/data_inventory.json`. The key current datasets are:

| Data | Path | Description |
| --- | --- | --- |
| Market panel | `data/processed/market_panel/{interval}/panel.csv` | `1d`, `1h`, `30m`, and `15m` price panels |
| EIA weekly | `data/processed/oil_fundamentals/eia_weekly.csv` | Long-history petroleum weekly series |
| CFTC COT | `data/processed/oil_fundamentals/cftc_cot_weekly.csv` | Weekly positioning series |
| News | `data/raw/news/public_market_news.csv` | Public news text |
| Event context | `data/processed/event_context/event_context_daily.csv` | Daily LLM/local context vectors |
| Inventory | `data/manifests/data_inventory.json` | Data quality, date range, and row count records |

Missing data: CME curve, longer news history, sufficient calibration residuals, and sub-daily release timestamps.

## LLM Configuration Status

The main server/script entrypoints auto-load the project-root `.env`. `export` applies only to that shell and disappears when the terminal closes. Shell `echo` may be empty even when Python processes can read `.env`.

Validation:

```bash
.venv/bin/python - <<'PY'
from market_ai.config import get_settings
s = get_settings()
print("llm_model:", s.llm_model)
print("llm_api_key_set:", bool(s.llm_api_key))
PY
.venv/bin/python scripts/llm/test_llm_context.py --mode google_generative --live
```

Success means `safety_check_passed=true` and no `External LLM fallback` warning.

## Can The Current Data Train Models?

Yes, the current data can train models, with the following interpretation:

- `horizon=8`: overlaps more with recent news context and is useful for smoke-level LLM/event context checks.
- `horizon=45`: can produce longer-horizon artifacts, but event context impact is limited if news history is short.
- Without CME curve data, term-structure edge is missing.
- Without sufficient calibration residuals, bands must not be called validated confidence intervals.

Use the training section in `docs/en/OPERATIONS.md` as the source of truth for commands.

## Completed Improvements

- Added EIA bulk download/normalization path
- Added CFTC ZIP/CSV/manual CSV normalization path
- Cleaned CME manual/URL CSV normalization path
- Added public news ingestion and event context generation
- Optimized deep dataset recent-origin sampling
- Fixed `train_deep_fusion_models.py --use-processed-data`
- Added `/api/market-context`
- Added frontend context markers, news panel, and scenario commentary
- Documented Google OpenAI-compatible LLM context setup
- Removed stale report Markdown and updated canonical docs

## Next Priorities

1. Verify Google LLM live call with `.env` or shell exports
2. Regenerate LLM context with live calls
3. Retrain `llm_context_seq_moe` and `deep_lstm_tcn_fusion` for h8/h45
4. Acquire and ingest CME settlement/curve CSV
5. Run rolling backtest and quantile calibration
6. Load longer news history
7. Exercise `/api/market-context` in the dashboard and check UX/performance
