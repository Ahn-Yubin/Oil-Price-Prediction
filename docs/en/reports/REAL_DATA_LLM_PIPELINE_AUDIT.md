# Real Data / LLM Pipeline Audit

Date: 2026-04-30

## Summary

The repository already has the Universal Market Forecasting Dashboard structure and deep-model inference path. `artifacts/models` contains production deep `.pt` artifacts and legacy `.npz` artifacts, and `/api/forecast` can use selected deep models. At the start of this audit, however, `data/` had no real event/news/fundamental source data beyond `sample_market_events.csv`, and deep training was still centered on yfinance prices plus sample event context.

## Audit Items

| Item | Status | Evidence |
| --- | --- | --- |
| Real `.pt`/`.npz` files in `artifacts/models` | Available | `deep_lstm_tcn_fusion_1d_h45.pt`, `llm_context_seq_moe_1d_h45.pt`, and interval `global_dl_*.npz` files exist. |
| Deep metadata in `artifacts/metadata` | Available | `deep_lstm_tcn_fusion_1d_h45.json` and `llm_context_seq_moe_1d_h45.json` exist, both with `status=available`. |
| `/api/models` deep status | Available | `deep_artifact_availability()` is returned through `user_facing_models[].status` and training commands. Current `1d/h45` deep models are `available`. |
| Deep model use in `/api/forecast` | Available | `market_ai.forecasting.service._deep_comparison_models()` calls `forecast_with_deep_model()` for selected deep models. |
| `llm_context_seq_moe` inference event provider | Partial | `FileEventProvider.from_env()` is passed only when `ENABLE_LLM_CONTEXT=true`. The default is off, so inference may use zero event context. |
| Real event data beyond `sample_market_events.csv` | Missing | At audit start, `data/` had no raw news/event/fundamental dataset beyond the sample CSV. |
| Real data/context inputs in `train_deep_fusion_models.py` | Partial | yfinance download and `--events-path` were connected, but EIA/CFTC/CME/processed market panel inputs were missing. |
| Cross-asset features | Placeholder | `empty_cross_asset_window()` creates a missing-indicator matrix. Real related-asset panel values were not connected to training. |
| Backtest result storage | Available | `outputs/backtests` stores summary, details, horizon metrics, probabilistic metrics, leaderboard, and model availability CSV files. |
| `PROJECT_STATUS.md` freshness | Inconsistent | Existing docs said deep `1d/h45` artifacts were missing, but actual artifacts and metadata exist. |

## Available

- `/api/forecast` and `/api/chart` contracts are preserved.
- LLM safety guardrails warn on price target, p50/p90, and future return path fields and do not overwrite numeric forecasts.
- The forecast target remains a volatility-scaled cumulative log return distribution.
- Deep `.pt` artifacts and metadata sidecars are separated from source code.

## Missing Or Incomplete

- EIA/CFTC/CME/manual CSV ingest and point-in-time daily feature store.
- Reproducible yfinance market panel raw/cache/processed storage.
- News headline CSV and daily event context dataset generation.
- External API/local HTTP/offline file LLM context operation scripts.
- Processed-data deep training CLI.
- Latest snapshot structure for rolling leaderboards.
- Conformal quantile calibration artifacts and API wiring.

## Placeholders

- The cross-asset feature matrix is still dominated by a missing indicator rather than real related market panel values.
- `llm_context_seq_moe` metadata uses sample event CSV input, not a real news/event source.
- Current quantile bands are residual-volatility adapters and are not validated confidence intervals.

## Conclusion

The existing forecast/model API surface can be reused. This work should add `data/raw`, `data/processed`, manifests, manual/live providers, LLM context cache, processed-data training, leaderboard, and calibration without replacing the existing model/API contracts.
