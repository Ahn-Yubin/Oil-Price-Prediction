# Real Data / LLM Pipeline Implementation Report

Date: 2026-04-30

## 1. Implementation Summary

- Added data lake directories and manifest layer.
- Added yfinance market panel storage/cleaning CLI.
- Added EIA/CFTC/CME manual CSV ingest and point-in-time daily conversion paths.
- Added event/news CSV to daily event context and `llm_context_cache.jsonl` pipeline.
- Added LLM modes `none`, `local_rules`, `openai_compatible`, `local_http`, and `offline_file`; live calls require explicit options.
- Extended the deep dataset and training CLI to accept processed market panels, fundamentals, COT, CME curves, and event context.
- Wired rolling leaderboard latest outputs, conformal calibration artifacts, `/api/backtests`, `/api/forecast` calibration status, and frontend diagnostics badge/panel.

## 2. New Data Pipeline

Added structure:

- `data/raw/market`, `data/raw/eia`, `data/raw/cftc`, `data/raw/cme`, `data/raw/events`, `data/raw/news`
- `data/interim/market`, `data/interim/fundamentals`, `data/interim/events`
- `data/processed/market_panel`, `data/processed/oil_fundamentals`, `data/processed/event_context`
- `data/features/deep_training`
- `data/manifests/data_inventory.json`, `data/manifests/latest_snapshot.json`

Run results:

- `fetch_market_prices.py --universe oil_core --interval 1d --period 5y` succeeded.
- Processed market panel: 6,294 rows, symbols `CL=F,BZ=F,NG=F,RB=F,HO=F`.
- Daily event context: 39 rows for `CL=F,BZ=F,NG=F`.
- Data inventory: 11 dataset entries.

## 3. Available Real Data Sources

- yfinance market prices: verified.
- EIA petroleum: API key or manual CSV.
- CFTC COT: official/manual CSV URL or manual CSV.
- CME settlements: manual CSV or licensed URL. No fake scraping.
- Events/news: manual event CSV, news headline CSV, offline LLM cache.

## 4. LLM Operating Modes

Validation command results:

- `local_rules`: passed, embedding dim 13.
- `openai_compatible --dry-run`: passed, no external call.
- `local_http --dry-run`: passed, local_rules fallback used.

The LLM safety check disallows price targets, p50/p90, and future return paths. The LLM is only a context/event encoder and explanation layer.

## 5. Training Results

- Quick-test smoke training:
  - `deep_lstm_tcn_fusion`, `1d/h8`, synthetic smoke, 1 epoch passed.
  - `llm_context_seq_moe`, `1d/h8`, synthetic smoke, 1 epoch passed.
- Existing production deep artifacts:
  - `deep_lstm_tcn_fusion_1d_h45.pt`: available.
  - `llm_context_seq_moe_1d_h45.pt`: available.
- Long processed-data retraining was not run in this pass to avoid overwriting production artifacts. The CLI and dataset paths are implemented and tested.

## 6. Artifact/Metadata State

- `.pt` and `.npz` artifacts remain under `artifacts/models`.
- Deep metadata remains under `artifacts/metadata`.
- Quick-test artifacts live under `artifacts/smoke` and are not production default candidates.
- Example calibration artifact: `artifacts/calibration/motif_CL_F_1d.json`.

## 7. Backtest Results

`CL=F`, `1d`, 5-origin smoke leaderboard:

| Rank | Model | RMSE | MAE | Pinball | Coverage 80 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `drift` | 13.115 | 9.970 | 0.044 | 0.600 |
| 2 | `pattern_mlp` | 14.487 | 11.110 | 0.054 | 0.578 |
| 3 | `random_walk` | 15.210 | 11.730 | 0.057 | 0.560 |
| 4 | `deep_lstm_tcn_fusion` | 16.379 | 13.299 | 0.068 | 0.369 |
| 5 | `llm_context_seq_moe` | 16.976 | 13.832 | 0.071 | 0.324 |
| 6 | `motif` | 17.080 | 13.534 | 0.069 | 0.413 |

In this smoke run, deep models underperformed the baselines. The next step is long retraining with processed fundamentals/news context and a 50+ origin leaderboard.

## 8. Calibration State

- Ran `calibrate_quantiles.py --model motif --symbol CL=F --interval 1d`.
- Current artifact has `n_origins=5`, so `calibration_status=uncalibrated`.
- `/api/forecast` adjusts bands only when a calibration artifact is calibrated; otherwise it keeps the unvalidated warning.

## 9. Dashboard Changes

- Added data pipeline status and LLM context status to `/api/models`.
- `/api/backtests` now prefers the latest leaderboard snapshot.
- Added additive `calibration_status` to `/api/forecast`.
- Added frontend band calibration badge and Model Diagnostics/Leaderboard panel.

## 10. Test Results

Passed:

- `python scripts/maintenance/check_docs_i18n.py --check-legacy`
- `python -m compileall backend market_ai scripts`
- `python scripts/maintenance/smoke_test_api.py`
- `python -m pytest` -> 87 passed
- `node --check frontend/src/main.js`
- yfinance oil_core 1d 5y fetch
- event context build
- LLM local/openai-compatible/local-http dry-runs
- two quick-test deep trainings
- backtest smoke

Skipped:

- `npm run build`: `frontend/node_modules` is absent.
- Real EIA/CFTC/CME live/API ingest: no API key or licensed/manual source files were provided; parsers/unit tests and CLIs validate the path.

## 11. Failures/Skip Reasons

- Long-horizon production retraining was not run because it would risk overwriting existing production artifacts.
- Calibration was not promoted because the run used only 5 smoke origins.

## 12. Next Work

1. Load real EIA/CFTC/CME/manual CSV files.
2. Connect news headline CSVs or an external news provider.
3. Run long processed-data `1d/h45` retraining.
4. Run CL/BZ/NG 50+ origin leaderboards.
5. Generate model-specific conformal calibration from enough residual origins.
6. Expand source-level stale/coverage details in the API/frontend.
