# Deep Artifact Operations Report

Date: 2026-04-30

## 1. Change Summary

- Added a shared deep artifact availability policy based on filename, interval, horizon, and metadata status.
- Extended `/api/models` with `expected_artifact_file`, `expected_metadata_file`, `training_command`, and `status`.
- Default `/api/forecast` no longer tries deep models with missing artifacts.
- Explicit deep model requests with missing artifacts return 200 fallback plus an actionable warning object.
- Preserved `warnings: list[str]` and added `warning_objects` as an optional additive field.
- The frontend now renders `severity=info` as a small info badge rather than a yellow warning banner.
- Connected training CLI `--events-path` to the actual `FileEventProvider`.
- Stopped silent synthetic fallback for production training after yfinance failure.
- Separated quick-test artifacts into `artifacts/smoke` and records them as `status=smoke_only`.
- Updated Korean/English `PROJECT_STATUS.md` docs for current model, LLM, and deep artifact status.

## 2. Artifact Availability Policy

A production deep artifact is available only when all conditions are met:

- The expected artifact file exists.
- The expected metadata file exists or artifact metadata can be read.
- Metadata `status` is `available`.
- Metadata with `synthetic_used=true` is not treated as production available.

`smoke_only`, `synthetic_only`, `failed`, and `metadata_only` are excluded from default `/api/forecast` candidates.

## 3. Training CLI Changes

Changed files:

- `scripts/train/train_deep_fusion_models.py`
- `market_ai/data/deep_dataset.py` already accepted an `event_provider` and continues to do so.
- `market_ai/data/event_providers.py` did not require changes; comma-separated paths are handled in the CLI.

Main changes:

- Supports `--events-path a.csv,b.json`
- Preserves `NEWS_EVENTS_PATH`, `ECONOMIC_EVENTS_PATH`, `MARKET_EVENTS_PATH` env fallback
- Adds `--allow-synthetic-fallback`
- Allows synthetic smoke training with `--quick-test`
- Supports explicit synthetic data with `--synthetic`
- Writes quick-test output to `artifacts/smoke/models` and `artifacts/smoke/metadata`
- Adds required operational metadata fields

## 4. `--events-path` Fix

Fixed. The CLI now creates `FileEventProvider(paths=[...])` and passes it into `build_deep_dataset_from_frame(..., event_provider=provider)`.

`tests/unit/test_train_deep_fusion_cli_policy.py` verifies that an event CSV changes the actual event vector.

## 5. Synthetic Fallback Policy

Current policy:

- Default production training fails if yfinance produces no usable samples.
- `--allow-synthetic-fallback` permits synthetic fallback after yfinance failure.
- `--synthetic` uses a synthetic dataset from the start.
- `--quick-test` permits a synthetic smoke dataset.
- Synthetic and smoke artifacts are not exposed as production available.

## 6. Generated Artifacts/Metadata

Production artifacts:

- `artifacts/models/deep_lstm_tcn_fusion_1d_h45.pt`
- `artifacts/metadata/deep_lstm_tcn_fusion_1d_h45.json`
- `artifacts/models/llm_context_seq_moe_1d_h45.pt`
- `artifacts/metadata/llm_context_seq_moe_1d_h45.json`

Production metadata summary:

- `interval=1d`, `horizon=45`, `lookback=128`
- `data_source=yfinance`
- `synthetic_used=false`
- `status=available`
- `n_train=3584`, `n_val=768`, `n_test=768`
- `training_cutoff=2026-02-24T00:00:00+00:00`

Smoke artifacts:

- `artifacts/smoke/models/deep_lstm_tcn_fusion_1d_h8.pt`
- `artifacts/smoke/metadata/deep_lstm_tcn_fusion_1d_h8.json`
- `artifacts/smoke/models/llm_context_seq_moe_1d_h8.pt`
- `artifacts/smoke/metadata/llm_context_seq_moe_1d_h8.json`

Existing `artifacts/metadata/*_1d_h8.json` files were also changed to `status=smoke_only`, `synthetic_used=true`.

`.pt` artifacts are not commit candidates under `.gitignore`. Metadata JSON files describe operational artifact state and are commit candidates.

## 7. `/api/models` Summary

Deep model status from `/api/models`:

- `deep_lstm_tcn_fusion`: `status=available`, `expected_artifact_file=deep_lstm_tcn_fusion_1d_h45.pt`
- `llm_context_seq_moe`: `status=available`, `expected_artifact_file=llm_context_seq_moe_1d_h45.pt`

Each deep model also returns a `training_command`.

## 8. `/api/forecast` Warning Changes

Default request:

- Deep models are included when their h45 artifacts are available. Before artifact creation, missing deep models were silently excluded and recorded only in `artifact_status`.
- After h45 artifact creation, the primary model is `deep_lstm_tcn_fusion`.
- Missing deep artifact warnings are not shown on the default dashboard load.

Explicit deep request:

- If the artifact is missing, the response returns 200 fallback and `warning_objects[].code=deep_artifact_unavailable`.
- The warning action contains the training command.

Quantile warning:

- The legacy warning string is preserved.
- In `warning_objects`, it uses `severity=info` and `code=quantile_bands_uncalibrated`.

## 9. `PROJECT_STATUS.md` Update Summary

Both Korean and English mirrors now reflect:

- Active model list: `motif`, `pattern_mlp`, `deep_lstm_tcn_fusion`, `llm_context_seq_moe`, and baselines
- `cycle`, `lstm`, `tcn`, and `ensemble` only as removed/deprecated models
- External LLM call conditions aligned with the current code
- Deep models are code-complete but depend on artifact availability
- Quick-test h8 artifacts are not used by dashboard default h45 requests
- Warning severity policy and updated next work sequence

## 10. Backtest Smoke Result

Command:

```bash
.venv/bin/python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 5 --models random_walk,drift,motif,pattern_mlp,deep_lstm_tcn_fusion,llm_context_seq_moe --no-plots
```

Result:

- Run succeeded.
- `outputs/backtests/latest_model_availability.csv` was created.
- `/api/backtests?symbol=CL=F&interval=1d` returns `model_availability`.
- Both deep models are `available`, `origins_ok=5`, `origins_error=0`.

## 11. Test Results

Passed:

- `.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy`
- `.venv/bin/python -m compileall backend market_ai scripts`
- `.venv/bin/python scripts/maintenance/smoke_test_api.py`
- `.venv/bin/python -m pytest` (`80 passed`)
- `node --check frontend/src/main.js`
- 2 quick smoke training runs
- 2 full `1d/h45` training runs
- deep backtest smoke

## 12. Failures/Skips

- The `python` executable was unavailable, so commands were run with `.venv/bin/python`.
- `npm run build` was skipped because `frontend/node_modules` does not exist.
- `ruff` and `mypy` were skipped because executables or configuration were unavailable.

## 13. Next Work

1. Retrain the generated h45 artifacts with longer epochs and broader validation.
2. Extend the deep leaderboard across more symbols and intervals.
3. Connect quantile coverage calibration to rolling backtest outputs.
4. Build real event ingestion beyond the sample event file.
5. Expand the cross-asset feature matrix with real related asset values.
6. Add a frontend model diagnostics panel.
7. Add provider cache/storage.
