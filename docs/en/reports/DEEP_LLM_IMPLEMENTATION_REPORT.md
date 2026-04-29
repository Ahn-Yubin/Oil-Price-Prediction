# Deep + LLM Context Implementation Report

Date: 2026-04-30

## 1. Implementation Summary

Added the LSTM + TCN + event/LLM context deep sequence path and made the `/api/forecast` `models` query affect actual model selection. Legacy `cycle`, live `lstm`, live `tcn`, and fixed `ensemble` were removed from the default API, frontend selector, and default backtest list.

## 2. Removed/Replaced Models

- `cycle`: standalone forecast removed. Cycle signals moved to `cycle_strength`, `cycle_phase_sin`, and `cycle_phase_cos` features.
- `lstm`: live cached request-time model removed. Replaced by the LSTM encoder in `deep_lstm_tcn_fusion`.
- `tcn`: live cached request-time model removed. Replaced by the TCN encoder in `deep_lstm_tcn_fusion`.
- `ensemble`: fixed-weight mix removed. Replaced by learned `llm_context_seq_moe`.

## 3. Kept Models

- `motif`: historical analogue and explainability model.
- `pattern_mlp`: legacy `.npz` artifact fallback.
- `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive`: user-facing baselines.
- `flat`, `simple_moving_average_path`, optional `regime_ensemble`: backtest-only.

## 4. New Models

- `deep_lstm_tcn_fusion`: causal LSTM encoder, causal TCN encoder, fusion gate, FiLM-style context conditioning, and quantile/direction/volatility/confidence heads.
- `llm_context_seq_moe`: LSTM expert, TCN expert, baseline adapter, motif adapter, and event/static-context gating network.

## 5. Added Files

- `market_ai/modeling/deep/*`
- `market_ai/modeling/forecasters/deep_fusion.py`
- `market_ai/data/deep_dataset.py`
- `market_ai/data/event_providers.py`
- `market_ai/data/symbol_universe.py`
- `market_ai/features/deep_features.py`
- `market_ai/features/context_features.py`
- `market_ai/schemas/deep_learning.py`
- `scripts/train/train_deep_fusion_models.py`
- `configs/symbol_universe.yaml`
- `data/external/events/sample_market_events.csv`

## 6. Modified Files

- `market_ai/forecasting/service.py`: model selection, deep fallback, additive response fields.
- `market_ai/modeling/registry.py`: `.pt` artifact scanning.
- `market_ai/backtesting/runner.py`: cleaned model list and availability reporting.
- `backend/app/api/routes/forecast.py`, `chart.py`, `models.py`: model query and cleanup policy.
- `frontend/index.html`, `frontend/src/main.js`: model selector and query forwarding.
- `market_ai/llm/event_encoder.py`: local encoder, optional external adapter, safety validator.

## 7. Deleted/Archived Files

No files were moved to archive. Legacy wrappers `cycle.py` and `ensemble.py` now return explicit removal errors, and live LSTM/TCN plus fixed ensemble logic were removed from active default paths.

## 8. New Model Structure

Both models output volatility-scaled cumulative log return quantiles. Prices are reconstructed only with `current_price * exp(cumulative_log_return_h)`. LLM context affects gating and confidence only; it cannot directly create numeric paths.

## 9. Data Pipeline

The deep dataset builder creates price features, cross-asset placeholder/missing indicators, event context vectors, and static features. The target is `future cumulative log return / recent_realized_volatility`. Splits are time-based and do not use random splitting.

## 10. LLM Context Handling

`LocalEventContextEncoder` deterministically reads CSV/JSON event files. `OpenAICompatibleLLMEventEncoder` calls externally only when `ENABLE_EXTERNAL_LLM_CALLS=true` and a key exists. Forbidden numeric forecast fields are ignored with warnings.

## 11. Training

```bash
python scripts/train/train_deep_fusion_models.py --model both --interval 1d --universe oil_core --epochs 10 --batch-size 64
python scripts/train/train_deep_fusion_models.py --model deep_lstm_tcn_fusion --interval 1d --quick-test --epochs 1 --max-samples 256
```

Quick synthetic training generated `deep_lstm_tcn_fusion_1d_h8` and `llm_context_seq_moe_1d_h8` metadata.

## 12. Backtesting

```bash
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 10 --models random_walk,drift,motif,pattern_mlp,deep_lstm_tcn_fusion,llm_context_seq_moe --no-plots
```

If a deep artifact is missing, model availability records it as unavailable and the overall backtest continues.

## 13. API Changes

Added additive `ForecastResponse` fields: `model_paths`, `selected_models`, `primary_model`, `deprecated_models_requested`, `removed_models_requested`, `llm_context_summary`, `deep_model_info`, `feature_version`, and `artifact_status`.

`/api/forecast` returns 400 for unknown/removed models. `/api/chart` keeps the existing schema and accepts compatibility queries.

## 14. Frontend Changes

Added a model selector and forwarded `/api/forecast?models=...`. Removed models are not displayed. LLM context enabled/disabled state is reflected in the model display text.

## 15. Test Results

- `.venv/bin/python -m compileall backend market_ai scripts`: passed
- `.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy`: passed
- `.venv/bin/python scripts/maintenance/smoke_test_api.py`: passed
- `.venv/bin/python -m pytest`: 75 passed
- Deep quick training: `deep_lstm_tcn_fusion` and `llm_context_seq_moe` each passed one synthetic epoch

## 16. Failures/Skips

- The local shell does not provide `python`, so `.venv/bin/python` was used.
- `npm run build` was not run because `frontend/node_modules` is absent.
- `ruff` and `mypy` executables are absent.
- The network-dependent yfinance backtest CLI was not run as an optional gate.

## 17. Remaining Risks

- Quick synthetic artifacts are smoke artifacts, not evidence of production performance.
- Default 1d deep horizon artifacts still need full training.
- Cross-asset features currently fall back mostly to missing indicators.
- Coverage calibration is not yet sufficiently accumulated in model metadata.

## 18. Recommended Next Work

1. Run full interval training for the `oil_core` universe and save `.pt` artifacts.
2. Recompute backtest leaderboards for trained deep artifacts.
3. Connect event ingestion to real operational feeds while preserving no-lookahead checks.
4. Expand cross-asset alignment into a real related-asset matrix.
