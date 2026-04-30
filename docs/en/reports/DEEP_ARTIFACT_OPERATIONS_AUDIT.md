# Deep Artifact Operations Audit

Date: 2026-04-30

## Summary

The repository does not currently contain the `1d/h45` deep artifacts used by the dashboard default request. It only contains quick-test `1d/h8` `.pt` artifacts under `artifacts/models`, and their metadata status is `available`. As a result, `/api/models` reports the deep models as available, while `/api/forecast` tries to load `1d/h45` artifacts, fails, and surfaces artifact unavailable warnings on the dashboard first load.

## Artifact Inventory

`artifacts/models`:

- `global_dl_1d_h45.npz`
- `global_dl_1h_h72.npz`
- `global_dl_30m_h120.npz`
- `global_dl_15m_h192.npz`
- `deep_lstm_tcn_fusion_1d_h8.pt`
- `llm_context_seq_moe_1d_h8.pt`

`artifacts/metadata`:

- `global_dl_1d_h45.json`
- `global_dl_1h_h72.json`
- `global_dl_30m_h120.json`
- `global_dl_15m_h192.json`
- `deep_lstm_tcn_fusion_1d_h8.json`
- `llm_context_seq_moe_1d_h8.json`

## API Behavior

Current `/api/models` behavior:

- `ModelRegistry.scan()` scans both `.npz` and `.pt` artifacts.
- `user_facing_models` marks a deep model as `available` if any artifact with the same `model_name` exists.
- Therefore `deep_lstm_tcn_fusion_1d_h8.pt` and `llm_context_seq_moe_1d_h8.pt` make the deep models appear available even though the default dashboard artifact is `1d/h45`.

Current default `/api/forecast` behavior:

- When the models query is absent, the full `USER_FACING_MODELS` tuple is used as the default selection.
- The service tries `deep_lstm_tcn_fusion` and `llm_context_seq_moe`.
- The default `1d` horizon is 45, so it looks for `deep_lstm_tcn_fusion_1d_h45.pt` and `llm_context_seq_moe_1d_h45.pt`.
- Those files are missing, so `artifact_status` becomes `missing_or_unavailable` and string warnings are added.

Current `/api/forecast?models=deep_lstm_tcn_fusion` behavior:

- The endpoint returns status code 200.
- It returns a deep artifact missing warning and internally falls back to non-deep models such as `motif`.
- The warning does not include the training command needed to create the missing artifact.

## Quick-Test Artifacts

All current deep `.pt` artifacts are `1d/h8`.

- `deep_lstm_tcn_fusion_1d_h8.json`: `horizon=8`, `lookback=32`, `epochs_ran=1`, `status=available`
- `llm_context_seq_moe_1d_h8.json`: `horizon=8`, `lookback=32`, `epochs_ran=1`, `status=available`

These artifacts are not used by the default dashboard `1d/h45` request. Their `available` metadata status makes them look like production artifacts.

## Training CLI

`scripts/train/train_deep_fusion_models.py` defines `--events-path`, but it does not build `FileEventProvider(paths=[...])` and pass it into `build_deep_dataset_from_frame(..., event_provider=provider)`. The dataset builder currently uses only env-based `FileEventProvider.from_env()` when `config.event_context_enabled` is true.

When all yfinance downloads fail, `build_dataset()` silently falls back to `build_synthetic_deep_dataset()` and can create an artifact with `source=synthetic_fallback`. This is unsafe for production training.

## Documentation Consistency

`docs/ko/PROJECT_STATUS.md` and `docs/en/PROJECT_STATUS.md` contain these contradictions:

- The top section classifies `cycle`, `lstm`, `tcn`, and `ensemble` as removed/deprecated, but later active model tables and frontend descriptions still describe them as comparison models.
- The OpenAI-compatible LLM adapter is described as a placeholder with no external call, but `OpenAICompatibleLLMEventEncoder` can call chat completions through `urllib` when `ENABLE_EXTERNAL_LLM_CALLS=true` and `LLM_API_KEY` are set.
- Cross-asset features are still mostly missing-indicator placeholders, but the docs can read as if the full feature matrix is complete.
- Deep quick training artifacts can be misread as production performance artifacts.

## Frontend Warning UX

The frontend joins `/api/forecast` `warnings: list[str]` into one string and displays it in `status-banner`. There is no severity distinction, so artifact missing, stale data, and uncalibrated quantile messages all appear in the same yellow box.

## Root Causes

1. Deep artifact availability checks only `model_name` existence, not horizon or metadata status.
2. Default forecast selection always attempts deep models even when their artifact is missing.
3. Quick-test h8 metadata is marked `available`, so smoke artifacts are not separated from production artifacts.
4. The warning contract is string-only, so the frontend cannot render severity-specific UX.
5. The training CLI does not enforce production-safe explicit event paths and synthetic fallback behavior.
