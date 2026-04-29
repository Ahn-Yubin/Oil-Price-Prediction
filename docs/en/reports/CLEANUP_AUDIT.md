# Cleanup Audit

No protected artifact, metadata JSON, documentation, or test file was deleted. Uncertain legacy content was archived under `_archive/legacy_20260429`.

| path | current purpose | imported_by | referenced_by_docs | action | reason | risk | new_path |
|---|---|---|---|---|---|---|---|
| `oil-tv-dashboard/app/main.py` | FastAPI monolith | tests, uvicorn entrypoint | README, reports | move | Split HTTP routes from domain logic | high | `backend/app/main.py`, `backend/app/api/routes/*` |
| `oil-tv-dashboard/app/config.py` | Runtime settings | most services | README, AGENTS | move | Settings must be portable and shared without backend dependency | high | `market_ai/config.py`, `backend/app/core/config.py` |
| `oil-tv-dashboard/app/services/forecast_service.py` | Forecast orchestration | API, tests | README | move | Core forecasting belongs outside HTTP layer | high | `market_ai/forecasting/service.py` |
| `oil-tv-dashboard/app/services/model_registry.py` | Artifact registry | API, training | README | move | Registry is model-domain logic | high | `market_ai/modeling/registry.py` |
| `oil-tv-dashboard/app/services/market_data.py` | yfinance loading and fallback policy | API, forecast, tests | audit docs | move | Provider logic belongs in data package | high | `market_ai/data/providers/yfinance_provider.py` |
| `oil-tv-dashboard/app/services/symbols.py` | Symbol normalization | data services, tests | audit docs | move | Market metadata belongs in data package | medium | `market_ai/data/symbols.py` |
| `oil-tv-dashboard/app/services/timeframes.py` | Timeframe normalization | data services, tests | audit docs | move | Timeframe metadata belongs in data package | medium | `market_ai/data/timeframes.py` |
| `oil-tv-dashboard/app/services/related_assets.py` | Cross-asset context | forecast service, tests | docs | move | Related assets are market data context | medium | `market_ai/data/related_assets.py` |
| `oil-tv-dashboard/app/features/*.py` | Feature engineering and target transforms | forecast, tests | docs | move | Features must be FastAPI-independent | high | `market_ai/features/*` |
| `oil-tv-dashboard/app/forecasters/baselines.py` | Baseline forecasters | forecast, backtest, tests | docs | move | Model code belongs in modeling package | high | `market_ai/modeling/forecasters/baselines.py` |
| `oil-tv-dashboard/app/forecasters/moe.py` | Regime ensemble baseline | forecast, tests | docs | move | Regime-aware model code belongs in modeling package | high | `market_ai/modeling/regimes/moe.py` |
| `oil-tv-dashboard/app/regime/detector.py` | Regime detection | forecast, tests | docs | move | Regime detection is model-domain logic | medium | `market_ai/modeling/regimes/detector.py` |
| `oil-tv-dashboard/app/global_dl_model.py` | `.npz` neural forecaster | forecast, train, backtest | README | move | Model implementation belongs in modeling package | high | `market_ai/modeling/forecasters/neural_npz.py` |
| `oil-tv-dashboard/app/market_pattern_model.py` | Motif/pattern forecast comparison | forecast service | docs | move | Model implementation belongs in modeling package | high | `market_ai/modeling/forecasters/motif.py` |
| `oil-tv-dashboard/app/llm/event_encoder.py` | LLM context encoder | API, tests | LLM docs | move | LLM context logic belongs in `market_ai.llm` | medium | `market_ai/llm/event_encoder.py` |
| `oil-tv-dashboard/app/llm/context_schema.py` | LLM context schema | LLM encoder, tests | LLM docs | move | Schema belongs in shared schema package | medium | `market_ai/schemas/llm_context.py` |
| `oil-tv-dashboard/app/schemas/market.py` | Market/API schema models | API, services, tests | API docs | move | Shared schema belongs outside backend | high | `market_ai/schemas/market.py` |
| `oil-tv-dashboard/app/models/*.npz` | Model weights | model registry, neural forecaster | README | move | Artifacts must not live under app code | high | `artifacts/models/*.npz` |
| `oil-tv-dashboard/app/models/*.json` | Model metadata | model registry | README | move | Metadata is versioned separately from weights | high | `artifacts/metadata/*.json` |
| `oil-tv-dashboard/train_pretrained_models.py` | Training CLI | docs, operator | README | move | Human-run entrypoints belong in scripts | medium | `scripts/train/train_pretrained_models.py` |
| `oil-tv-dashboard/backtest_forecasters.py` | Backtest implementation and CLI | tests, docs | README | move | Core backtest logic belongs in `market_ai`; CLI wrapper in scripts | high | `market_ai/backtesting/runner.py`, `scripts/backtest/run_backtest.py` |
| `oil-tv-dashboard/app/static/*` | Chart frontend assets | frontend | README | move | UI assets belong in frontend | medium | `frontend/src/*` |
| `oil-tv-dashboard/app/templates/index.html` | Dashboard HTML | backend static server | README | move | UI shell belongs in frontend | medium | `frontend/index.html` |
| `oil-tv-dashboard/tests/*.py` | Tests | pytest | reports | move | Tests belong in root test layout | low | `tests/unit/*`, `tests/integration/*` |
| `oil-tv-dashboard/docs/*.md` | English docs | docs references | README | move | English docs belong in `docs/en` | low | `docs/en/*.md` |
| `oil-tv-dashboard/*_REPORT.md`, `ARCHITECTURE_AUDIT.md`, `IMPLEMENTATION_PLAN.md` | Generated reports | docs | README | move | Reports must not stay at root | low | `docs/en/reports/*`, `docs/ko/reports/*` |
| `oil-tv-dashboard/outputs/backtests/*` | Generated backtest outputs | API `/api/backtests` | reports | keep | Already isolated under `outputs` after move | low | `outputs/backtests/*` |
| `oil-price-baseline/` | Older oil baseline experiment | none in new import graph | old audit | archive | Dirty worktree and uncertain legacy value; do not delete | medium | `_archive/legacy_20260429/oil-price-baseline` |
| `oil-tv-dashboard/` remnants | Empty/cached old project shell | none | none | archive | Contains only remnants after migration; archive instead of delete | low | `_archive/legacy_20260429/oil-tv-dashboard-remnants` |
| `app/main.py` | Legacy uvicorn wrapper | optional old command | README history | keep | Backward-compatible `app.main:app` shim | low | unchanged |
| `.DS_Store`, `__pycache__`, `.pytest_cache` remnants | Local/generated files | none | none | archive | Destructive delete was avoided; remnants moved with legacy archive | low | `_archive/legacy_20260429/*` |
