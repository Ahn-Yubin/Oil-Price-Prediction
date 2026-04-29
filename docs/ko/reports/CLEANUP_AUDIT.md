# Cleanup Audit

보호 대상인 model artifact, metadata JSON, docs, tests는 삭제하지 않았습니다. 판단이 불확실한 legacy content는 `_archive/legacy_20260429` 아래로 archive했습니다.

| path | current purpose | imported_by | referenced_by_docs | action | reason | risk | new_path |
|---|---|---|---|---|---|---|---|
| `oil-tv-dashboard/app/main.py` | FastAPI monolith | tests, uvicorn entrypoint | README, reports | move | HTTP route와 domain logic 분리 | high | `backend/app/main.py`, `backend/app/api/routes/*` |
| `oil-tv-dashboard/app/config.py` | Runtime settings | 대부분의 service | README, AGENTS | move | backend dependency 없이 공유되어야 함 | high | `market_ai/config.py`, `backend/app/core/config.py` |
| `oil-tv-dashboard/app/services/forecast_service.py` | Forecast orchestration | API, tests | README | move | Core forecasting은 HTTP layer 밖에 있어야 함 | high | `market_ai/forecasting/service.py` |
| `oil-tv-dashboard/app/services/model_registry.py` | Artifact registry | API, training | README | move | Registry는 model-domain logic | high | `market_ai/modeling/registry.py` |
| `oil-tv-dashboard/app/services/market_data.py` | yfinance loading 및 fallback policy | API, forecast, tests | audit docs | move | Provider logic은 data package 소관 | high | `market_ai/data/providers/yfinance_provider.py` |
| `oil-tv-dashboard/app/services/symbols.py` | Symbol normalization | data services, tests | audit docs | move | Market metadata는 data package 소관 | medium | `market_ai/data/symbols.py` |
| `oil-tv-dashboard/app/services/timeframes.py` | Timeframe normalization | data services, tests | audit docs | move | Timeframe metadata는 data package 소관 | medium | `market_ai/data/timeframes.py` |
| `oil-tv-dashboard/app/services/related_assets.py` | Cross-asset context | forecast service, tests | docs | move | Related asset은 market data context | medium | `market_ai/data/related_assets.py` |
| `oil-tv-dashboard/app/features/*.py` | Feature engineering 및 target transform | forecast, tests | docs | move | Feature는 FastAPI와 독립적이어야 함 | high | `market_ai/features/*` |
| `oil-tv-dashboard/app/forecasters/baselines.py` | Baseline forecaster | forecast, backtest, tests | docs | move | Model code는 modeling package 소관 | high | `market_ai/modeling/forecasters/baselines.py` |
| `oil-tv-dashboard/app/forecasters/moe.py` | Regime ensemble baseline | forecast, tests | docs | move | Regime-aware model code는 modeling package 소관 | high | `market_ai/modeling/regimes/moe.py` |
| `oil-tv-dashboard/app/regime/detector.py` | Regime detection | forecast, tests | docs | move | Regime detection은 model-domain logic | medium | `market_ai/modeling/regimes/detector.py` |
| `oil-tv-dashboard/app/global_dl_model.py` | `.npz` neural forecaster | forecast, train, backtest | README | move | Model implementation은 modeling package 소관 | high | `market_ai/modeling/forecasters/neural_npz.py` |
| `oil-tv-dashboard/app/market_pattern_model.py` | Motif/pattern forecast comparison | forecast service | docs | move | Model implementation은 modeling package 소관 | high | `market_ai/modeling/forecasters/motif.py` |
| `oil-tv-dashboard/app/llm/event_encoder.py` | LLM context encoder | API, tests | LLM docs | move | LLM context logic은 `market_ai.llm` 소관 | medium | `market_ai/llm/event_encoder.py` |
| `oil-tv-dashboard/app/llm/context_schema.py` | LLM context schema | LLM encoder, tests | LLM docs | move | Schema는 shared schema package 소관 | medium | `market_ai/schemas/llm_context.py` |
| `oil-tv-dashboard/app/schemas/market.py` | Market/API schema model | API, services, tests | API docs | move | Shared schema는 backend 밖에 있어야 함 | high | `market_ai/schemas/market.py` |
| `oil-tv-dashboard/app/models/*.npz` | Model weights | model registry, neural forecaster | README | move | Artifact는 app code 아래에 두지 않음 | high | `artifacts/models/*.npz` |
| `oil-tv-dashboard/app/models/*.json` | Model metadata | model registry | README | move | Metadata는 weight와 분리해 versioning | high | `artifacts/metadata/*.json` |
| `oil-tv-dashboard/train_pretrained_models.py` | Training CLI | docs, operator | README | move | 직접 실행 script는 scripts 소관 | medium | `scripts/train/train_pretrained_models.py` |
| `oil-tv-dashboard/backtest_forecasters.py` | Backtest 구현 및 CLI | tests, docs | README | move | Core backtest는 `market_ai`, CLI는 scripts | high | `market_ai/backtesting/runner.py`, `scripts/backtest/run_backtest.py` |
| `oil-tv-dashboard/app/static/*` | Chart frontend assets | frontend | README | move | UI asset은 frontend 소관 | medium | `frontend/src/*` |
| `oil-tv-dashboard/app/templates/index.html` | Dashboard HTML | backend static server | README | move | UI shell은 frontend 소관 | medium | `frontend/index.html` |
| `oil-tv-dashboard/tests/*.py` | Tests | pytest | reports | move | Root test layout으로 정리 | low | `tests/unit/*`, `tests/integration/*` |
| `oil-tv-dashboard/docs/*.md` | English docs | docs references | README | move | English docs는 `docs/en` 소관 | low | `docs/en/*.md` |
| `oil-tv-dashboard/*_REPORT.md`, `ARCHITECTURE_AUDIT.md`, `IMPLEMENTATION_PLAN.md` | Generated reports | docs | README | move | Report는 root에 두지 않음 | low | `docs/en/reports/*`, `docs/ko/reports/*` |
| `oil-tv-dashboard/outputs/backtests/*` | Generated backtest outputs | API `/api/backtests` | reports | keep | 이동 후 `outputs` 아래에 격리됨 | low | `outputs/backtests/*` |
| `oil-price-baseline/` | 과거 oil baseline experiment | 새 import graph 없음 | old audit | archive | Dirty worktree이며 legacy 가치 불확실, 삭제 금지 | medium | `_archive/legacy_20260429/oil-price-baseline` |
| `oil-tv-dashboard/` remnants | 비어 있거나 cache만 남은 old project shell | none | none | archive | migration 후 남은 잔여물, 삭제 대신 archive | low | `_archive/legacy_20260429/oil-tv-dashboard-remnants` |
| `app/main.py` | Legacy uvicorn wrapper | optional old command | README history | keep | `app.main:app` compatibility shim | low | unchanged |
| `.DS_Store`, `__pycache__`, `.pytest_cache` remnants | Local/generated files | none | none | archive | destructive delete 대신 legacy archive에 포함 | low | `_archive/legacy_20260429/*` |
