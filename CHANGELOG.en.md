# Changelog

## 2026-05-10

- Documented EIA bulk, CFTC ZIP/manual CSV, and CME manual CSV data paths.
- Reflected public news and LLM/event context generation in operational docs.
- Added documentation for `/api/market-context` and frontend context markers/news/scenario panel.
- Documented Google Gemma/Gemini OpenAI-compatible LLM setup, `export` checks, and `.env` usage.
- Folded stale report Markdown into canonical docs and deleted the report files.
- Updated the Korean/English mirrors for `README`, `PROJECT_STATUS`, `DATA_PIPELINE`, `OPERATIONS`, `LLM_CONTEXT`, `API`, `FRONTEND`, `MODEL_DESIGN`, and `ROADMAP`.

## 2026-04-29

- Clarified the documentation policy as Korean source + English mirror.
- Expanded `README.md` as the Korean primary README and aligned `README.en.md` as the English mirror.
- Updated `AGENTS.md` with both Korean and English project instructions.
- Improved `scripts/maintenance/check_docs_i18n.py` to compare all relative Markdown paths under `docs/ko` and `docs/en`.
- Reorganized the repository into `backend`, `frontend`, `market_ai`, `scripts`, `docs`, `data`, `artifacts`, `outputs`, `notebooks`, and `tests`.
- Moved FastAPI entrypoint to `backend.app.main:app` while keeping `app.main:app` as a thin compatibility wrapper.
- Moved `.npz` model artifacts to `artifacts/models` and metadata JSON to `artifacts/metadata`.
- Moved training and backtest CLIs to `scripts/train/train_pretrained_models.py` and `scripts/backtest/run_backtest.py`.
- Added maintenance scripts for docs parity, unused-file audit, and API smoke testing.
