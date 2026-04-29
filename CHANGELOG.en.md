# Changelog

## 2026-04-29

- Clarified the documentation policy as Korean source + English mirror.
- Expanded `README.md` as the Korean primary README and aligned `README.en.md` as the English mirror.
- Updated `AGENTS.md` with both Korean and English project instructions.
- Added `docs/ko/reports/DOCS_AUDIT.md` and `docs/en/reports/DOCS_AUDIT.md`.
- Improved `scripts/maintenance/check_docs_i18n.py` to compare all relative Markdown paths under `docs/ko` and `docs/en`.
- Reorganized the repository into `backend`, `frontend`, `market_ai`, `scripts`, `docs`, `data`, `artifacts`, `outputs`, `notebooks`, and `tests`.
- Moved FastAPI entrypoint to `backend.app.main:app` while keeping `app.main:app` as a thin compatibility wrapper.
- Moved `.npz` model artifacts to `artifacts/models` and metadata JSON to `artifacts/metadata`.
- Moved training and backtest CLIs to `scripts/train/train_pretrained_models.py` and `scripts/backtest/run_backtest.py`.
- Added maintenance scripts for docs parity, unused-file audit, and API smoke testing.
