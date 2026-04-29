# Final Implementation Report

## 2026-04-29 Order Recovery Audit Update

This pass audited and corrected inconsistencies that could have resulted from performing the project cleanup before Markdown i18n.

### Recovery Summary

- Added `docs/ko/reports/ORDER_RECOVERY_AUDIT.md` and `docs/en/reports/ORDER_RECOVERY_AUDIT.md`.
- Hardened `scripts/maintenance/check_docs_i18n.py` to validate `AGENTS.md`, root-level generated report clutter, relative-path pairs under `docs/ko` and `docs/en`, and optional legacy string scanning.
- Hardened `scripts/maintenance/smoke_test_api.py` to classify dependency-related 503 responses as expected 503.
- Added the `ORDER_RECOVERY_AUDIT.md` pair to `DOCS_AUDIT.md`.
- Verified that model artifacts and metadata are preserved under `artifacts/models` and `artifacts/metadata`, matching config/docs/API descriptions.
- Verified that `market_ai` does not import `backend`.

### Latest Verification Results

- `python scripts/maintenance/check_docs_i18n.py`: failed because `python` is not installed in the current shell.
- `python3 scripts/maintenance/check_docs_i18n.py`: passed. Required root docs: 5, root pairs: 2, `docs/ko`: 15 files, `docs/en`: 15 files.
- `.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy`: passed. Only 22 historical report references were allowed.
- `python -m compileall backend market_ai scripts`: failed because `python` is not installed in the current shell.
- `python3 -m compileall backend market_ai scripts`: passed.
- `.venv/bin/python -m compileall backend market_ai scripts`: passed.
- `python scripts/maintenance/smoke_test_api.py`: failed because `python` is not installed in the current shell.
- `python3 scripts/maintenance/smoke_test_api.py`: failed because the system `python3` environment does not have the `numpy` module.
- `.venv/bin/python scripts/maintenance/smoke_test_api.py`: passed. All required endpoints returned 200.
- `.venv/bin/python scripts/maintenance/audit_unused_files.py`: passed and emitted a Markdown audit table.
- `pytest`: failed because `pytest` is not on the current shell PATH.
- `python3 -m pytest`: failed because the system `python3` environment does not have the `pytest` module.
- `.venv/bin/python -m pytest`: passed, 39 tests.
- `npm test` from `frontend/`: failed because `npm` is not installed in the current shell.
- `npm run build` from `frontend/`: failed because `npm` is not installed in the current shell.
- `npm run lint` from `frontend/`: failed because `npm` is not installed in the current shell.
- `ruff check .`: failed because `ruff` is not installed in the current shell.
- `mypy`: failed because `mypy` is not installed in the current shell.

## 2026-04-29 Documentation Cleanup Update

This pass focused on Markdown documentation, without large functional code moves. The documentation policy is now Korean source + English mirror.

### Documentation Summary

- Expanded `README.md` as the Korean primary README and aligned `README.en.md` as the English mirror with the same structure.
- Updated `AGENTS.md` with both Korean and English instructions for Codex.
- Kept the core `docs/ko` and `docs/en` structures aligned.
- Added `docs/ko/reports/DOCS_AUDIT.md` and `docs/en/reports/DOCS_AUDIT.md`.
- Improved `scripts/maintenance/check_docs_i18n.py` from a fixed-list check to validation of required root docs, root report clutter, and full relative-path comparison between `docs/ko` and `docs/en`.
- Recorded historical Markdown under `_archive` as obsolete/duplicate docs in the audit, not as active documentation.

### Documentation Verification Results

- `python scripts/maintenance/check_docs_i18n.py`: failed because `python` is not installed in the current shell.
- `python3 scripts/maintenance/check_docs_i18n.py`: passed. Required root docs: 5, root pairs: 2, `docs/ko`: 15 files, `docs/en`: 15 files.
- `.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy`: passed. Only historical report references are allowed.
- `python -m compileall backend market_ai scripts`: failed because `python` is not installed in the current shell.
- `python3 -m compileall backend market_ai scripts`: passed.
- `.venv/bin/python -m compileall backend market_ai scripts`: passed.
- `.venv/bin/python scripts/maintenance/smoke_test_api.py`: passed; all required endpoints returned 200.
- `pytest`: failed because `pytest` is not installed on the shell PATH.
- `python3 -m pytest`: failed because the system `python3` environment does not have the `pytest` module.
- `.venv/bin/python -m pytest`: passed, 39 tests.

## Summary

The repository is now organized as a Universal Market Forecasting Platform. FastAPI code lives in `backend`, market AI logic in `market_ai`, frontend assets in `frontend`, CLI entrypoints in `scripts`, model weights in `artifacts/models`, metadata JSON in `artifacts/metadata`, and generated outputs in `outputs`.

## Moved Files

- `oil-tv-dashboard/app/main.py` -> `backend/app/main.py` plus `backend/app/api/routes/*`.
- `oil-tv-dashboard/app/config.py` -> `market_ai/config.py` and `backend/app/core/config.py`.
- `oil-tv-dashboard/app/services/*` -> `market_ai/forecasting`, `market_ai/data`, and `market_ai/modeling`.
- `oil-tv-dashboard/app/features/*` -> `market_ai/features/*`.
- `oil-tv-dashboard/app/forecasters/*` and `app/regime/*` -> `market_ai/modeling/*`.
- `oil-tv-dashboard/app/llm/*` -> `market_ai/llm` and `market_ai/schemas/llm_context.py`.
- `oil-tv-dashboard/app/models/*.npz` -> `artifacts/models/*.npz`.
- `oil-tv-dashboard/app/models/*.json` -> `artifacts/metadata/*.json`.
- `oil-tv-dashboard/train_pretrained_models.py` -> `scripts/train/train_pretrained_models.py`.
- `oil-tv-dashboard/backtest_forecasters.py` -> `market_ai/backtesting/runner.py` with `scripts/backtest/run_backtest.py` wrapper.
- `oil-tv-dashboard/app/static/*` and `app/templates/index.html` -> `frontend/`.
- Tests moved to `tests/unit` and `tests/integration`.
- Reports moved to `docs/en/reports` and `docs/ko/reports`.

## Deleted Files

None. Destructive deletion was avoided.

## Archived Files

- `oil-price-baseline/` -> `_archive/legacy_20260429/oil-price-baseline`.
- Remaining `oil-tv-dashboard/` shell/cache remnants -> `_archive/legacy_20260429/oil-tv-dashboard-remnants`.
- Root `.DS_Store` -> `_archive/legacy_20260429/root.DS_Store`.

## Legacy Compatibility

- `app/main.py` remains as a thin wrapper for `backend.app.main:app`.
- `GET /api/chart` preserves the legacy chart payload keys.

## New Directory Structure

Top-level active directories: `backend`, `frontend`, `market_ai`, `scripts`, `configs`, `docs`, `data`, `artifacts`, `outputs`, `notebooks`, `tests`, and `_archive`.

## Tests Run

- `python -m compileall backend market_ai scripts`: unavailable because `python` is not installed.
- `python3 -m compileall backend market_ai scripts`: passed.
- `.venv/bin/python -m compileall backend market_ai scripts`: passed.
- `.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy`: passed.
- `.venv/bin/python scripts/maintenance/audit_unused_files.py`: passed and emitted a Markdown audit table.
- `.venv/bin/python scripts/maintenance/smoke_test_api.py`: passed; all required endpoints returned 200.
- `.venv/bin/python -m pytest`: passed, 39 tests.
- `npm test` from `frontend/`: failed because `npm` is not installed in this shell.
- `npm run build` from `frontend/`: failed because `npm` is not installed in this shell.
- `ruff check .`: failed because `ruff` is not installed in this shell.
- `mypy`: failed because `mypy` is not installed in this shell.

## Test Results

Compile, docs parity, legacy string scan, audit script, API smoke, and pytest passed under `.venv/bin/python`. Frontend test/build and lint commands failed because the local tools are unavailable.

## Remaining User Checks

- Confirm whether `_archive/legacy_20260429` should be kept long term or removed after review.
- Confirm whether generated `outputs/backtests` should remain locally or be regenerated on demand.
