# Order Recovery Audit

This document verifies the current repository after the project cleanup and Markdown i18n tasks were performed in the wrong order. The audit focuses on consistency between the actual code structure, documentation policy, artifacts, and verification scripts.

## 1. Issues That Actually Occurred

- The documentation check script initially did not validate `AGENTS.md` or root-level generated report clutter.
- The API smoke test could treat dependency-unavailable responses only as failures instead of classifying expected 503 responses.
- `FINAL_IMPLEMENTATION_REPORT.md` contained both cleanup and documentation results, but the order-recovery audit was not recorded as a dedicated report.
- Legacy path strings were not found in active user-facing docs. They remained only in cleanup/final/audit reports as historical move records, which this audit classifies as intentional historical references.

## 2. Issues That Did Not Occur

- The `backend.app.main:app` entrypoint exists, and the `app/main.py` compatibility wrapper is still present.
- No reverse dependency from `market_ai` to `backend` was found.
- `.npz` model artifacts and metadata JSON files are preserved under `artifacts/models` and `artifacts/metadata`.
- No absolute local paths were found in active README/docs/AGENTS/CHANGELOG files.
- No generated report Markdown files remain in the repository root.
- `GET /api/chart`, `GET /api/forecast`, `GET /api/models`, `GET /api/data-status`, and `GET /api/health` all returned 200 in the smoke test.

## 3. Modified Files

- `scripts/maintenance/check_docs_i18n.py`
- `scripts/maintenance/smoke_test_api.py`
- `docs/ko/reports/ORDER_RECOVERY_AUDIT.md`
- `docs/en/reports/ORDER_RECOVERY_AUDIT.md`
- `docs/ko/reports/DOCS_AUDIT.md`
- `docs/en/reports/DOCS_AUDIT.md`
- `docs/ko/reports/FINAL_IMPLEMENTATION_REPORT.md`
- `docs/en/reports/FINAL_IMPLEMENTATION_REPORT.md`

## 4. Documentation Path Fixes

- Active docs now use `uvicorn backend.app.main:app --reload --port 8000` as the backend entrypoint.
- Training docs use `python scripts/train/train_pretrained_models.py --interval 1d`.
- Backtest docs use `python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 5 --models random_walk,drift,flat --no-plots`.
- Model artifact docs point to `artifacts/models`; metadata docs point to `artifacts/metadata`.
- Legacy path strings are intentionally allowed only inside `docs/*/reports` cleanup history.

## 5. Code Import Fixes

- No application import paths were changed during this recovery pass.
- No `market_ai` -> `backend` reverse import was found.
- `app/main.py` remains a thin compatibility wrapper for the new entrypoint.
- Code changes in this pass were limited to maintenance script hardening.

## 6. Artifact Path Verification

- `.npz` files are under `artifacts/models`.
- Metadata JSON files are under `artifacts/metadata`.
- `market_ai/config.py` defaults to `artifacts/models` and `artifacts/metadata`.
- `market_ai/modeling/registry.py` uses `settings.model_dir` and `settings.metadata_dir`.
- Documentation, `.env.example`, and `configs/default.yaml` describe the same locations.

## 7. API Compatibility Verification

`scripts/maintenance/smoke_test_api.py` checks the following endpoints with FastAPI `TestClient`.

- `GET /api/health`
- `GET /api/models`
- `GET /api/data-status?symbol=CL=F&interval=1d`
- `GET /api/forecast?symbol=CL=F&interval=1d`
- `GET /api/chart?symbol=CL=F&interval=1d`

Current results are all 200. The script now classifies dependency-related 503 responses as expected 503 instead of treating them as generic failures.

## 8. Test Results

- `python scripts/maintenance/check_docs_i18n.py`: failed because `python` is not installed in the current shell.
- `python3 scripts/maintenance/check_docs_i18n.py`: passed.
- `.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy`: passed. Only historical report references were allowed.
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
- `npm test`: failed because `npm` is not installed in the current shell.
- `npm run build`: failed because `npm` is not installed in the current shell.
- `ruff check .`: failed because `ruff` is not installed in the current shell.
- `mypy`: failed because `mypy` is not installed in the current shell.

## 9. Remaining Risks

- `_archive/legacy_20260429` still contains historical project and cache remnants. It is not active scan input, but it can still create repository size and navigation noise.
- The root working tree still has a large uncommitted migration diff from the prior restructuring.
- Frontend build/test and lint checks could not run because local Node/Ruff/Mypy tooling is unavailable.
- In this environment, commands documented with `python` must be run as `python3` or `.venv/bin/python`.

## 10. User Checks

- Decide whether `_archive/legacy_20260429` should be kept long term or removed after separate backup.
- If the frontend is deployed, rerun `npm test` and `npm run build` in an environment with Node.js and npm.
- In production, verify that `ALLOW_MOCK_DATA` is not enabled and that real providers plus artifact paths are ready.
- Review the current uncommitted migration diff and lock it into a branch/commit once accepted.
