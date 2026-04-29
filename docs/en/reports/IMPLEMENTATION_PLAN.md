# Implementation Plan

1. Preserve API compatibility for `/api/chart`, `/api/forecast`, `/api/models`, `/api/data-status`, and `/api/health`.
2. Move FastAPI code to `backend/app` and market logic to `market_ai`.
3. Move model artifacts to `artifacts/models` and metadata JSON to `artifacts/metadata`.
4. Move human-run scripts to `scripts`.
5. Add maintenance scripts for docs parity, unused file audit, and API smoke testing.
6. Archive uncertain legacy content instead of deleting it.
7. Verify with compile, smoke API, docs parity, audit script, pytest, frontend test, and lint when available.
