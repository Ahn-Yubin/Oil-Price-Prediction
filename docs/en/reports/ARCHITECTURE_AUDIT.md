# Architecture Audit

The previous repository mixed FastAPI endpoints, forecasting logic, model artifacts, reports, outputs, and an older baseline project under oil-specific directories.

Key findings:

- HTTP code and market AI logic were coupled through the old `app` package.
- `.npz` model artifacts and JSON metadata were stored under application code.
- Backtest and train scripts lived at project root.
- Generated reports lived at root.
- A sibling `oil-price-baseline` experiment existed beside the dashboard.

The new architecture separates `backend`, `market_ai`, `frontend`, `scripts`, `artifacts`, `outputs`, and bilingual `docs`.
