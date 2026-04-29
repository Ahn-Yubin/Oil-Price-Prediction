# Frontend

The frontend is a dashboard UI with TradingView Lightweight Charts-style overlays.

## Location

- `frontend/index.html`: dashboard HTML shell
- `frontend/src/main.js`: current chart interaction implementation
- `frontend/src/dashboard.css`: dashboard styles
- `frontend/src/api`: API client stubs

## API Usage

The UI tries `/api/forecast` first and falls back to the `/api/chart` compatibility payload when needed. `/api/chart` must remain compatible with the existing overlay.

## Future Structure

As the UI grows, split components under `frontend/src/components/chart`, `components/controls`, `components/badges`, and `components/panels`.
