# Frontend

The frontend is a TradingView Lightweight Charts-style forecast overlay dashboard. The first screen is the actual chart and model/context workspace, not a marketing page.

## Location

- `frontend/index.html`: dashboard shell and panel markup
- `frontend/src/main.js`: chart rendering, API fetching, marker rendering, interaction
- `frontend/src/dashboard.css`: responsive layout and dashboard styles
- `frontend/src/api`: future API client split point

## API Usage

The UI calls `/api/forecast` first and uses `/api/chart` compatibility payloads when needed. `/api/chart` must remain compatible with the existing overlay.

The context panel calls `/api/market-context`.

- `news`: recent headlines and sources
- `context_points`: event/context dates for chart markers
- `scenario_commentary`: bull/base/bear scenario commentary
- `llm_context_summary`: whether LLM context is active and usable

## Chart Display

Current display targets:

- historical candles
- forecast p50 path
- p10/p90 or p05/p95 band
- bull/base/bear scenario summary
- historical news/context markers
- event/context card list

Markers indicate what context existed around that date. They are not numeric price forecasts made by the LLM.

## UX Principles

- Do not hide data quality and warnings.
- Mock/fallback data must not silently appear in production.
- Forecast bands must not be labeled confidence intervals until coverage is validated.
- Unavailable deep artifacts should be handled with warnings in model selectors.
- Operational tool UI should stay quiet, dense, and information-focused.

## Future Structure

As the UI grows, split it into:

- `frontend/src/components/chart`
- `frontend/src/components/controls`
- `frontend/src/components/panels`
- `frontend/src/api`
- `frontend/src/state`
