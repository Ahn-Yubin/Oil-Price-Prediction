# Frontend

The frontend is a WTI oil (`CL=F`) TradingView Lightweight Charts-style forecast overlay dashboard. The first screen is the actual chart and model/context workspace, not a marketing page.

## Location

- `frontend/index.html`: dashboard shell and panel markup
- `frontend/src/main.js`: chart rendering, API fetching, marker rendering, interaction
- `frontend/src/dashboard.css`: responsive layout and dashboard styles
- `frontend/src/api`: future API client split point

## API Usage

The UI calls `/api/forecast` first and uses `/api/chart` compatibility payloads when needed. `/api/chart` must remain compatible with the existing overlay.

The desktop screen uses a three-column layout: the chart on the left, AI market commentary/news interpretation in the middle via `/api/model-commentary` and `/api/market-context`, and the forecast report on the right. When a user selects a historical candle and runs a backtest visualization, the chart calls `/api/backtests/visualization`.

- `news`: recent headlines and sources
- `context_points`: event/context dates for chart markers
- `scenario_commentary`: backend compatibility field. The UI shows news and LLM interpretation instead of bull/base/bear cards.
- `llm_context_summary`: whether LLM context is active and usable

The backtest visualization payload keeps the same base chart keys and adds:

- `origin_time`: historical candle time used to rebuild the forecast
- `actual_future_candles`: realized OHLCV after the origin
- `backtest`: history/future row counts and horizon metadata

The model commentary payload explains the single operational model forecast path through the LLM or deterministic fallback. The commentary is analyst-style market reasoning based on news and chart action, not a technical description of the model internals. The LLM is only an explainer of already-produced model outputs, not a new numeric price forecaster.

## Chart Display

Current display targets:

- `CL=F` historical candles
- forecast p50 path
- p10/p90 or p05/p95 band
- historical news/context markers
- news headline cards with the LLM interpretation of each news/event point
- translucent realized candle overlay after the selected origin
- middle AI commentary/news interpretation panels and right-side forecast report panel

Markers indicate what context existed around that date. They are not numeric price forecasts made by the LLM.

## Current UX Adjustments

- The symbol search/input control was removed. The screen and API requests always use `CL=F`.
- The interval selector offers 1D and 1H. 15M/30M are excluded from the operating UI while the project first stabilizes the 1H/1D h30 unified model artifacts.
- The forecast length selector offers `7`, `14`, and `30`. The backend runs one h30 artifact per interval and displays the requested leading segment.
- The only user-facing model is `oil_context_fusion`. Older models remain only as internal benchmark/fallback paths.
- `/api/forecast`, `/api/chart`, `/api/market-context`, `/api/model-commentary`, and `/api/backtests/visualization` receive the same horizon value.
- Chart height is reduced to 460px so page-level scrolling is easier.
- The right context event list no longer has its own inner scroll, avoiding conflicts with page scrolling.
- The news panel shows only headlines and LLM interpretation, without extra scenario card copy.
- Running a backtest tries to preserve the existing chart time scale and price scale.

## UX Principles

- Do not hide data quality and warnings.
- Mock/fallback data must not silently appear in production.
- Forecast bands must not be labeled confidence intervals until coverage is validated.
- An unavailable `oil_context_fusion` artifact should surface a warning and training command.
- Operational tool UI should stay quiet, dense, and information-focused.

## Future Structure

As the UI grows, split it into:

- `frontend/src/components/chart`
- `frontend/src/components/controls`
- `frontend/src/components/panels`
- `frontend/src/api`
- `frontend/src/state`
