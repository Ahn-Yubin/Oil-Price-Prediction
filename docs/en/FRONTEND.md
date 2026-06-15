# Frontend

The frontend is a WTI oil (`CL=F`) TradingView Lightweight Charts-style forecast overlay dashboard. The first screen is the actual chart and model/context workspace, not a marketing page.

## Location

- `frontend/index.html`: dashboard shell and panel markup
- `frontend/src/main.js`: chart rendering, API fetching, marker rendering, interaction
- `frontend/src/dashboard.css`: responsive layout and dashboard styles
- `frontend/src/api`: future API client split point

## API Usage

The UI calls `/api/forecast` first and uses `/api/chart` compatibility payloads when needed. `/api/chart` must remain compatible with the existing overlay.

The desktop screen uses a three-column layout: chart plus AI chat on the left, AI market commentary/news interpretation in the middle, and the forecast report on the right. The three AI panels prefer `/api/dashboard-analysis`, which returns market commentary, news interpretation, and report prose from one external LLM response. The chart mode control is a three-way `Backtest / Live / Scenario` segmented control. When a user selects a historical candle and runs a backtest visualization, the chart calls `/api/backtests/visualization`. In Scenario mode, the UI groups future events into scenario folders, calls `/api/scenarios/forecast`, and draws the returned path as a scenario-specific overlay above the live forecast. When a backtest origin is active, the news interpretation panel requests point-in-time news/event context up to the same `origin_time` instead of live current news, and generated prose uses absolute dates/times rather than live relative wording.

- `news`: recent headlines and sources
- `context_points`: event/context dates for chart markers and news interpretation rows
- `chart_context_points`: deduplicated and spaced event/context dates for chart markers
- `scenario_commentary`: backend compatibility field. The UI shows news and LLM interpretation instead of bull/base/bear cards.
- `llm_context_summary`: whether LLM context is active and usable

The backtest visualization payload keeps the same base chart keys and adds:

- `origin_time`: historical candle time used to rebuild the forecast
- `actual_future_candles`: realized OHLCV after the origin
- `backtest`: history/future row counts and horizon metadata

The scenario forecast payload is kept in the list-style bottom panel.

- `points`: scenario chart overlay. The first point is the current-price anchor.
- `llm_context_summary`: LLM-derived bias/impact/uncertainty, `scenario_override` source, and horizon-level `model_context_schedule`
- `llm_context`: validated structured event context
- `warning_objects`: missing event_time, LLM fallback, and artifact/data-quality warnings

The dashboard analysis payload explains the single operational model forecast path through the LLM. The commentary is analyst-style market reasoning based on news and chart action, not a technical description of the model internals. The LLM is only an explainer of already-produced model outputs, not a new numeric price forecaster. The UI renders commentary as body-style paragraphs in the selected language and does not list raw English headlines in Korean mode. Standalone `/api/model-commentary`, `/api/market-context`, and `/api/report` remain compatibility/diagnostic paths.

## Chart Display

Current display targets:

- `CL=F` historical candles
- forecast display path with 1-week, 2-week, and 1-month endpoint markers
- p10/p90 or p05/p95 band recentered around the display path
- historical news/context markers. Active request and origin keys prevent future forecast-window news from leaking into a backtest chart.
- news headline cards with the LLM interpretation of each news/event point
- translucent realized candle overlay after the selected origin
- middle AI commentary/news interpretation panels and right-side forecast report panel

Markers indicate what context existed around that date. They are not numeric price forecasts made by the LLM.

## Current UX Adjustments

- The symbol search/input control was removed. The screen and API requests always use `CL=F`.
- The operating screen is fixed on `CL=F / 1D / 30-day` forecasting. 1H/15M/30M remain research and API-validation targets while the UI focuses on a stable 1D h30 artifact.
- The forecast-length selector was removed. The backend runs one h30 artifact, and the UI marks the 1-week, 2-week, and 1-month endpoints on the same 30-day path.
- The only user-facing model is `oil_context_fusion`. Older models remain only as internal benchmark/fallback paths.
- `/api/forecast`, `/api/chart`, `/api/dashboard-analysis`, and `/api/backtests/visualization` receive the same horizon/origin/language key.
- Scenario mode keeps the live forecast as the baseline and draws user-added scenarios as separate line overlays. Each scenario can be shown or hidden from the toggle inside the list.
- A scenario is a folder with a title. Event input consists of title, event time, and content. Event time is used as the activation point for horizon-level model context inside the forecast window.
- In desktop landscape, the chart and chat share the left column in a 2:1 row ratio, and the dashboard frame adjusts row heights to fill the viewport.
- The AI chat panel sits below the chart. The full panel is the message area; the title header and bottom input area are fixed neutral-glass overlays matching the commentary/news/report headers.
- Middle/right side panels use internal scroll only under constrained viewport conditions, with max-height limits to avoid fighting the page scroll.
- The news panel shows point-in-time news and LLM interpretation, without extra scenario card copy.
- The news panel hides internal placeholders such as `Deterministic local event context encoder...`, non-public diagnostics, and blank `-` explanations. Duplicate headline/source/date rows are deduplicated.
- AI market commentary and forecast reports use body-style paragraphs instead of bullet lists. Reports avoid developer-facing terms such as data status, band status, or internal context scores, and explain the forecast in public market-analysis language.
- Running a backtest tries to preserve the existing chart time scale and price scale.
- While backtest mode is active, clicking another candle immediately calls `/api/backtests/visualization` for the new origin. The user no longer has to return to live mode before changing the origin.
- Language switches, live/backtest changes, and backtest-origin changes mark older LLM responses stale. If another refresh is requested while one is running, it is queued as a pending refresh and rerun once with the latest key after the active request finishes.
- While chat is waiting for an answer, the assistant bubble shows a left-to-right growing-dot typing animation.
- The larger UX refactor is to split live chart state and backtest visualization state into separate chart instances, then use a backtest panel visibility toggle. That would keep live refresh state and historical origin validation state clearly separated.

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
