# Roadmap

The project is stabilizing a WTI oil forecasting-only model and dashboard. The current priority order is data reliability, LLM context stabilization, single-model retraining, backtest/calibration, and frontend operations.

## Immediate Priorities

1. Validate the Google Gemma/Gemini LLM connection live.
2. Regenerate live LLM event context from `data/raw/news/public_market_news.csv`.
3. Retrain `oil_context_fusion` for 1D/1H h30 and validate 7/14/30 display lengths.
4. Acquire CME settlement/curve CSV and build `cme_curve_daily.csv`.
5. Run rolling backtests and quantile calibration.
6. Validate `/api/market-context` and frontend markers/panels on the actual chart.

## Data Expansion

- Add secondary price sources beyond yfinance.
- Add long-history CME futures curve, settlement, volume, and open interest features.
- Extend news history to multiple years.
- Manage EIA/CFTC/FRED release timestamps more precisely.
- Refresh data inventory and latest snapshot regularly.

## Model Expansion

- Stabilize the oil/energy model in `oil_core`.
- Keep ETFs, FX, metals, indices, rates, and crypto as auxiliary oil features rather than independent forecast targets.
- Keep the LLM as a context encoder only; numeric forecasts remain owned by time-series models.
- Use calibrated interval language only after coverage has been measured.

## Frontend Expansion

- Make context markers and news cards useful as model diagnostics.
- Display forecast scenario commentary alongside data quality.
- Split the UI into chart, controls, panels, api, and state modules as it grows.

## Principles To Preserve

- Preserve `/api/chart` compatibility until it is explicitly removed.
- Keep artifacts and metadata separate from source code.
- Do not silently use mock/synthetic fallback in production.
- Update Korean source docs and English mirror docs together.
