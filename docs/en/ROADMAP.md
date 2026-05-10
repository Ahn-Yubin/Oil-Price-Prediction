# Roadmap

The project starts with oil forecasting and expands toward a universal market AI platform. The current priority order is data reliability, LLM context stabilization, retraining, backtest/calibration, and frontend operations.

## Immediate Priorities

1. Validate the Google Gemma/Gemini LLM connection live.
2. Regenerate live LLM event context from `data/raw/news/public_market_news.csv`.
3. Retrain `llm_context_seq_moe` and `deep_lstm_tcn_fusion` for h8/h45.
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

- Stabilize oil/energy models in `research_core`.
- Expand the universe to ETFs, FX, metals, indices, rates, and crypto.
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
