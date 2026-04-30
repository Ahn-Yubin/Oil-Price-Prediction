# Data Pipeline

The data pipeline consists of market data providers, symbol normalization, timeframe normalization, the deep dataset builder, event context, and data status reporting.

## Provider

The default provider is yfinance. Provider implementations live under `market_ai/data/providers`. CSV, vendor API, and database providers should be added behind the same interface.

## Deep Dataset

`market_ai/data/deep_dataset.py` creates `DeepLearningSample` records. Main inputs:

- `x_price`: log return, vol-scaled return, range, rolling volatility, momentum, drawdown, autocorr, trend, skew/kurtosis, and cycle features
- `x_cross_asset`: related return/correlation/spread/relative strength/risk proxy/missing indicator
- `x_event_context`: point-in-time event context vector
- `x_static`: current price, recent realized volatility, lookback, and horizon

The target is `future cumulative log return / recent_realized_volatility`; raw future price is not used as a target.

## Event Context

The CSV/JSON event provider reads `NEWS_EVENTS_PATH`, `ECONOMIC_EVENTS_PATH`, and `MARKET_EVENTS_PATH`. Events with timestamps after `as_of_time` are excluded from the sample. If no events exist, the system uses zero context with high uncertainty.

## Split and No-Lookahead

Random splits are not used. The later segment of each symbol is used for validation/test through time-based splitting. Feature windows contain only data available at the origin, and targets use only the following horizon.

## Data Quality

API responses surface source, resolved symbol, interval, last bar, stale status, and warnings through `DataStatus` whenever possible. Production must not silently use mock data. Mock/fallback data is allowed only when `APP_ENV=development` or `ALLOW_MOCK_DATA=true`.

## Storage

- `data/raw`: collected raw data
- `data/interim`: intermediate data
- `data/processed`: cleaned data
- `data/features`: feature matrices
- `data/external`: event files and external auxiliary data
- `configs/symbol_universe.yaml`: training universes

## 2026-04-30 Real Data Expansion

The data lake uses these standard relative paths:

- `data/raw/market`, `data/raw/eia`, `data/raw/cftc`, `data/raw/cme`, `data/raw/events`, `data/raw/news`
- `data/interim/market`, `data/interim/fundamentals`, `data/interim/events`
- `data/processed/market_panel`, `data/processed/oil_fundamentals`, `data/processed/event_context`
- `data/features/deep_training`
- `data/manifests/data_inventory.json`, `data/manifests/latest_snapshot.json`

Market panel:

```bash
python scripts/data/fetch_market_prices.py --universe oil_core --interval 1d --period 10y
python scripts/data/fetch_market_prices.py --universe default_global --interval 1d --period 10y
```

Raw cache is stored under `data/raw/market/{provider}/{interval}/{symbol}.csv`; processed panels are written to `data/processed/market_panel/{interval}/panel.parquet`, or `panel.csv` when no parquet engine is installed. yfinance failures are reported and are not silently replaced with synthetic data.

EIA/CFTC/CME:

```bash
python scripts/data/fetch_eia_petroleum.py --manual-csv path/to/eia.csv
python scripts/data/fetch_cftc_cot.py --manual-csv path/to/cftc.csv
python scripts/data/fetch_cme_settlements.py --manual-csv path/to/cme.csv
```

Manual CSV ingest works even without API keys or licensed providers. CME may require paid/licensed data, so the project does not fake scrape it. Weekly fundamental/COT data is forward-filled only after `release_time` or a conservative release timestamp.

Event context:

```bash
python scripts/data/build_event_context.py --events-path data/external/events/sample_market_events.csv --mode local_rules
```

Outputs are `data/processed/event_context/event_context_daily.csv` and `llm_context_cache.jsonl`. Event/news/fundamental features are merged only when `feature_available_at <= as_of_time`.

Manifest:

```bash
python scripts/data/build_data_inventory.py
```

Each manifest entry records dataset name, source, path, symbol/series, frequency, start/end, rows, columns, generated time, source/provider, point-in-time safety flag, and notes.
