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
