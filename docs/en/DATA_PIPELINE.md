# Data Pipeline

The data pipeline consists of market data providers, symbol normalization, timeframe normalization, and data status reporting.

## Provider

The default provider is yfinance. Provider implementations live under `market_ai/data/providers`. CSV, vendor API, and database providers should be added behind the same interface over time.

## Data Quality

API responses should expose source, resolved symbol, interval, last bar, stale state, and warnings through `DataStatus` whenever possible.

## Mock Data Policy

Production must not silently use mock data. Mock/fallback data is allowed only when `APP_ENV=development` or `ALLOW_MOCK_DATA=true`.

## Storage Layout

- `data/raw`: raw ingested data
- `data/interim`: intermediate data
- `data/processed`: cleaned data
- `data/features`: feature matrices
- `data/external`: external auxiliary data
