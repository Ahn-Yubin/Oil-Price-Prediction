# Data Pipeline

The data pipeline converts market prices, oil fundamentals, positioning, and news/event context into point-in-time features for deep learning and dashboard inference. Production must not silently mix mock or synthetic data into real runs.

## Current Data

The current processed datasets are organized as follows.

| Area | Path | Contents | Current Use |
| --- | --- | --- | --- |
| Market panel | `data/processed/market_panel/{interval}/panel.csv` | yfinance-based OHLCV multi-symbol panel for `1d`, `1h`, `30m`, and `15m` | Price windows for all forecast models |
| EIA petroleum | `data/processed/oil_fundamentals/eia_weekly.csv` | EIA petroleum weekly bulk supply/inventory series | Oil fundamental features |
| CFTC COT | `data/processed/oil_fundamentals/cftc_cot_weekly.csv` | CFTC Commitment of Traders positioning | Managed money/commercial positioning features |
| Event context | `data/processed/event_context/event_context_daily.csv` | Daily context vectors generated from news/events | `llm_context_seq_moe` event/context input |
| Raw news | `data/raw/news/public_market_news.csv` | Public news text from Yahoo Finance RSS/GDELT | LLM context input |
| Manifest | `data/manifests/data_inventory.json` | Dataset rows, date ranges, sources, and point-in-time safety flags | Data monitoring and reproducibility |

EIA/CFTC data is weekly and is forward-filled into daily samples using conservative availability timestamps. News and event context rows after the sample origin are not used.

## Missing Or Limited Data

| Missing Data | Impact | Resolution |
| --- | --- | --- |
| Long-history CME futures curve/settlements | Limited term structure, roll yield, and curve slope features | Acquire CME DataMine/settlement CSV and ingest with `fetch_cme_settlements.py --manual-csv` |
| Longer news history | `llm_context_seq_moe` has limited long-regime news reaction data | Split GDELT requests by period or add licensed news CSV through `NEWS_EVENTS_PATH` |
| Measured calibration residuals | Forecast bands cannot be called validated confidence intervals | Run rolling backtests and then `scripts/evaluate/calibrate_quantiles.py` |
| Intraday fundamental/event alignment | Weekly/daily features have coarse release timing for sub-daily intervals | Improve `feature_available_at` with actual release timestamps |
| Vendor-grade market data | yfinance can have gaps, revisions, or delays | Add Stooq, broker/vendor CSV, or database providers as additional sources |

Licensed data must not be scraped without authorization. If the user has an official CSV/API export, place it under `data/external` and process it with the ingest scripts.

## Providers

Provider implementations live in `market_ai/data/providers`.

- `market_price_provider.py`: yfinance/Stooq market panel ingestion
- `eia_provider.py`: EIA petroleum bulk/API/manual CSV normalization
- `cftc_provider.py`: CFTC COT ZIP/CSV/manual CSV normalization
- `cme_provider.py`: CME settlement manual/URL CSV normalization
- `fred_provider.py`: FRED macro series ingestion
- `public_news_provider.py`: Yahoo RSS/GDELT public news ingestion

Providers separate raw cache and processed output. Failures are reported through status/warnings, and production does not create synthetic fallback data.

## Storage

- `data/raw`: provider source data
- `data/interim`: intermediate joins and status outputs
- `data/processed`: cleaned data used by training and inference
- `data/features`: feature matrices and training derivatives
- `data/external`: user-provided CSV, event files, and licensed exports
- `data/manifests`: data inventory and latest snapshot
- `configs/symbol_universe.yaml`: training universe definitions

## Market Panel

```bash
.venv/bin/python scripts/data/fetch_market_prices.py --universe research_core --interval 1d --period 10y
.venv/bin/python scripts/data/fetch_market_prices.py --universe research_core --interval 1h --period 730d
.venv/bin/python scripts/data/fetch_market_prices.py --universe research_core --interval 30m --period 60d
.venv/bin/python scripts/data/fetch_market_prices.py --universe research_core --interval 15m --period 60d
```

The preferred output is `data/processed/market_panel/{interval}/panel.parquet`. If a parquet engine is unavailable, the pipeline writes `panel.csv`, which the current training commands can read.

## EIA/CFTC/CME

EIA and CFTC can be processed from public bulk/ZIP sources or manual CSV files.

```bash
.venv/bin/python scripts/data/fetch_eia_petroleum.py
.venv/bin/python scripts/data/fetch_cftc_cot.py
```

With manual CSV files:

```bash
.venv/bin/python scripts/data/fetch_eia_petroleum.py --manual-csv data/external/fundamentals/eia_petroleum_weekly.csv
.venv/bin/python scripts/data/fetch_cftc_cot.py --manual-csv data/external/fundamentals/cftc_cot_weekly.csv
.venv/bin/python scripts/data/fetch_cme_settlements.py --manual-csv data/external/fundamentals/cme_settlements.csv
```

Minimum manual CSV schemas:

- EIA: `report_date` or `date`, plus wide metric columns such as `crude_stocks`, or `series_id,value`
- CFTC: `report_date` or `date`, `open_interest`, and managed money/commercial long-short fields
- CME: `trade_date` or `date`, `settle` or `settlement`, and preferably `contract` or `contract_month`

## News/LLM Event Context

Local deterministic context:

```bash
.venv/bin/python scripts/data/build_event_context.py \
  --news-path data/raw/news/public_market_news.csv \
  --symbols CL=F,BZ=F,NG=F \
  --mode local_rules
```

External LLM context such as Google Gemma/Gemini:

```bash
.venv/bin/python scripts/data/build_event_context.py \
  --news-path data/raw/news/public_market_news.csv \
  --symbols CL=F,BZ=F,NG=F,RB=F,HO=F,GC=F,SI=F,HG=F,DX-Y.NYB,EURUSD=X,USDKRW=X,JPY=X,SPY,QQQ,^GSPC,^VIX,XLE,USO \
  --mode google_generative \
  --live
```

## Real Dataset Orchestration

Build the available public datasets in one run:

```bash
.venv/bin/python scripts/data/build_real_dataset.py \
  --universe research_core \
  --interval 1d \
  --period 10y \
  --news-timespan 3m \
  --news-maxrecords 30 \
  --skip-stooq-secondary
```

GDELT may rate-limit requests. Yahoo RSS news collection can still succeed even if GDELT fails.

## Deep Dataset

`market_ai/data/deep_dataset.py` creates `DeepLearningSample` objects.

- `x_price`: log returns, vol-scaled returns, range, rolling volatility, momentum, drawdown, autocorrelation, trend, skew/kurtosis, cycle features
- `x_cross_asset`: related returns, correlation, spread, relative strength, risk proxy, missing indicators
- `x_event_context`: event/LLM context vector
- `x_static`: current price, realized volatility, lookback, horizon

The target is `future cumulative log return / recent_realized_volatility`. Raw future price is not used directly as a training target.

## Split And No-Lookahead

Random splits are not used. Each symbol uses the later time range for validation/test. Features are included only when `feature_available_at <= as_of_time`.

## Manifest

```bash
.venv/bin/python scripts/data/build_data_inventory.py
```

The manifest records dataset name, source, path, symbol/series, frequency, start/end, rows, columns, generated time, point-in-time safety flag, and notes.
