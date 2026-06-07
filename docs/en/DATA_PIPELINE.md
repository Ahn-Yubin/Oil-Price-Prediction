# Data Pipeline

The data pipeline converts market prices, oil fundamentals, positioning, and news/event context into point-in-time features for deep learning and dashboard inference. Production must not silently mix mock or synthetic data into real runs.

## Current Data

The current processed datasets are organized as follows.

| Area | Path | Contents | Current Use |
| --- | --- | --- | --- |
| Market panel | `data/processed/market_panel/{interval}/panel.csv` | yfinance-based OHLCV multi-symbol panel for `1d`, `1h`, `30m`, and `15m` | Price windows for all forecast models |
| EIA petroleum | `data/processed/oil_fundamentals/eia_weekly.csv` | EIA petroleum weekly bulk supply/inventory series | Oil fundamental features |
| CFTC COT | `data/processed/oil_fundamentals/cftc_cot_weekly.csv` | CFTC Commitment of Traders positioning | Managed money/commercial positioning features |
| Macro panel | `data/processed/macro_panel/fred_daily_wide.csv` | FRED macro rates/indices daily wide panel | Macro/rates cross-asset features |
| Event context | `data/processed/event_context/event_context_daily.csv` | Daily context vectors generated from news/events. It is 27-dimensional: 13 LLM features plus 14 raw-news-pool aggregate features. | `oil_context_fusion` event/context input |
| Raw news | `data/raw/news/public_market_news.csv` | Public news text from Yahoo Finance RSS, Google News RSS backfill, public RSS, and GDELT | LLM context input |
| Manifest | `data/manifests/data_inventory.json` | Dataset rows, date ranges, sources, and point-in-time safety flags | Data monitoring and reproducibility |

EIA/CFTC data is weekly and is forward-filled into daily samples using conservative availability timestamps. News and event context rows after the sample origin are not used.

## Current Symbol Universe

The operational training target is the `oil_core` universe; FX, metal, equity, and volatility symbols from `research_core` can remain auxiliary cross-asset features.

| Category | Symbols | Count | Use |
| --- | --- | ---: | --- |
| Energy futures/ETF/sector | `CL=F`, `BZ=F`, `NG=F`, `RB=F`, `HO=F`, `USO`, `XLE` | 7 | Crude oil, Brent, natural gas, refined products, energy ETF, and sector proxies |
| Metals | `GC=F`, `SI=F`, `HG=F` | 3 | Gold, silver, and copper macro/commodity cross-asset signals |
| FX/macro | `DX-Y.NYB`, `EURUSD=X`, `USDKRW=X`, `JPY=X` | 4 | Dollar index and major FX proxies |
| Equity/volatility | `SPY`, `QQQ`, `^GSPC`, `^VIX` | 4 | Risk-on/off, equity market, and volatility regime proxies |

Current dataset sizes:

| Dataset | Rows | Date Range | Notes |
| --- | ---: | --- | --- |
| `market_panel/1d` | 45,528 | 2016-06-06 ~ 2026-06-05 | 18 symbols, enough for 10-year daily experiments |
| `market_panel/1h` | 208,056 | 2023-06-05 ~ 2026-05-04 | Intraday 1h |
| `market_panel/30m` | 33,765 | 2026-02-05 ~ 2026-05-04 | Short because of Yahoo interval limits |
| `market_panel/15m` | 67,207 | 2026-02-05 ~ 2026-05-04 | Short because of Yahoo interval limits |
| `eia_weekly` | 15,966 | 1982-08-25 ~ 2026-05-11 | Petroleum supply/inventory data |
| `cftc_cot_weekly` | 3,776 | 2016-01-08 ~ 2026-05-10 | Positioning data |
| `macro_panel/fred_daily_wide` | 16,402 | 1962-01-02 ~ 2026-05-01 | Macro rates/indices |
| `public_market_news` | 148,408 | 2016-11-01 ~ 2026-06-05 | Google News RSS backfill + Yahoo Finance RSS + public RSS |
| `event_context_daily` | 45,188 | 2016-11-01 ~ 2026-05-08 | Daily context for 13 related symbols. The 3,476 CL=F rows use Google Generative LLM context + raw-news-pool features with 0 fallback rows |

Sufficiency assessment:

- Daily price, supply, inventory, and positioning data are enough for the h30 operating artifact, the fixed 30-day path, and 1W/2W/1M endpoint markers.
- 30m/15m data is too short for reliable deep model generalization tests.
- News and LLM event context now overlap most of the 1d CL=F price panel, so the long-regime context gap caused by relying only on GDELT has been reduced.
- `event_context_daily` stores local_rules context for related symbols plus external LLM context for CL=F. The CL=F rows were reprocessed with the new API key and cache/resume retries until fallback reached 0 rows. If quota is hit again during future rebuilds, rerun the same command to resume only failed or missing rows.
- On 2026-06-05 the news-compression bottleneck was reduced by adding raw-news-pool features alongside the bounded recent news items read by the LLM. The added features cover recent 1/3/7/30-day news volume, selection coverage, bullish/bearish pressure, energy/geopolitical/macro/supply/demand pressure, and source diversity. The 148k-row raw news CSV contains duplicate energy-news rows across energy symbols, so CL=F/ALL point-in-time rows are used to avoid duplicate weighting.

## Missing Or Limited Data

| Missing Data | Impact | Resolution |
| --- | --- | --- |
| Long-history CME futures curve/settlements | Limited term structure, roll yield, and curve slope features | Acquire CME DataMine/settlement CSV and ingest with `fetch_cme_settlements.py --manual-csv` |
| Vendor-grade/licensed news and broader event coverage | Public RSS backfill is now long enough, but may contain duplicates, search bias, or sparse full text | Add official user-provided news/API exports through `NEWS_EVENTS_PATH` or `data/external` ingest |
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
- `public_news_provider.py`: Yahoo RSS, Google News RSS date-window backfill, generic public RSS, and GDELT public news ingestion

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
.venv/bin/python scripts/data/fetch_market_prices.py --universe oil_core --interval 1d --period 10y
.venv/bin/python scripts/data/fetch_market_prices.py --universe oil_core --interval 1h --period 730d
.venv/bin/python scripts/data/fetch_market_prices.py --universe oil_core --interval 30m --period 60d
.venv/bin/python scripts/data/fetch_market_prices.py --universe oil_core --interval 15m --period 60d
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
  --symbols CL=F \
  --mode google_generative \
  --live \
  --start 2016-11-01 \
  --end 2026-05-08 \
  --news-limit-per-context 8 \
  --llm-batch-size 1 \
  --llm-min-interval-seconds 0.25 \
  --progress-every 500
```

When the LLM API has a daily quota, processed rows are appended immediately to `llm_context_cache.jsonl`. Re-running the same command skips rows with the same `symbol/date/news_hash`.

```bash
.venv/bin/python scripts/data/build_event_context.py \
  --news-path data/raw/news/public_market_news.csv \
  --symbols CL=F \
  --mode google_generative \
  --live \
  --start 2016-11-01 \
  --end 2026-05-08 \
  --news-limit-per-context 8 \
  --llm-batch-size 1 \
  --llm-min-interval-seconds 5.0 \
  --progress-every 100000
```

`--news-limit-per-context` controls how many recent news items the LLM reads directly for one `symbol/date` context. The current historical cache is aligned to the latest 8 news items from the prior 7 days. This limit controls token cost; independently, the builder computes 14 aggregate features from the full raw point-in-time news pool over recent 1/3/7/30-day windows to reduce the model-input bottleneck. `--llm-batch-size` controls how many `symbol/date` contexts are encoded in one external LLM request. `--llm-min-interval-seconds` throttles requests to stay below RPM limits. Use `--no-resume-cache` only when forcing a full recomputation.

External LLM training-context builds are operated strictly by default. When `--live` and an external LLM mode are used together, fallback rows stop the build; keep the cache and retry only the failed dates. Use `--allow-external-llm-fallback` only when deliberately testing fallback behavior during development.

## Real Dataset Orchestration

Build the available public datasets in one run:

```bash
.venv/bin/python scripts/data/build_real_dataset.py \
  --universe oil_core \
  --interval 1d \
  --period 10y \
  --news-timespan 3m \
  --news-maxrecords 30 \
  --skip-stooq-secondary
```

GDELT may rate-limit requests. Yahoo RSS, Google News RSS backfill, and generic public RSS collection can still continue; production records failures through warnings/status and does not synthesize news.

## Deep Dataset

`market_ai/data/deep_dataset.py` creates `DeepLearningSample` objects.

- `x_price`: log returns, vol-scaled returns, range, rolling volatility, momentum, drawdown, autocorrelation, trend, skew/kurtosis, cycle features
- `x_cross_asset`: related returns, correlation, spread, relative strength, risk proxy, missing indicators
- `x_event_context`: recency-weighted event/LLM context vector so recent news is not diluted by the full 128-day lookback. It is currently 27-dimensional: 13 LLM features plus 14 raw-news-pool aggregate features.
- `x_static`: current price, realized volatility, lookback, horizon

The target is `future cumulative log return / recent_realized_volatility`, currently capped to `[-36, 36]` during training. Raw future price is not used directly as a training target; forecast prices are reconstructed with `current_price * exp(predicted_cumulative_log_return_h)`.

## Split And No-Lookahead

Random splits are not used. Each symbol uses the later time range for validation/test. Features are included only when `feature_available_at <= as_of_time`.

## Manifest

```bash
.venv/bin/python scripts/data/build_data_inventory.py
```

The manifest records dataset name, source, path, symbol/series, frequency, start/end, rows, columns, generated time, point-in-time safety flag, and notes.
