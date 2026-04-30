# 데이터 파이프라인

데이터 파이프라인은 market data provider, symbol normalization, timeframe normalization, deep dataset builder, event context, data status reporting으로 구성됩니다.

## Provider

기본 provider는 yfinance입니다. Provider 구현은 `market_ai/data/providers`에 둡니다. 향후 CSV, vendor API, database provider를 같은 interface로 확장할 수 있어야 합니다.

## Deep Dataset

`market_ai/data/deep_dataset.py`는 `DeepLearningSample`을 생성합니다. 주요 입력은 다음과 같습니다.

- `x_price`: log return, vol-scaled return, range, rolling volatility, momentum, drawdown, autocorr, trend, skew/kurtosis, cycle feature
- `x_cross_asset`: related return/correlation/spread/relative strength/risk proxy/missing indicator
- `x_event_context`: point-in-time event context vector
- `x_static`: current price, recent realized volatility, lookback, horizon

Target은 `future cumulative log return / recent_realized_volatility`이며 raw future price를 target으로 사용하지 않습니다.

## Event Context

CSV/JSON event provider는 `NEWS_EVENTS_PATH`, `ECONOMIC_EVENTS_PATH`, `MARKET_EVENTS_PATH`를 읽습니다. Event timestamp가 `as_of_time`보다 미래이면 sample에서 제외합니다. Event가 없으면 zero context와 high uncertainty를 사용합니다.

## Split과 No-Lookahead

Random split은 사용하지 않습니다. 각 symbol의 뒤쪽 구간을 validation/test로 쓰는 time-based split을 사용합니다. Feature window는 origin 시점까지의 값만 포함하고 target은 그 이후 horizon만 사용합니다.

## 데이터 품질

모든 API 응답은 가능한 경우 `DataStatus`를 통해 source, resolved symbol, interval, last bar, stale 여부, warning을 노출합니다. Production에서는 mock data를 조용히 사용하지 않습니다. Mock/fallback data는 `APP_ENV=development` 또는 `ALLOW_MOCK_DATA=true`일 때만 허용됩니다.

## 저장 위치

- `data/raw`: 원본 수집 데이터
- `data/interim`: 중간 처리 데이터
- `data/processed`: 정제된 데이터
- `data/features`: feature matrix
- `data/external`: event file 등 외부 보조 데이터
- `configs/symbol_universe.yaml`: 학습 universe

## 2026-04-30 실전 데이터 확장

데이터 lake 구조는 다음 상대경로를 표준으로 사용합니다.

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

원본 cache는 `data/raw/market/{provider}/{interval}/{symbol}.csv`에 저장하고, processed panel은 `data/processed/market_panel/{interval}/panel.parquet` 또는 parquet engine이 없을 때 `panel.csv`로 저장합니다. yfinance 실패 시 synthetic으로 대체하지 않고 실패 report를 남깁니다.

EIA/CFTC/CME:

```bash
python scripts/data/fetch_eia_petroleum.py --manual-csv path/to/eia.csv
python scripts/data/fetch_cftc_cot.py --manual-csv path/to/cftc.csv
python scripts/data/fetch_cme_settlements.py --manual-csv path/to/cme.csv
```

API key나 licensed provider가 없어도 manual CSV ingest는 지원합니다. CME는 유료/라이선스 데이터가 필요할 수 있으므로 fake scraping을 하지 않습니다. Weekly fundamental/COT 데이터는 `release_time` 또는 보수적 release timestamp 이후에만 daily feature로 forward-fill됩니다.

Event context:

```bash
python scripts/data/build_event_context.py --events-path data/external/events/sample_market_events.csv --mode local_rules
```

출력은 `data/processed/event_context/event_context_daily.csv`와 `llm_context_cache.jsonl`입니다. 모든 event/news/fundamental feature는 `feature_available_at <= as_of_time` 조건을 기준으로 merge됩니다.

Manifest:

```bash
python scripts/data/build_data_inventory.py
```

Manifest entry는 dataset name, source, path, symbol/series, frequency, start/end, rows, columns, generated time, source/provider, point-in-time safety flag, notes를 기록합니다.
