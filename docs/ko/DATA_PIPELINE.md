# 데이터 파이프라인

데이터 파이프라인은 시장 가격, 원유 수급, 포지셔닝, 뉴스/이벤트 context를 point-in-time feature로 정리해 deep learning 학습과 dashboard inference에 공급합니다. Production에서는 mock 또는 synthetic 데이터를 조용히 섞지 않습니다.

## 현재 보유 데이터

현재 구축된 주요 processed dataset은 다음 계층으로 나뉩니다.

| 영역 | 경로 | 내용 | 현재 용도 |
| --- | --- | --- | --- |
| Market panel | `data/processed/market_panel/{interval}/panel.csv` | yfinance 기반 OHLCV multi-symbol panel. `1d`, `1h`, `30m`, `15m` interval 사용 | 모든 forecast model의 가격 window |
| EIA petroleum | `data/processed/oil_fundamentals/eia_weekly.csv` | EIA petroleum weekly bulk에서 추출한 재고/수급 계열 | 원유 fundamental feature |
| CFTC COT | `data/processed/oil_fundamentals/cftc_cot_weekly.csv` | CFTC Commitment of Traders 포지셔닝 | managed money/commercial positioning feature |
| Macro panel | `data/processed/macro_panel/fred_daily_wide.csv` | FRED macro rates/indices daily wide panel | macro/rates cross-asset feature |
| Event context | `data/processed/event_context/event_context_daily.csv` | 뉴스/이벤트를 daily context vector로 변환한 데이터 | `oil_context_fusion` event/context input |
| Raw news | `data/raw/news/public_market_news.csv` | Yahoo Finance RSS, Google News RSS, GDELT 등 공개 뉴스 원문 | LLM context 생성 입력 |
| Manifest | `data/manifests/data_inventory.json` | dataset별 rows, 기간, source, point-in-time safety 기록 | 데이터 감시와 재현성 |

EIA/CFTC는 weekly 데이터이므로 daily sample에는 보수적인 available timestamp 기준으로 forward-fill됩니다. 뉴스와 event context는 timestamp가 sample origin 이후인 행을 사용하지 않습니다.

## 현재 수집 종목 Universe

운영 학습 대상은 `oil_core` universe이며, `research_core`의 FX/metal/equity/volatility 종목은 보조 cross-asset feature로 활용할 수 있습니다.

| 분류 | 종목 | 개수 | 용도 |
| --- | --- | ---: | --- |
| Energy futures/ETF/sector | `CL=F`, `BZ=F`, `NG=F`, `RB=F`, `HO=F`, `USO`, `XLE` | 7 | 원유/브렌트/천연가스/정제유/에너지 ETF 및 섹터 proxy |
| Metals | `GC=F`, `SI=F`, `HG=F` | 3 | 금/은/구리 macro 및 commodity cross-asset signal |
| FX/macro | `DX-Y.NYB`, `EURUSD=X`, `USDKRW=X`, `JPY=X` | 4 | 달러, 주요 FX, 원화 proxy |
| Equity/volatility | `SPY`, `QQQ`, `^GSPC`, `^VIX` | 4 | risk-on/off, 주식시장, 변동성 regime |

현재 데이터 크기:

| Dataset | Rows | 기간 | 비고 |
| --- | ---: | --- | --- |
| `market_panel/1d` | 45,523 | 2016-05-09 ~ 2026-05-08 | 18 symbols, 10년 daily 학습 가능 |
| `market_panel/1h` | 208,056 | 2023-06-05 ~ 2026-05-04 | intraday 1h |
| `market_panel/30m` | 33,765 | 2026-02-05 ~ 2026-05-04 | Yahoo interval 제한으로 짧음 |
| `market_panel/15m` | 67,207 | 2026-02-05 ~ 2026-05-04 | Yahoo interval 제한으로 짧음 |
| `eia_weekly` | 15,966 | 1982-08-25 ~ 2026-05-11 | 원유 수급/재고 |
| `cftc_cot_weekly` | 3,776 | 2016-01-08 ~ 2026-05-10 | 포지셔닝 |
| `macro_panel/fred_daily_wide` | 16,402 | 1962-01-02 ~ 2026-05-01 | macro rates/indices |
| `public_market_news` | 2,240 | 2026-01-11 ~ 2026-05-11 | Yahoo Finance RSS 340 + Google News RSS 1,900 |
| `event_context_daily` | 1,080 | 2026-03-13 ~ 2026-05-11 | 아직 새 Google News 전체가 반영되지 않은 기존 LLM context |

충분성 판단:

- 1d 가격/수급/포지셔닝 데이터는 h30 운영 artifact와 7/14/30 표시 길이 실험을 시작하기에 충분합니다.
- 30m/15m 데이터는 기간이 너무 짧아 deep model 일반화 평가에는 부족합니다.
- 뉴스는 340건에서 2,240건으로 늘었지만 기간이 2026년 1월 이후라 장기 regime 학습에는 여전히 부족합니다.
- `event_context_daily`는 아직 2,240건 뉴스 전체를 반영하지 못했습니다. LLM API 일일 500회 한도 때문에 cache/resume 방식으로 여러 날에 나눠 처리해야 합니다.

## 아직 부족하거나 제한적인 데이터

| 부족 데이터 | 영향 | 해결 방식 |
| --- | --- | --- |
| CME futures curve/settlement 장기 데이터 | term structure, roll yield, curve slope feature 부족 | CME DataMine/정산가 CSV를 확보해 `fetch_cme_settlements.py --manual-csv`로 적재 |
| 더 긴 뉴스 history | `oil_context_fusion`이 장기 regime별 뉴스 반응을 충분히 학습하기 어려움 | GDELT rate limit을 피해 기간을 나눠 재수집하거나 licensed news CSV를 `NEWS_EVENTS_PATH`로 추가 |
| 실측 calibration residual | quantile band를 검증된 confidence interval로 부를 수 없음 | rolling backtest 후 `scripts/evaluate/calibrate_quantiles.py` 실행 |
| Intraday fundamental/event alignment | 1h 이하 interval에서 주간/일간 feature의 release timing 정밀도 부족 | `feature_available_at`을 실제 발표 시각 기준으로 보강 |
| Vendor-grade market data | yfinance 결측/수정/지연 가능성 | Stooq, broker/vendor CSV, database provider를 추가 source로 저장 |

라이선스 데이터는 무단 scraping하지 않습니다. 필요한 경우 사용자가 정식으로 받은 CSV/API export를 `data/external` 아래에 두면 ingest script로 처리합니다.

## Provider

Provider 구현은 `market_ai/data/providers`에 있습니다.

- `market_price_provider.py`: yfinance/Stooq market panel 수집
- `eia_provider.py`: EIA petroleum bulk/API/manual CSV 정규화
- `cftc_provider.py`: CFTC COT ZIP/CSV/manual CSV 정규화
- `cme_provider.py`: CME settlement manual/URL CSV 정규화
- `fred_provider.py`: FRED macro series 수집
- `public_news_provider.py`: Yahoo RSS, Google News RSS, GDELT public news 수집

Provider는 raw cache와 processed output을 분리합니다. 실패하면 status/warning을 기록하고 production에서 synthetic fallback을 만들지 않습니다.

## 저장 위치

- `data/raw`: provider 원본 수집 데이터
- `data/interim`: 중간 결합/상태 데이터
- `data/processed`: 학습과 inference에 바로 쓰는 정제 데이터
- `data/features`: feature matrix 또는 학습용 파생 데이터
- `data/external`: 사용자가 제공하는 CSV, event file, licensed export
- `data/manifests`: 데이터 inventory와 latest snapshot
- `configs/symbol_universe.yaml`: 학습 universe 정의

## Market Panel 생성

```bash
.venv/bin/python scripts/data/fetch_market_prices.py --universe oil_core --interval 1d --period 10y
.venv/bin/python scripts/data/fetch_market_prices.py --universe oil_core --interval 1h --period 730d
.venv/bin/python scripts/data/fetch_market_prices.py --universe oil_core --interval 30m --period 60d
.venv/bin/python scripts/data/fetch_market_prices.py --universe oil_core --interval 15m --period 60d
```

출력은 `data/processed/market_panel/{interval}/panel.parquet`를 우선 사용하고, parquet engine이 없으면 `panel.csv`로 저장됩니다. 현재 학습 명령은 CSV fallback을 읽을 수 있습니다.

## EIA/CFTC/CME 생성

EIA와 CFTC는 기본적으로 공개 bulk/ZIP 또는 manual CSV를 처리할 수 있습니다.

```bash
.venv/bin/python scripts/data/fetch_eia_petroleum.py
.venv/bin/python scripts/data/fetch_cftc_cot.py
```

Manual CSV가 있을 때:

```bash
.venv/bin/python scripts/data/fetch_eia_petroleum.py --manual-csv data/external/fundamentals/eia_petroleum_weekly.csv
.venv/bin/python scripts/data/fetch_cftc_cot.py --manual-csv data/external/fundamentals/cftc_cot_weekly.csv
.venv/bin/python scripts/data/fetch_cme_settlements.py --manual-csv data/external/fundamentals/cme_settlements.csv
```

Manual CSV 최소 schema:

- EIA: `report_date` 또는 `date`, 그리고 `crude_stocks` 같은 wide metric column 또는 `series_id,value`
- CFTC: `report_date` 또는 `date`, `open_interest`, managed money/commercial long-short 계열
- CME: `trade_date` 또는 `date`, `settle` 또는 `settlement`, 가능하면 `contract` 또는 `contract_month`

## News/LLM Event Context 생성

Local deterministic context:

```bash
.venv/bin/python scripts/data/build_event_context.py \
  --news-path data/raw/news/public_market_news.csv \
  --symbols CL=F,BZ=F,NG=F \
  --mode local_rules
```

Google Gemma/Gemini 같은 external LLM context:

```bash
.venv/bin/python scripts/data/build_event_context.py \
  --news-path data/raw/news/public_market_news.csv \
  --symbols CL=F,BZ=F,NG=F,RB=F,HO=F,GC=F,SI=F,HG=F,DX-Y.NYB,EURUSD=X,USDKRW=X,JPY=X,SPY,QQQ,^GSPC,^VIX,XLE,USO \
  --mode google_generative \
  --live
```

LLM API 한도가 있는 경우 `llm_context_cache.jsonl`에 처리 결과가 행 단위로 즉시 append됩니다. 같은 명령을 다시 실행하면 `symbol/date/news_hash`가 같은 행은 cache hit로 건너뜁니다.

```bash
.venv/bin/python scripts/data/build_event_context.py \
  --news-path data/raw/news/public_market_news.csv \
  --symbols CL=F,BZ=F,NG=F,RB=F,HO=F,GC=F,SI=F,HG=F,DX-Y.NYB,EURUSD=X,USDKRW=X,JPY=X,SPY,QQQ,^GSPC,^VIX,XLE,USO \
  --mode google_generative \
  --live \
  --start 2026-01-11 \
  --end 2026-05-11 \
  --news-limit-per-context 10 \
  --llm-batch-size 10 \
  --llm-min-interval-seconds 4.2 \
  --progress-every 50
```

`--news-limit-per-context`는 한 `symbol/date` context에 넣는 최근 뉴스 개수입니다. `--llm-batch-size`는 여러 `symbol/date` context를 한 external LLM request로 묶는 개수입니다. `--llm-min-interval-seconds`는 RPM 한도를 넘기지 않기 위한 request 간 최소 대기 시간입니다. 새로 전부 다시 계산하고 싶을 때만 `--no-resume-cache`를 사용합니다.

## Real Dataset Orchestration

가능한 공개 데이터 전체를 한 번에 구축합니다.

```bash
.venv/bin/python scripts/data/build_real_dataset.py \
  --universe oil_core \
  --interval 1d \
  --period 10y \
  --news-timespan 3m \
  --news-maxrecords 30 \
  --skip-stooq-secondary
```

GDELT는 rate limit이 있을 수 있으므로 실패가 나도 Yahoo RSS 기반 뉴스 수집은 계속됩니다.

## Deep Dataset

`market_ai/data/deep_dataset.py`는 `DeepLearningSample`을 생성합니다.

- `x_price`: log return, vol-scaled return, range, rolling volatility, momentum, drawdown, autocorr, trend, skew/kurtosis, cycle feature
- `x_cross_asset`: related return/correlation/spread/relative strength/risk proxy/missing indicator
- `x_event_context`: event/LLM context vector
- `x_static`: current price, realized volatility, lookback, horizon

Target은 `future cumulative log return / recent_realized_volatility`입니다. Raw future price를 직접 target으로 학습하지 않습니다.

## Split과 No-Lookahead

Random split을 사용하지 않습니다. 각 symbol의 뒤쪽 구간을 validation/test로 쓰는 time-based split을 사용합니다. Feature는 `feature_available_at <= as_of_time` 조건을 만족하는 값만 들어갑니다.

## Manifest 업데이트

```bash
.venv/bin/python scripts/data/build_data_inventory.py
```

Manifest에는 dataset name, source, path, symbol/series, frequency, start/end, rows, columns, generated time, point-in-time safety flag, notes를 기록합니다.
