# 로드맵

이 프로젝트는 WTI 유가 예측 전용 모델과 dashboard를 안정화합니다. 현재 우선순위는 “데이터 신뢰도 -> LLM context 안정화 -> 단일 모델 재학습 -> backtest/calibration -> frontend 운영성” 순서입니다.

## 즉시 우선순위

1. Google Gemma/Gemini LLM 연결을 live로 검증합니다.
2. `data/raw/news/public_market_news.csv`를 기반으로 live LLM event context를 재생성합니다.
3. `oil_context_fusion` 1D h30 운영 artifact를 재학습하고 고정 30일 경로와 1주/2주/한달 endpoint 표시를 검증합니다.
4. CME settlement/curve CSV를 확보해 `cme_curve_daily.csv`를 생성합니다.
5. Rolling backtest와 quantile calibration을 실행합니다.
6. `/api/dashboard-analysis`, `/api/market-context`, frontend marker/panel을 실제 차트에서 검증합니다.

## 데이터 확장

- yfinance 외 보조 가격 source를 추가합니다.
- CME futures curve, settlement, volume, open interest를 장기 feature로 넣습니다.
- 뉴스 history를 최소 수년 단위로 늘립니다.
- EIA/CFTC/FRED release timestamp를 더 정밀하게 관리합니다.
- 데이터 inventory와 latest snapshot을 정례적으로 갱신합니다.

## 모델 확장

- `oil_core`에서 원유/에너지 중심 모델을 안정화합니다.
- ETF, FX, metals, indices, rates, crypto는 독립 예측 대상이 아니라 유가 보조 feature로 유지합니다.
- LLM은 context encoder로만 유지하고, 숫자 예측은 time-series model이 담당합니다.
- Coverage가 측정된 뒤에만 calibrated interval 표현을 사용합니다.

## Frontend 확장

- Context marker와 뉴스 card를 모델 진단용 UI로 정리합니다.
- Forecast scenario 설명을 data quality와 함께 표시합니다.
- UI 규모가 커지면 chart, controls, panels, api, state 모듈로 분리합니다.

## 유지해야 할 원칙

- `/api/chart` compatibility를 명시적 제거 전까지 유지합니다.
- Artifact와 metadata는 source code와 분리합니다.
- Production에서 mock/synthetic fallback을 조용히 사용하지 않습니다.
- 한국어 원본과 영어 mirror 문서를 항상 함께 갱신합니다.
