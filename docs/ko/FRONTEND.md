# 프론트엔드

Frontend는 WTI 유가(`CL=F`) 전용 TradingView Lightweight Charts 스타일의 forecast overlay dashboard입니다. 첫 화면은 마케팅 페이지가 아니라 실제 chart와 모델/context 정보를 보여주는 작업 화면입니다.

## 위치

- `frontend/index.html`: dashboard shell과 panel markup
- `frontend/src/main.js`: chart rendering, API fetch, marker rendering, interaction
- `frontend/src/dashboard.css`: responsive layout과 dashboard style
- `frontend/src/api`: 향후 API client 분리 위치

## API 사용

UI는 `/api/forecast`를 우선 호출하고, 필요한 경우 `/api/chart` compatibility payload를 사용합니다. `/api/chart`는 기존 overlay와 호환되어야 합니다.

오른쪽 context panel은 `/api/market-context`를 호출합니다. 차트에서 과거 candle을 선택해 backtest를 실행하면 `/api/backtests/visualization`을 호출합니다. 화면 하단 commentary panel은 `/api/model-commentary`를 호출합니다.

- `news`: 최근 headline과 source
- `context_points`: chart marker로 표시할 이벤트/context 날짜
- `scenario_commentary`: bull/base/bear 시나리오 해설
- `llm_context_summary`: LLM context 사용 여부와 상태

Backtest visualization payload는 `/api/chart`와 같은 기본 chart key에 다음 field를 추가합니다.

- `origin_time`: forecast를 다시 생성한 과거 candle time
- `actual_future_candles`: origin 이후 실제 OHLCV
- `backtest`: history/future row 수와 horizon metadata

Model commentary payload는 단일 운영 모델의 forecast path를 바탕으로 LLM 또는 deterministic fallback 해설을 제공합니다. 이 해설은 모델 구조를 기술적으로 설명하는 글이 아니라, 뉴스와 차트 흐름을 근거로 예측 방향을 설명하는 애널리스트식 시황 코멘트입니다. LLM은 새 가격 예측기가 아니라 이미 생성된 모델 출력의 해설자로만 사용합니다.

## Chart 표시

현재 표시 대상:

- `CL=F` historical candles
- forecast p50 path
- p10/p90 또는 p05/p95 band
- bull/base/bear scenario summary
- 과거 뉴스/context marker
- event/context card list
- 선택 origin 이후 실제 candle의 반투명 overlay
- 하단 full-width 모델 예측 해설

Marker는 뉴스/event가 있는 날짜 근처에 표시합니다. LLM이 숫자 가격을 예측한 marker가 아니라, 해당 날짜에 어떤 context가 있었는지 설명하는 marker입니다.

## 현재 UX 조정

- 심볼 검색/입력창은 제거했습니다. 화면과 API 요청은 항상 `CL=F`를 사용합니다.
- 주기는 1D와 1H만 제공합니다. 15M/30M은 데이터 기간과 뉴스/수급 정렬 품질이 부족해 운영 UI에서 제외하고, 먼저 1H/1D h30 통합 모델을 안정화합니다.
- 예측 기간 selector는 `7`, `14`, `30`입니다. 내부 모델은 주기별 h30 artifact 하나를 실행하고 사용자가 고른 길이만큼 앞부분을 표시합니다.
- 사용자-facing 모델은 `oil_context_fusion` 하나입니다. 기존 모델들은 내부 benchmark/fallback으로만 남깁니다.
- `/api/forecast`, `/api/chart`, `/api/market-context`, `/api/model-commentary`, `/api/backtests/visualization` 호출에 같은 horizon 값을 전달합니다.
- 차트 높이는 460px 기준으로 줄여 화면 전체 이동 스크롤이 더 잘 동작하게 했습니다.
- 오른쪽 context event list의 독립 내부 스크롤을 제거해 페이지 스크롤과 충돌하지 않게 했습니다.
- Backtest를 실행할 때는 기존 차트 time scale과 price scale을 최대한 유지합니다.

## UX 원칙

- 데이터 품질과 warning을 숨기지 않습니다.
- Mock/fallback data가 production에서 조용히 섞이면 안 됩니다.
- Forecast band는 coverage가 검증되기 전까지 confidence interval이라고 표시하지 않습니다.
- `oil_context_fusion` artifact가 unavailable이면 warning과 training command를 노출해야 합니다.
- 운영 tool UI는 조용하고 정보 밀도가 높은 방향을 유지합니다.

## 향후 구조

UI가 커지면 다음처럼 분리합니다.

- `frontend/src/components/chart`
- `frontend/src/components/controls`
- `frontend/src/components/panels`
- `frontend/src/api`
- `frontend/src/state`
