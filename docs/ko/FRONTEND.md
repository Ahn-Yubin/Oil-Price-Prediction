# 프론트엔드

Frontend는 TradingView Lightweight Charts 스타일의 forecast overlay dashboard입니다. 첫 화면은 마케팅 페이지가 아니라 실제 chart와 모델/context 정보를 보여주는 작업 화면입니다.

## 위치

- `frontend/index.html`: dashboard shell과 panel markup
- `frontend/src/main.js`: chart rendering, API fetch, marker rendering, interaction
- `frontend/src/dashboard.css`: responsive layout과 dashboard style
- `frontend/src/api`: 향후 API client 분리 위치

## API 사용

UI는 `/api/forecast`를 우선 호출하고, 필요한 경우 `/api/chart` compatibility payload를 사용합니다. `/api/chart`는 기존 overlay와 호환되어야 합니다.

추가 context panel은 `/api/market-context`를 호출합니다.

- `news`: 최근 headline과 source
- `context_points`: chart marker로 표시할 이벤트/context 날짜
- `scenario_commentary`: bull/base/bear 시나리오 해설
- `llm_context_summary`: LLM context 사용 여부와 상태

## Chart 표시

현재 표시 대상:

- historical candles
- forecast p50 path
- p10/p90 또는 p05/p95 band
- bull/base/bear scenario summary
- 과거 뉴스/context marker
- event/context card list

Marker는 뉴스/event가 있는 날짜 근처에 표시합니다. LLM이 숫자 가격을 예측한 marker가 아니라, 해당 날짜에 어떤 context가 있었는지 설명하는 marker입니다.

## UX 원칙

- 데이터 품질과 warning을 숨기지 않습니다.
- Mock/fallback data가 production에서 조용히 섞이면 안 됩니다.
- Forecast band는 coverage가 검증되기 전까지 confidence interval이라고 표시하지 않습니다.
- 모델 selector에서 unavailable deep artifact는 warning과 함께 다뤄야 합니다.
- 운영 tool UI는 조용하고 정보 밀도가 높은 방향을 유지합니다.

## 향후 구조

UI가 커지면 다음처럼 분리합니다.

- `frontend/src/components/chart`
- `frontend/src/components/controls`
- `frontend/src/components/panels`
- `frontend/src/api`
- `frontend/src/state`
