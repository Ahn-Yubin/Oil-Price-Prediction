# 프론트엔드

Frontend는 WTI 유가(`CL=F`) 전용 TradingView Lightweight Charts 스타일의 forecast overlay dashboard입니다. 첫 화면은 마케팅 페이지가 아니라 실제 chart와 모델/context 정보를 보여주는 작업 화면입니다.

## 위치

- `frontend/index.html`: dashboard shell과 panel markup
- `frontend/src/main.js`: chart rendering, API fetch, marker rendering, interaction
- `frontend/src/dashboard.css`: responsive layout과 dashboard style
- `frontend/src/api`: 향후 API client 분리 위치

## API 사용

UI는 `/api/forecast`를 우선 호출하고, 필요한 경우 `/api/chart` compatibility payload를 사용합니다. `/api/chart`는 기존 overlay와 호환되어야 합니다.

데스크톱 화면은 3컬럼 구조입니다. 왼쪽은 차트와 AI 채팅 패널, 가운데는 AI 시황 해설/뉴스 해석, 오른쪽은 예측 리포트입니다. 화면의 세 AI 패널은 `/api/dashboard-analysis`를 우선 호출해 한 번의 외부 LLM 응답에서 시황, 뉴스, 리포트를 받아 나눠 표시합니다. 차트 상단 mode control은 `백테스트 / 라이브 / 시나리오` 3분할입니다. 차트에서 과거 candle을 선택해 backtest를 실행하면 `/api/backtests/visualization`을 호출합니다. Scenario mode에서 시나리오 폴더에 미래 이벤트를 묶으면 `/api/scenarios/forecast`를 호출하고, 반환된 path를 시나리오별 overlay로 live forecast 위에 표시합니다. Backtest 기준시점이 있을 때 뉴스 해석은 실시간 뉴스가 아니라 같은 `origin_time`까지의 point-in-time 뉴스/이벤트 컨텍스트를 요청하고, 해설 문장은 절대 날짜/시각 기준으로 작성합니다.

- `news`: 최근 headline과 source
- `context_points`: chart marker와 뉴스 해석 목록에 사용하는 이벤트/context 날짜
- `chart_context_points`: 차트 marker 전용으로 중복과 밀집도를 정리한 이벤트/context 날짜
- `scenario_commentary`: backend compatibility field. 화면에서는 bull/base/bear 카드 대신 뉴스와 LLM 해석만 표시합니다.
- `llm_context_summary`: LLM context 사용 여부와 상태

Backtest visualization payload는 `/api/chart`와 같은 기본 chart key에 다음 field를 추가합니다.

- `origin_time`: forecast를 다시 생성한 과거 candle time
- `actual_future_candles`: origin 이후 실제 OHLCV
- `backtest`: history/future row 수와 horizon metadata

Scenario forecast payload는 목록형 하단 패널에 보관됩니다.

- `points`: 시나리오 차트 overlay. 첫 점은 현재 가격 anchor입니다.
- `llm_context_summary`: LLM이 만든 bias/impact/uncertainty, `scenario_override` 출처, horizon별 `model_context_schedule`
- `llm_context`: 검증된 구조화 이벤트 컨텍스트
- `warning_objects`: event_time 누락, LLM fallback, artifact/data 품질 경고

Dashboard analysis payload는 단일 운영 모델의 forecast path와 뉴스 evidence를 바탕으로 LLM 해설을 제공합니다. 이 해설은 모델 구조를 기술적으로 설명하는 글이 아니라, 뉴스와 차트 흐름을 근거로 예측 방향을 설명하는 애널리스트식 시황 코멘트입니다. LLM은 새 가격 예측기가 아니라 이미 생성된 모델 출력의 해설자로만 사용합니다. UI는 설정 언어에 맞춰 본문형 문단으로 표시하며, 한국어 모드에서는 영어 뉴스 제목을 그대로 나열하지 않습니다. 개별 `/api/model-commentary`, `/api/market-context`, `/api/report`는 호환/진단 경로로 남아 있습니다.

## Chart 표시

현재 표시 대상:

- `CL=F` historical candles
- forecast display path와 1주/2주/한달 구간 endpoint marker
- display path를 중심으로 재배치된 p10/p90 또는 p05/p95 band
- 과거 뉴스/context marker. Forecast 기간에 속한 미래 뉴스가 backtest chart에 섞이지 않도록 active request key와 origin key로 필터링합니다.
- 뉴스 headline과 해당 뉴스/이벤트에 대한 LLM 해석 card list
- 선택 origin 이후 실제 candle의 반투명 overlay
- 가운데 AI 시황 해설/뉴스 해석 panel과 오른쪽 예측 리포트 panel

Marker는 뉴스/event가 있는 날짜 근처에 표시합니다. LLM이 숫자 가격을 예측한 marker가 아니라, 해당 날짜에 어떤 context가 있었는지 설명하는 marker입니다.

## 현재 UX 조정

- 심볼 검색/입력창은 제거했습니다. 화면과 API 요청은 항상 `CL=F`를 사용합니다.
- 운영 화면은 `CL=F · 1D · 30일` 예측을 기본으로 고정합니다. 1H/15M/30M은 연구와 API 검증 대상으로 남기고, 현재 UI에서는 1D h30 artifact를 안정적으로 보여주는 데 집중합니다.
- 예측 기간 selector는 제거했습니다. 내부 모델은 h30 artifact 하나를 실행하고, 화면에는 같은 30일 경로 위에 1주/2주/한달 endpoint를 점과 텍스트로 표시합니다.
- 사용자-facing 모델은 `oil_context_fusion` 하나입니다. 기존 모델들은 내부 benchmark/fallback으로만 남깁니다.
- `/api/forecast`, `/api/chart`, `/api/dashboard-analysis`, `/api/backtests/visualization` 호출에 같은 horizon/origin/language key를 전달합니다.
- Scenario mode는 live forecast를 기준선으로 유지하고, 사용자가 추가한 시나리오를 별도 line overlay로 표시합니다. 목록 안의 토글로 각 시나리오 표시 여부를 켜고 끌 수 있습니다.
- Scenario는 제목만 가진 폴더이고, 이벤트 입력은 제목, 발생 시점, 내용으로 구성됩니다. 발생 시점은 모델 예측 구간 안에서 horizon별 context 활성화 기준으로 사용됩니다.
- 데스크톱 가로 화면에서는 차트와 채팅을 같은 컬럼 안에서 2:1 비율로 배치하고, dashboard frame이 viewport 높이를 최대한 채우도록 grid row 높이를 조정합니다.
- AI 채팅 패널은 차트 아래에 있으며, 패널 전체가 message 공간입니다. 상단 제목과 하단 입력 영역은 시황/뉴스/리포트 패널 헤더와 같은 중립 유리 효과를 갖는 overlay로 고정합니다.
- 오른쪽/가운데 side panel은 viewport 조건에 따라 내부 스크롤을 가지며, 페이지 전체 스크롤과 충돌하지 않도록 max-height를 제한합니다.
- 뉴스 panel은 잡다한 scenario 설명 없이 기준시점 뉴스와 LLM 해석만 보여줍니다.
- 뉴스 panel은 internal placeholder(`Deterministic local event context encoder...`)나 비공개 진단 문구, `-`만 있는 해석을 표시하지 않습니다. 같은 뉴스 headline/source/date 조합은 dedupe합니다.
- AI 시황 해설과 예측 리포트는 점 목록 대신 본문형 문단을 사용합니다. 리포트는 데이터 상태, band 상태, 내부 context 점수 같은 개발자용 표현을 노출하지 않고 대중적인 시황 언어로 설명합니다.
- Backtest를 실행할 때는 기존 차트 time scale과 price scale을 최대한 유지합니다.
- Backtest가 활성화된 상태에서도 다른 candle을 클릭하면 새 origin으로 `/api/backtests/visualization`을 즉시 다시 호출합니다. Live chart로 돌아갔다가 다시 backtest를 켜지 않아도 됩니다.
- 언어 전환, live/backtest 전환, backtest origin 변경 중에는 이전 LLM 응답을 stale 처리합니다. 새 요청이 진행 중일 때 추가 갱신이 들어오면 pending refresh로 접수했다가 현재 요청 종료 후 최신 key로 한 번만 다시 실행합니다.
- 채팅 답변 생성 중에는 assistant 말풍선에 왼쪽부터 오른쪽으로 커지는 점 애니메이션을 표시합니다.
- 더 큰 개편안은 live chart와 backtest visualization을 별도 chart state로 분리하고, backtest panel visible toggle로 표시만 전환하는 방식입니다. 이 방식은 live 데이터 자동 갱신과 과거 origin 검증 상태를 더 명확히 분리할 수 있습니다.

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
