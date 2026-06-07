# LLM Context

LLM은 이 프로젝트에서 숫자 가격 예측기가 아니라 시장 뉴스와 이벤트를 구조화하는 context encoder입니다. 딥러닝 모델은 LLM이 만든 context vector를 입력 feature 중 하나로 사용하지만, `p50`, `p90`, target price, 미래 가격 경로는 LLM이 만들 수 없습니다.

## 현재 사용 방식

현재 의도한 흐름은 다음과 같습니다.

```text
뉴스/이벤트 원문
-> LLM context encoder
-> 방향성, 영향도, 불확실성, 이벤트 수, 설명, event embedding
-> 최근 1/3/7/30일 원시 뉴스 풀의 뉴스량, 방향 압력, 섹터 압력, 소스 다양성
-> data/processed/event_context/event_context_daily.csv
-> oil_context_fusion의 x_event_context 입력
-> 시계열 모델이 volatility-scaled cumulative log return distribution 예측
-> 가격은 current_price * exp(predicted_cumulative_log_return_h)로 복원
```

즉, 사용자가 질문한 “뉴스 -> 뉴스 내용을 LLM이 읽고 점수로 변환 -> 모델 input 중 하나”가 맞습니다. 다만 LLM 점수는 투자 의견이나 가격 목표가 아니라 point-in-time context feature입니다. 로컬 규칙은 14만 건 이상의 뉴스에서 날짜/심볼/주제 후보를 고르고 원시 뉴스 풀 통계를 계산하는 전처리 또는 개발 진단용입니다. 실제 학습용 event context에서 외부 LLM 호출이 실패하면 로컬 추정값으로 조용히 채우지 않고, 실패 날짜를 다시 LLM으로 처리한 뒤 병합합니다.

학습 sample에서는 origin 날짜 하나만 보지 않습니다. 현재 `lookback=128` 기준으로 각 origin 이전 128일 창 안의 event context를 집계해 `x_event_context [27]`로 넣습니다. 앞의 13개 feature는 LLM 또는 deterministic encoder가 만든 방향성/영향도/불확실성/event embedding이고, 뒤의 14개 feature는 LLM 입력으로 선택되지 않은 원시 뉴스 풀 전체에서 계산한 뉴스량, 선택 coverage, 상승/하락 압력, 에너지/지정학/거시/수급 압력, 소스 다양성입니다. Event count 계열은 합산하고, 방향성/영향도/품질 계열은 impact score와 recency decay로 가중 평균합니다. 이렇게 해야 최신 지정학/수급 충격 신호가 오래된 뉴스에 희석되지 않고, LLM에 들어가는 기사 수 제한이 모델 입력 병목으로 이어지지 않습니다.

## 실시간 Dashboard 파이프라인

현재 화면에서 종목을 열거나 주기/예측 길이를 바꾸면 다음 순서로 동작합니다.

1. Frontend가 `/api/models?interval={주기}&horizon={예측길이}`를 먼저 호출합니다.
   이 응답은 각 모델이 현재 조합에서 실행 가능한지 알려줍니다. Artifact가 없는 딥러닝 모델은 화면 토글만 비활성화하고 forecast 요청에는 넣지 않습니다.

2. Frontend가 `/api/forecast`를 호출합니다.
   이때 요청 모델은 현재 조합에서 사용 가능한 모델 전체입니다. 사용자가 모델 chip을 눌러도 재계산하지 않고, 이미 받은 `model_paths` 중 표시/숨김만 바꿉니다.

3. Backend는 yfinance/processed provider에서 OHLCV candle을 가져옵니다.
   화면과 API 예시는 yfinance provider symbol을 기본으로 사용합니다. 예를 들어 원유는 `CL=F`, 브렌트유는 `BZ=F`, 달러/원은 `USDKRW=X`입니다. `NYMEX:CL1!` 같은 기존 TradingView alias는 backward compatibility 목적으로만 정규화해서 받습니다.

4. Backend는 선택 모델의 숫자 예측을 계산합니다.
   `motif`, `pattern_mlp`, baseline, deep model이 각각 누적 로그수익률 또는 가격 경로를 만들고, 최종 가격 경로는 `current_price * exp(predicted_cumulative_log_return_h)` 방식으로 복원됩니다. LLM은 이 단계에서 가격 숫자를 만들지 않습니다.

5. 단일 운영 모델 `oil_context_fusion`이 선택되면 Backend가 `build_live_event_context()`를 호출합니다.
   이 함수는 Google News RSS와 Yahoo Finance RSS에서 최대 `news_limit`개의 최신 공개 뉴스를 가져오고, 종목별 query/relevance filter를 적용합니다. 현재 기본 실시간 context 호출은 dashboard에서 최대 60개, 채팅에서는 최대 24개를 요청합니다.

6. 뉴스 DataFrame은 `RawNewsItem` 목록으로 변환되어 encoder에 들어갑니다.
   외부 LLM이 켜져 있고 rate guard가 허용하면 LLM encoder가 호출됩니다. 그렇지 않으면 `local_rules`가 같은 schema를 deterministic하게 생성합니다. LLM 출력 schema는 `overall_bias`, `impact_score`, `uncertainty`, `event_count`, `explanation`, `event_embedding[13]`입니다. 이후 builder가 최근 원시 뉴스 풀의 14개 aggregate feature를 덧붙여 모델 입력은 27차원이 됩니다.
   `local_rules`와 raw-news aggregate는 전쟁, 공격, 제재, 호르무즈/홍해 같은 키워드를 `geopolitical_supply_shock` 성격의 공급 리스크로 승격합니다. 그래서 “가격은 잠시 하락했지만 미국-이란 전쟁으로 공급 차질 리스크가 커졌다” 같은 혼합 뉴스도 단순 neutral로 희석되지 않습니다.

7. Encoder 출력은 한 행짜리 `context_frame`으로 만들어져 deep model 입력에 들어갑니다.
   `oil_context_fusion`은 이 `x_event_context`를 context expert gating, uncertainty, confidence 계산에 사용합니다. LLM이 직접 가격선을 그리는 것이 아니라, 뉴스 context vector가 시계열 모델의 입력 feature로 들어가고 시계열 모델이 예측선을 계산합니다.

8. `/api/market-context?live=1`은 같은 live context payload를 UI에 반환합니다.
   차트 news marker, popover, “뉴스 해석” 패널은 이 payload의 `news`, `context_points`, `chart_context_points`를 같은 기준으로 사용합니다. 표시용 context는 중복 headline/source/date를 제거하고, 내부 placeholder나 `-`만 있는 설명은 화면에 내보내지 않습니다.

9. `/api/dashboard-analysis`는 이미 계산된 예측 경로와 뉴스 context 요약을 한 번의 외부 LLM 호출로 설명합니다.
   이 endpoint는 AI 시황 해설, 뉴스 해석, 예측 리포트를 JSON 형태로 한 번에 받아서 각 panel로 나눠 출력합니다. 개별 `/api/model-commentary`, `/api/market-context`, `/api/report` 경로는 호환/진단용으로 남기지만, 운영 dashboard는 외부 LLM 호출 수를 줄이기 위해 통합 endpoint를 우선 사용합니다. 서버는 60초 window에서 외부 LLM 호출을 제한하고, dashboard-analysis cache를 사용합니다. 응답 JSON 뒤에 모델이 불필요한 텍스트를 덧붙여도 첫 JSON object만 파싱해 복구합니다.

10. `/api/assistant-chat`은 사용자가 직접 입력한 짧은 질문에 답합니다.
    이 endpoint도 숫자 가격 예측을 새로 만들지 않습니다. 화면에서는 답변 생성 중 assistant 말풍선의 점 애니메이션을 표시하고, 요청 중복 submit을 막습니다.

## 허용되는 역할

- 뉴스, 경제 이벤트, 수급 이벤트를 구조화된 market context로 변환합니다.
- `overall_bias`, `impact_score`, `uncertainty`, event embedding을 생성합니다.
- 과거 뉴스와 당시 context 해석을 dashboard에 보여주는 설명 자료로 사용합니다.
- `oil_context_fusion`에서 gating, confidence, uncertainty에 간접적으로 영향을 줍니다.
- `/api/market-context`에서 과거 context marker와 시나리오 해설을 제공합니다.
- `/api/dashboard-analysis`에서 이미 계산된 forecast와 뉴스 evidence를 바탕으로 시황/뉴스/리포트 문장을 작성합니다.

## 금지되는 역할

- LLM이 `p50`, `p90`, target price, future return path를 생성하면 안 됩니다.
- LLM output이 deep model 또는 baseline의 numeric forecast를 덮어쓰면 안 됩니다.
- LLM이 매수/매도 지시나 확정적인 목표가를 만들면 해당 output은 validator에서 무시해야 합니다.
- Coverage가 실제로 측정되기 전에는 forecast band를 검증된 confidence interval이라고 부르지 않습니다.
- Backtest 기준 해설에서 “현재/최근/지금/금일” 같은 live 표현으로 사용자를 혼동시키면 안 됩니다. `origin_time`이 있으면 기준 시점의 절대 날짜와 시간을 첫 문장에 포함합니다.

## 구현체

- `LocalEventContextEncoder`: CSV/JSON event file을 deterministic rule로 읽습니다. 개발, 진단, 또는 LLM 입력 뉴스가 실제로 없는 날짜의 빈 context 표현에만 사용합니다.
- `OpenAICompatibleLLMEventEncoder`: OpenAI-compatible chat completions endpoint를 호출합니다.
- `LocalHTTPLLMEventEncoder`: Ollama/vLLM 호환 local HTTP endpoint를 호출합니다.
- `OfflineFileLLMEventEncoder`: 사전 계산된 JSON/JSONL context cache를 읽습니다.
- `NullLLMEventEncoder`: LLM context를 완전히 끕니다.

외부 LLM 호출은 `--live`와 `ENABLE_EXTERNAL_LLM_CALLS=true`가 모두 있어야 실행됩니다. 학습용 외부 LLM 빌드는 기본적으로 strict 모드입니다. API key가 없거나 호출이 실패하면 `local_rules` 결과를 학습 CSV에 넣지 않고 빌드를 중단합니다. 이때 실패 날짜만 cache를 유지한 채 다시 처리해야 합니다. 개발 중 fallback 동작을 일부러 확인할 때만 `--allow-external-llm-fallback`을 사용합니다.

Dashboard runtime의 통합 LLM 호출도 외부 LLM 설정이 없으면 명시적 unavailable 상태를 반환합니다. Frontend는 언어 전환, live/backtest 전환, backtest origin 변경 때 이전 dashboard-analysis 응답을 stale 처리하고 panel/marker state를 먼저 비워 오래된 뉴스가 새 차트에 남지 않게 합니다.

## Google Gemma/Gemini 설정

Google API key를 받았다면 Gemma hosted API는 native `generateContent` endpoint로 호출합니다. API key는 문서, Git, 채팅에 노출하지 않습니다.

```bash
export ENABLE_LLM_CONTEXT=true
export ENABLE_EXTERNAL_LLM_CALLS=true
export LLM_CONTEXT_MODE=google_generative
export LLM_API_KEY="YOUR_GOOGLE_API_KEY"
export LLM_API_BASE="https://generativelanguage.googleapis.com/v1beta"
export LLM_MODEL="gemma-3-27b-it"
```

`LLM_MODEL`은 Google AI Studio 또는 model list에서 실제로 표시되는 모델 ID를 사용합니다. “Gemma 4”라는 이름을 UI에서 봤더라도 API model id가 `gemma4`라는 뜻은 아닙니다. Google OpenAI-compatible endpoint는 Gemini model 예시가 중심이고, Gemma hosted API는 `models/{model}:generateContent` native endpoint를 사용합니다.

## export 확인법

현재 shell에서 값이 설정됐는지 확인합니다.

```bash
echo "$ENABLE_LLM_CONTEXT"
echo "$ENABLE_EXTERNAL_LLM_CALLS"
echo "$LLM_CONTEXT_MODE"
echo "$LLM_API_BASE"
echo "$LLM_MODEL"
test -n "$LLM_API_KEY" && echo "LLM_API_KEY is set" || echo "LLM_API_KEY is missing"
```

API key 값을 그대로 출력하지 마십시오. 길이만 확인하려면 다음을 사용합니다.

```bash
python - <<'PY'
import os
key = os.environ.get("LLM_API_KEY", "")
print("LLM_API_KEY set:", bool(key), "length:", len(key))
PY
```

`export`는 현재 shell과 그 shell에서 실행한 child process에만 적용됩니다. 터미널을 닫으면 사라지고, 다른 터미널에는 자동으로 전달되지 않습니다. 영구적으로 쓰려면 `~/.zshrc`에 넣거나 프로젝트의 `.env`에 저장합니다. 이 저장소의 주요 server/script entrypoint는 프로젝트 루트 `.env`를 자동 로드합니다.

```bash
cp .env.example .env
```

shell에서 `echo "$LLM_MODEL"`을 실행하면 `.env` 값이 보이지 않을 수 있습니다. 프로젝트가 실제로 읽는 값은 다음으로 확인합니다.

```bash
.venv/bin/python - <<'PY'
from market_ai.config import get_settings
s = get_settings()
print("enable_llm_context:", s.enable_llm_context)
print("enable_external_llm_calls:", s.enable_external_llm_calls)
print("llm_model:", s.llm_model)
print("llm_api_base:", s.llm_api_base)
print("llm_api_key_set:", bool(s.llm_api_key))
PY
```

## 연결 검증

외부 호출 없이 adapter와 safety rule만 검증합니다.

```bash
.venv/bin/python scripts/llm/test_llm_context.py --mode openai_compatible --dry-run
```

실제 Google endpoint 호출을 검증합니다.

```bash
.venv/bin/python scripts/llm/test_llm_context.py --mode google_generative --live
```

성공 기준은 `safety_check_passed=true`이고 `warnings`에 `External LLM fallback`이 없는 상태입니다. `model not found`, `404`, `unsupported response_format`이 나오면 모델 ID 또는 endpoint 호환성 문제입니다.

## Event Context 생성

현재 뉴스 파일을 LLM context로 변환합니다.

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
  --llm-min-interval-seconds 5.0
```

`--news-limit-per-context`는 LLM이 직접 읽는 bounded input 수입니다. 현재 historical cache는 날짜별 최근 7일 최신 8개 뉴스로 생성됐습니다. 이 제한은 LLM 비용과 토큰 길이를 제어하기 위한 것이며, 모델 입력 병목을 피하기 위해 builder는 별도로 최근 1/3/7/30일의 전체 원시 뉴스 풀에서 14개 aggregate feature를 계산합니다.

외부 LLM 모드에서 `--live`를 쓰면 fallback row가 감지되는 즉시 빌드가 실패합니다. 실패 원인이 quota/rate limit이면 API 사용량을 초기화하거나 간격을 늘린 뒤 실패 날짜만 다시 실행합니다. 이렇게 해야 LLM을 거치지 않은 낮은 품질의 context가 모델 학습에 섞이지 않습니다.

출력은 다음 두 파일입니다.

- `data/processed/event_context/event_context_daily.csv`
- `data/processed/event_context/llm_context_cache.jsonl`

2026-06-05 정리 결과 CL=F event context는 3,476행, event/context 입력 차원은 27개이며, 외부 LLM fallback은 0건입니다. Google free-tier 429가 발생하면 cache를 유지한 채 `--llm-batch-size 1`과 충분한 `--llm-min-interval-seconds`로 실패 날짜만 다시 처리합니다.

## 학습 연결

LLM context를 쓰는 운영 모델은 `oil_context_fusion`입니다.

```bash
.venv/bin/python scripts/train/train_deep_fusion_models.py \
  --model oil_context_fusion \
  --interval 1d \
  --horizon 30 \
  --lookback 128 \
  --universe oil_core \
  --llm-context \
  --event-context data/processed/event_context/event_context_daily.csv \
  --events-path data/raw/news/public_market_news.csv \
  --use-processed-data \
  --market-panel data/processed/market_panel/1d/panel.csv \
  --oil-fundamentals data/processed/oil_fundamentals/eia_weekly.csv \
  --cot data/processed/oil_fundamentals/cftc_cot_weekly.csv \
  --macro-panel data/processed/macro_panel/fred_daily_wide.csv \
  --max-samples 0 \
  --epochs 10 \
  --patience 3 \
  --batch-size 128 \
  --device mps \
  --force
```

운영 배포용 최종 artifact를 최신 전체 데이터로 학습할 때는 holdout 성능을 별도로 기록한 뒤 `--fit-final-all-data`를 추가합니다.

Dashboard server도 같은 환경변수를 가진 shell에서 실행해야 LLM context와 `/api/market-context`가 같은 설정을 봅니다. 이미 서버가 떠 있다면 환경변수 변경 후 재시작해야 합니다.
