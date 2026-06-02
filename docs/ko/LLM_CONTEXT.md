# LLM Context

LLM은 이 프로젝트에서 숫자 가격 예측기가 아니라 시장 뉴스와 이벤트를 구조화하는 context encoder입니다. 딥러닝 모델은 LLM이 만든 context vector를 입력 feature 중 하나로 사용하지만, `p50`, `p90`, target price, 미래 가격 경로는 LLM이 만들 수 없습니다.

## 현재 사용 방식

현재 의도한 흐름은 다음과 같습니다.

```text
뉴스/이벤트 원문
-> LLM 또는 local_rules encoder
-> 방향성, 영향도, 불확실성, 이벤트 수, 설명, event embedding
-> data/processed/event_context/event_context_daily.csv
-> oil_context_fusion의 x_event_context 입력
-> 시계열 모델이 volatility-scaled cumulative log return distribution 예측
-> 가격은 current_price * exp(predicted_cumulative_log_return_h)로 복원
```

즉, 사용자가 질문한 “뉴스 -> 뉴스 내용을 LLM이 읽고 점수로 변환 -> 모델 input 중 하나”가 맞습니다. 다만 LLM 점수는 투자 의견이나 가격 목표가 아니라 point-in-time context feature입니다.

학습 sample에서는 origin 날짜 하나만 보지 않습니다. 현재 `lookback=128` 기준으로 각 origin 이전 128일 창 안의 event context를 집계해 `x_event_context [13]`로 넣습니다. Event count 계열은 합산하고, 방향성/영향도/품질 계열은 impact score로 가중 평균합니다.

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
   외부 LLM이 켜져 있고 rate guard가 허용하면 LLM encoder가 호출됩니다. 그렇지 않으면 `local_rules`가 같은 schema를 deterministic하게 생성합니다. 출력 schema는 `overall_bias`, `impact_score`, `uncertainty`, `event_count`, `explanation`, `event_embedding[13]`입니다.

7. Encoder 출력은 한 행짜리 `context_frame`으로 만들어져 deep model 입력에 들어갑니다.
   `oil_context_fusion`은 이 `x_event_context`를 context expert gating, uncertainty, confidence 계산에 사용합니다. LLM이 직접 가격선을 그리는 것이 아니라, 뉴스 context vector가 시계열 모델의 입력 feature로 들어가고 시계열 모델이 예측선을 계산합니다.

8. `/api/market-context?live=1`은 같은 live context payload를 UI에 반환합니다.
   차트 하단 news marker, popover, 우측 “뉴스와 시나리오” 패널은 이 payload의 `news`와 `context_points`를 같은 기준으로 사용합니다. 각 뉴스에는 현재 context point의 입력 팩터, 즉 방향성/중요도/불확실성/이벤트 수가 같이 표시됩니다.

9. `/api/model-commentary`와 `/api/assistant-chat`은 이미 계산된 예측 경로와 뉴스 context 요약을 설명합니다.
   이 둘도 숫자 가격 예측을 새로 만들지 않습니다. 호출 제한을 피하기 위해 서버는 60초 window에서 외부 LLM 호출을 제한하고, commentary/context cache를 사용합니다. 제한에 걸리면 deterministic 설명으로 대체하고 warning에 이유를 남깁니다.

## 허용되는 역할

- 뉴스, 경제 이벤트, 수급 이벤트를 구조화된 market context로 변환합니다.
- `overall_bias`, `impact_score`, `uncertainty`, event embedding을 생성합니다.
- 과거 뉴스와 당시 context 해석을 dashboard에 보여주는 설명 자료로 사용합니다.
- `oil_context_fusion`에서 gating, confidence, uncertainty에 간접적으로 영향을 줍니다.
- `/api/market-context`에서 과거 context marker와 시나리오 해설을 제공합니다.

## 금지되는 역할

- LLM이 `p50`, `p90`, target price, future return path를 생성하면 안 됩니다.
- LLM output이 deep model 또는 baseline의 numeric forecast를 덮어쓰면 안 됩니다.
- LLM이 매수/매도 지시나 확정적인 목표가를 만들면 해당 output은 validator에서 무시해야 합니다.
- Coverage가 실제로 측정되기 전에는 forecast band를 검증된 confidence interval이라고 부르지 않습니다.

## 구현체

- `LocalEventContextEncoder`: CSV/JSON event file을 deterministic rule로 읽습니다.
- `OpenAICompatibleLLMEventEncoder`: OpenAI-compatible chat completions endpoint를 호출합니다.
- `LocalHTTPLLMEventEncoder`: Ollama/vLLM 호환 local HTTP endpoint를 호출합니다.
- `OfflineFileLLMEventEncoder`: 사전 계산된 JSON/JSONL context cache를 읽습니다.
- `NullLLMEventEncoder`: LLM context를 완전히 끕니다.

외부 LLM 호출은 `--live`와 `ENABLE_EXTERNAL_LLM_CALLS=true`가 모두 있어야 실행됩니다. API key가 없거나 호출이 실패하면 local fallback이 사용되고 warning에 이유가 남습니다.

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
  --symbols CL=F,BZ=F,NG=F,RB=F,HO=F,GC=F,SI=F,HG=F,DX-Y.NYB,EURUSD=X,USDKRW=X,JPY=X,SPY,QQQ,^GSPC,^VIX,XLE,USO \
  --mode google_generative \
  --live
```

출력은 다음 두 파일입니다.

- `data/processed/event_context/event_context_daily.csv`
- `data/processed/event_context/llm_context_cache.jsonl`

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

Dashboard server도 같은 환경변수를 가진 shell에서 실행해야 LLM context와 `/api/market-context`가 같은 설정을 봅니다. 이미 서버가 떠 있다면 환경변수 변경 후 재시작해야 합니다.
