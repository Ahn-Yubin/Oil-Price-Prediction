# LLM Context

LLM은 시장 이벤트와 설명을 다루는 보조 계층입니다. 숫자 가격 예측은 시계열 모델과 baseline이 담당합니다.

## 허용되는 역할

- 뉴스, 경제 이벤트, 시장 이벤트를 구조화된 context로 변환합니다.
- 방향성 편향, 영향 강도, 불확실성, source quality를 요약합니다.
- Forecast response를 사람이 읽기 쉬운 설명으로 바꿉니다.
- `llm_context_seq_moe`에서는 gating과 confidence/uncertainty에만 영향을 줍니다.

## 금지되는 역할

- LLM이 `p50`, `p90`, target price, 직접적인 미래 return path를 생성하면 안 됩니다.
- LLM output이 시계열 모델의 numeric forecast를 덮어쓰면 안 됩니다.
- LLM이 목표가나 거래 지시를 만들면 해당 output은 validator에서 무시합니다.

## 구현

- `LocalEventContextEncoder`: CSV/JSON event file을 deterministic하게 읽습니다.
- `OpenAICompatibleLLMEventEncoder`: `ENABLE_EXTERNAL_LLM_CALLS=true`이고 API key가 있을 때만 외부 호출을 시도합니다.
- API key가 없거나 호출이 실패하면 local/null fallback을 사용합니다.
- Prompt는 `market_ai/llm/prompts`에 있습니다.

## Event 입력

기본 event file schema는 `data/external/events/README.md`에 있습니다. `NEWS_EVENTS_PATH`, `ECONOMIC_EVENTS_PATH`, `MARKET_EVENTS_PATH`로 파일을 연결할 수 있습니다. Event timestamp는 forecast `as_of_time` 이하만 사용합니다.
