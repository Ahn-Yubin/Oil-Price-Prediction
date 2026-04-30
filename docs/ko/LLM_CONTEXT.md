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
- `LocalHTTPLLMEventEncoder`: Ollama/vLLM 호환 local HTTP endpoint를 live 옵션에서만 호출합니다.
- `OfflineFileLLMEventEncoder`: 사전 계산된 JSON/JSONL context cache를 읽습니다.
- API key가 없거나 호출이 실패하면 local/null fallback을 사용합니다.
- Prompt는 `market_ai/llm/prompts`에 있습니다.

## Event 입력

기본 event file schema는 `data/external/events/README.md`에 있습니다. `NEWS_EVENTS_PATH`, `ECONOMIC_EVENTS_PATH`, `MARKET_EVENTS_PATH`로 파일을 연결할 수 있습니다. Event timestamp는 forecast `as_of_time` 이하만 사용합니다.

## 운영 모드와 명령

지원 모드는 `none`, `local_rules`, `openai_compatible`, `local_http`, `offline_file`입니다.

환경 변수:

- `ENABLE_LLM_CONTEXT`
- `ENABLE_EXTERNAL_LLM_CALLS`
- `LLM_API_KEY`, `LLM_API_BASE`, `LLM_MODEL`
- `LOCAL_LLM_API_BASE`, `LOCAL_LLM_MODEL`
- `MARKET_EVENTS_PATH`, `NEWS_EVENTS_PATH`

검증 명령:

```bash
python scripts/llm/test_llm_context.py --mode local_rules
python scripts/llm/test_llm_context.py --mode openai_compatible --dry-run
python scripts/llm/test_llm_context.py --mode local_http --dry-run
python scripts/data/build_event_context.py --events-path data/external/events/sample_market_events.csv --mode local_rules
```

실제 외부/local 호출은 `--live`와 `ENABLE_EXTERNAL_LLM_CALLS=true`가 함께 있을 때만 수행합니다. API key와 secret은 출력하거나 커밋하지 않습니다.
