# LLM Context

LLM은 시장 이벤트와 설명을 다루는 보조 계층입니다. 숫자 가격 예측은 시계열 모델이 담당합니다.

## 허용되는 역할

- 뉴스와 이벤트를 구조화된 market context로 변환합니다.
- 주요 driver, uncertainty, risk factor를 요약합니다.
- Forecast response를 사람이 읽기 쉬운 설명으로 바꿉니다.

## 금지되는 역할

- LLM이 `p50`, quantile, target price 같은 숫자 forecast를 직접 생성하면 안 됩니다.
- LLM output이 시계열 모델의 numeric forecast를 덮어쓰면 안 됩니다.

## 위치

LLM 관련 로직은 `market_ai/llm`과 `market_ai/schemas/llm_context.py`에 둡니다. Prompt는 `market_ai/llm/prompts`에 둡니다.
