# AI Agent Rules / AI 에이전트 규칙

이 저장소는 다시 유가 예측 전용 dashboard와 모델로 수렴 중입니다. 모든 변경은 아래 한국어 원칙과 영어 미러를 함께 따릅니다.

This repository is converging back to an oil-price-only forecasting dashboard and model. Every change must follow both the Korean policy and the English mirror below.

## 한국어 정책

### 문서

- 문서는 한국어와 영어로 모두 유지해야 합니다.
- 한국어 문서는 프로젝트 소유자가 읽는 기본 문서입니다.
- 영어 문서는 협업과 오픈소스 공개를 위한 mirror입니다.
- root에 생성 report를 흩어 놓지 마십시오.
- 생성 report는 `docs/ko/reports`와 `docs/en/reports`로 이동하십시오.
- `README.md`는 한국어 메인이고 `README.en.md`는 영어 mirror입니다.
- `docs/ko`와 `docs/en`은 동일한 상대경로 구조를 유지해야 합니다.

### 호환성

- 명시적으로 제거하기 전까지 `/api/chart` backward compatibility를 보존하십시오.
- 새 API field는 additive로 추가하십시오.

### 데이터 정책

- production에서 mock data를 조용히 사용하지 마십시오.
- Mock/fallback data는 `APP_ENV=development` 또는 `ALLOW_MOCK_DATA=true`일 때만 허용됩니다.
- 데이터 품질은 `data_status`, warning, 명시적 error로 드러내십시오.

### 예측 정책

- LLM은 context/event encoder로만 사용하고 숫자 가격 예측기로 사용하지 마십시오.
- Forecast target은 volatility-scaled cumulative log return distribution 구조를 유지해야 합니다.
- 예측 가격은 `price_t+h = current_price * exp(predicted_cumulative_log_return_h)` 방식으로 복원합니다.
- Coverage가 실제로 측정되기 전에는 probabilistic band를 검증된 confidence interval이라고 부르지 마십시오.

### 테스트와 경로

- 동작이 바뀌면 test를 추가하거나 업데이트하십시오.
- 절대 local path를 hardcode하지 마십시오.
- Model artifact와 metadata는 source code와 분리해서 보관하십시오.
- `.npz` model artifact는 `artifacts/models`, metadata JSON은 `artifacts/metadata`에 둡니다.

## English Policy

### Documentation

- Documentation must be maintained in Korean and English.
- Korean docs are the primary human-readable docs for the project owner.
- English docs are the collaboration and open-source mirror.
- Do not leave root-level generated reports cluttering the repository.
- Move generated reports into `docs/ko/reports` and `docs/en/reports`.
- `README.md` is the Korean primary README and `README.en.md` is the English mirror.
- `docs/ko` and `docs/en` must keep identical relative paths.

### Compatibility

- Preserve backward compatibility for `/api/chart` unless explicitly removed.
- Add new API fields additively.

### Data Policy

- Do not silently use mock data in production.
- Mock/fallback data is allowed only when `APP_ENV=development` or `ALLOW_MOCK_DATA=true`.
- Surface data quality through `data_status`, warnings, and explicit errors.

### Forecasting Policy

- LLM must be used as context/event encoder, not as the numeric price forecaster.
- Forecast target should remain volatility-scaled cumulative log return distribution.
- Reconstruct forecast prices with `price_t+h = current_price * exp(predicted_cumulative_log_return_h)`.
- Do not call probabilistic bands validated confidence intervals until coverage is actually measured.

### Tests and Paths

- Add or update tests when behavior changes.
- Do not hardcode absolute local paths.
- Keep model artifacts and metadata separate from source code.
- `.npz` model artifacts belong in `artifacts/models`, and metadata JSON belongs in `artifacts/metadata`.
