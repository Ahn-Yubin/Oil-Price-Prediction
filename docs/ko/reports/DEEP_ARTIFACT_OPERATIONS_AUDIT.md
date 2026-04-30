# Deep Artifact 운영 감사

작성일: 2026-04-30

## 요약

현재 저장소에는 dashboard 기본 요청이 사용하는 `1d/h45` deep artifact가 없습니다. 대신 quick-test 성격의 `1d/h8` `.pt` artifact가 `artifacts/models`에 존재하고, metadata status가 `available`로 기록되어 있습니다. 이 때문에 `/api/models`는 deep 모델을 available로 표시하지만, `/api/forecast` 기본 요청은 `1d/h45` artifact를 찾지 못해 dashboard 첫 화면에 artifact unavailable warning을 표시합니다.

## Artifact 현황

`artifacts/models`:

- `global_dl_1d_h45.npz`
- `global_dl_1h_h72.npz`
- `global_dl_30m_h120.npz`
- `global_dl_15m_h192.npz`
- `deep_lstm_tcn_fusion_1d_h8.pt`
- `llm_context_seq_moe_1d_h8.pt`

`artifacts/metadata`:

- `global_dl_1d_h45.json`
- `global_dl_1h_h72.json`
- `global_dl_30m_h120.json`
- `global_dl_15m_h192.json`
- `deep_lstm_tcn_fusion_1d_h8.json`
- `llm_context_seq_moe_1d_h8.json`

## API 동작

`/api/models` 현재 동작:

- `ModelRegistry.scan()`이 `.npz`와 `.pt`를 모두 스캔합니다.
- `user_facing_models`에서 deep 모델 status는 같은 `model_name` artifact가 하나라도 있으면 `available`로 표시합니다.
- 따라서 `deep_lstm_tcn_fusion_1d_h8.pt`, `llm_context_seq_moe_1d_h8.pt`만 있어도 dashboard 기본 horizon인 `1d/h45` artifact 존재 여부와 무관하게 available로 보입니다.

`/api/forecast` 기본 요청 현재 동작:

- models query가 없으면 `USER_FACING_MODELS` 전체를 기본 selection으로 사용합니다.
- `deep_lstm_tcn_fusion`, `llm_context_seq_moe`를 먼저 시도합니다.
- 기본 `1d` horizon은 45이므로 `deep_lstm_tcn_fusion_1d_h45.pt`, `llm_context_seq_moe_1d_h45.pt`를 찾습니다.
- 해당 파일이 없어 `artifact_status`는 `missing_or_unavailable`이 되고, 문자열 warning이 추가됩니다.

`/api/forecast?models=deep_lstm_tcn_fusion` 현재 동작:

- status code 200을 반환합니다.
- deep artifact missing warning을 반환하고, 내부적으로 `motif` 등 non-deep fallback으로 예측을 생성합니다.
- warning에는 필요한 학습 명령이 포함되어 있지 않습니다.

## Quick-Test Artifact

현재 deep `.pt` artifact는 모두 `1d/h8`입니다.

- `deep_lstm_tcn_fusion_1d_h8.json`: `horizon=8`, `lookback=32`, `epochs_ran=1`, `status=available`
- `llm_context_seq_moe_1d_h8.json`: `horizon=8`, `lookback=32`, `epochs_ran=1`, `status=available`

이 artifact는 dashboard 기본 `1d/h45`에는 사용되지 않습니다. metadata status가 `available`이므로 production artifact처럼 보이는 문제가 있습니다.

## Training CLI

`scripts/train/train_deep_fusion_models.py`에서 `--events-path` 인자는 정의되어 있지만 `FileEventProvider(paths=[...])`로 변환되어 `build_deep_dataset_from_frame(..., event_provider=provider)`에 전달되지 않습니다. 현재 dataset builder는 `config.event_context_enabled`가 true일 때 env 기반 `FileEventProvider.from_env()`만 사용합니다.

yfinance 데이터 수집이 모두 실패하면 현재 `build_dataset()`은 조용히 `build_synthetic_deep_dataset()`으로 fallback하고 `source=synthetic_fallback` artifact를 만들 수 있습니다. production training에서는 위험한 기본값입니다.

## 문서 정합성

`docs/ko/PROJECT_STATUS.md`와 `docs/en/PROJECT_STATUS.md`에는 다음 충돌이 있습니다.

- 앞부분은 `cycle`, `lstm`, `tcn`, `ensemble`을 removed/deprecated로 분류하지만, 뒤쪽 active model 표와 frontend 설명에는 여전히 comparison model처럼 설명합니다.
- OpenAI-compatible LLM adapter를 외부 call이 없는 placeholder로 설명하지만, 실제 `OpenAICompatibleLLMEventEncoder`에는 `ENABLE_EXTERNAL_LLM_CALLS=true`와 `LLM_API_KEY`가 있을 때 `urllib`로 chat completions를 호출하는 경로가 있습니다.
- cross-asset feature는 아직 missing indicator placeholder 중심인데, 문서가 완성된 feature matrix처럼 읽힐 수 있습니다.
- deep quick training artifact가 production 성능 artifact처럼 오해될 수 있습니다.

## Frontend Warning UX

frontend는 `/api/forecast`의 `warnings: list[str]`를 공백으로 이어 `status-banner`에 표시합니다. severity 구분이 없어 artifact missing, stale data, quantile uncalibrated message가 모두 동일한 노란 박스로 표시됩니다.

## 원인

1. deep artifact availability가 `model_name` 존재 여부만 보고 horizon/status를 보지 않습니다.
2. 기본 forecast selection이 artifact가 없는 deep 모델까지 항상 시도합니다.
3. quick-test h8 metadata가 `available`로 기록되어 production artifact와 구분되지 않습니다.
4. warning contract가 문자열뿐이라 frontend에서 severity별 표시를 할 수 없습니다.
5. training CLI가 explicit event path와 synthetic fallback 정책을 production-safe하게 강제하지 않습니다.
