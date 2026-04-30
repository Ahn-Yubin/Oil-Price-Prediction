# Deep Artifact 운영화 보고서

작성일: 2026-04-30

## 1. 변경 요약

- Deep artifact availability를 파일명, interval, horizon, metadata status 기준으로 판단하도록 공통 정책을 추가했습니다.
- `/api/models`가 `expected_artifact_file`, `expected_metadata_file`, `training_command`, `status`를 반환하도록 확장했습니다.
- `/api/forecast` 기본 요청은 artifact가 없는 deep model을 시도하지 않습니다.
- 사용자가 deep model을 명시 요청했는데 artifact가 없으면 200 fallback과 actionable warning object를 반환합니다.
- `warnings: list[str]`는 유지하고 `warning_objects`를 optional additive field로 추가했습니다.
- frontend는 `severity=info`를 노란 warning banner가 아니라 작은 info badge로 표시합니다.
- training CLI의 `--events-path`를 실제 `FileEventProvider`로 연결했습니다.
- production training에서 yfinance 실패 시 synthetic fallback을 조용히 사용하지 않도록 바꿨습니다.
- quick-test artifact는 `artifacts/smoke`로 분리하고 `status=smoke_only`로 기록합니다.
- `PROJECT_STATUS.md` 한국어/영어 문서의 outdated model/LLM/deep artifact 설명을 최신화했습니다.

## 2. Artifact Availability 정책

Production deep artifact는 다음 조건을 모두 만족해야 available입니다.

- expected artifact file이 존재합니다.
- expected metadata file이 존재하거나 artifact metadata를 읽을 수 있습니다.
- metadata `status`가 `available`입니다.
- metadata `synthetic_used`가 true이면 production available로 보지 않습니다.

`smoke_only`, `synthetic_only`, `failed`, `metadata_only`는 default `/api/forecast` candidate에서 제외됩니다.

## 3. Training CLI 변경

수정 파일:

- `scripts/train/train_deep_fusion_models.py`
- `market_ai/data/deep_dataset.py`는 기존 `event_provider` 입력을 그대로 사용합니다.
- `market_ai/data/event_providers.py`는 변경 없이 comma-separated path를 CLI에서 처리합니다.

주요 변경:

- `--events-path a.csv,b.json` 지원
- `NEWS_EVENTS_PATH`, `ECONOMIC_EVENTS_PATH`, `MARKET_EVENTS_PATH` env fallback 유지
- `--allow-synthetic-fallback` 추가
- `--quick-test`는 synthetic smoke training 허용
- `--synthetic`은 명시적 synthetic dataset 사용
- quick-test output은 `artifacts/smoke/models`, `artifacts/smoke/metadata`에 저장
- metadata 필수 운영 필드 보강

## 4. `--events-path` 수정 여부

수정 완료. CLI가 `FileEventProvider(paths=[...])`를 만들고 `build_deep_dataset_from_frame(..., event_provider=provider)`에 전달합니다.

테스트 `tests/unit/test_train_deep_fusion_cli_policy.py`에서 event CSV가 실제 event vector에 반영되는지 확인했습니다.

## 5. Synthetic Fallback 정책

현재 정책:

- 기본 production training: yfinance sample이 없으면 실패
- `--allow-synthetic-fallback`: yfinance 실패 뒤 synthetic fallback 허용
- `--synthetic`: 처음부터 synthetic dataset 사용
- `--quick-test`: synthetic smoke dataset 허용
- synthetic/smoke artifact는 production available로 노출하지 않음

## 6. 생성된 Artifact/Metadata

Production artifact:

- `artifacts/models/deep_lstm_tcn_fusion_1d_h45.pt`
- `artifacts/metadata/deep_lstm_tcn_fusion_1d_h45.json`
- `artifacts/models/llm_context_seq_moe_1d_h45.pt`
- `artifacts/metadata/llm_context_seq_moe_1d_h45.json`

Production metadata 요약:

- `interval=1d`, `horizon=45`, `lookback=128`
- `data_source=yfinance`
- `synthetic_used=false`
- `status=available`
- `n_train=3584`, `n_val=768`, `n_test=768`
- `training_cutoff=2026-02-24T00:00:00+00:00`

Smoke artifact:

- `artifacts/smoke/models/deep_lstm_tcn_fusion_1d_h8.pt`
- `artifacts/smoke/metadata/deep_lstm_tcn_fusion_1d_h8.json`
- `artifacts/smoke/models/llm_context_seq_moe_1d_h8.pt`
- `artifacts/smoke/metadata/llm_context_seq_moe_1d_h8.json`

기존 `artifacts/metadata/*_1d_h8.json`도 `status=smoke_only`, `synthetic_used=true`로 수정했습니다.

`.pt` artifact는 `.gitignore` 정책에 따라 commit 대상이 아닙니다. Metadata JSON은 artifact 운영 상태를 설명하므로 commit 후보입니다.

## 7. `/api/models` 결과 요약

`/api/models`에서 deep model 상태:

- `deep_lstm_tcn_fusion`: `status=available`, `expected_artifact_file=deep_lstm_tcn_fusion_1d_h45.pt`
- `llm_context_seq_moe`: `status=available`, `expected_artifact_file=llm_context_seq_moe_1d_h45.pt`

각 deep model은 `training_command`를 함께 반환합니다.

## 8. `/api/forecast` Warning 변화

기본 요청:

- selected models에 deep model이 포함됩니다. h45 artifact 생성 전에는 missing deep model을 조용히 제외하고 `artifact_status`에만 기록했습니다.
- h45 artifact 생성 후 primary model은 `deep_lstm_tcn_fusion`입니다.
- missing deep artifact warning은 기본 dashboard load에 표시되지 않습니다.

명시 deep 요청:

- artifact가 없으면 200 fallback과 `warning_objects[].code=deep_artifact_unavailable`을 반환합니다.
- warning action에 학습 명령이 포함됩니다.

Quantile warning:

- 문자열 warning은 유지합니다.
- `warning_objects`에서는 `severity=info`, `code=quantile_bands_uncalibrated`입니다.

## 9. `PROJECT_STATUS.md` 수정 요약

한국어/영어 mirror 모두 다음을 반영했습니다.

- active model list를 `motif`, `pattern_mlp`, `deep_lstm_tcn_fusion`, `llm_context_seq_moe`, baseline 계열로 정리
- `cycle`, `lstm`, `tcn`, `ensemble`을 removed/deprecated로만 설명
- LLM external call 조건을 실제 코드와 일치하도록 수정
- deep artifact가 code complete이지만 artifact availability에 의존한다고 명시
- quick-test h8 artifact는 dashboard default h45에 쓰이지 않는다고 명시
- warning severity 정책과 다음 작업 순서를 업데이트

## 10. Backtest Smoke 결과

실행:

```bash
.venv/bin/python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 5 --models random_walk,drift,motif,pattern_mlp,deep_lstm_tcn_fusion,llm_context_seq_moe --no-plots
```

결과:

- 전체 실행 성공
- `outputs/backtests/latest_model_availability.csv` 생성
- `/api/backtests?symbol=CL=F&interval=1d`가 `model_availability` 반환
- deep model 모두 `available`, `origins_ok=5`, `origins_error=0`

## 11. 테스트 실행 결과

통과:

- `.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy`
- `.venv/bin/python -m compileall backend market_ai scripts`
- `.venv/bin/python scripts/maintenance/smoke_test_api.py`
- `.venv/bin/python -m pytest` (`80 passed`)
- `node --check frontend/src/main.js`
- quick smoke training 2건
- full `1d/h45` training 2건
- deep backtest smoke

## 12. 실패/스킵 사유

- `python` executable이 없어 `.venv/bin/python`으로 실행했습니다.
- `frontend/node_modules`가 없어 `npm run build`는 스킵했습니다.
- `ruff`와 `mypy` 실행 파일 또는 설정이 없어 스킵했습니다.

## 13. 다음 작업

1. 생성된 h45 artifact를 더 긴 epoch와 넓은 validation으로 재학습
2. deep leaderboard를 여러 symbol/interval로 확장
3. quantile coverage calibration을 rolling backtest 결과와 연결
4. sample event가 아닌 real event ingestion pipeline 구축
5. cross-asset feature matrix를 실제 related asset 값으로 확장
6. frontend model diagnostics panel 추가
7. provider cache/storage 계층 도입
