# 작업 순서 복구 감사

이 문서는 프로젝트 구조 정리와 Markdown 이중화 작업 순서가 뒤바뀐 뒤, 현재 저장소가 실제 코드 구조와 문서 정책을 일관되게 반영하는지 재검증한 결과입니다.

## 1. 실제로 발생한 문제

- 문서 검증 스크립트가 처음에는 `AGENTS.md`와 root generated report 잔존 여부를 검사하지 못했습니다.
- API smoke test가 dependency unavailable 상황을 `expected 503`으로 구분하지 못하고 단순 실패로만 처리할 수 있었습니다.
- `FINAL_IMPLEMENTATION_REPORT.md`가 문서 정리 결과와 구조 정리 결과를 모두 담고 있었지만, 작업 순서 오류 복구 감사 결과는 별도 문서로 분리되어 있지 않았습니다.
- 구형 경로 문자열은 일반 사용 문서에는 남아 있지 않았고, cleanup/final/audit report 안의 과거 이동 내역으로만 남아 있었습니다. 이 보고서에서는 해당 사용을 의도적인 historical reference로 분류했습니다.

## 2. 발생하지 않은 문제

- `backend.app.main:app` entrypoint는 존재하며 `app/main.py` compatibility wrapper도 유지되어 있습니다.
- `market_ai` 패키지가 `backend`를 import하는 역방향 의존성은 발견되지 않았습니다.
- `.npz` 모델 artifact와 metadata JSON은 `artifacts/models`, `artifacts/metadata`에 보존되어 있습니다.
- active README/docs/AGENTS/CHANGELOG에서 절대 로컬 경로는 발견되지 않았습니다.
- root 디렉터리에 generated report Markdown은 남아 있지 않습니다.
- `GET /api/chart`, `GET /api/forecast`, `GET /api/models`, `GET /api/data-status`, `GET /api/health`는 smoke test에서 모두 200을 반환했습니다.

## 3. 수정한 파일 목록

- `scripts/maintenance/check_docs_i18n.py`
- `scripts/maintenance/smoke_test_api.py`
- `docs/ko/reports/ORDER_RECOVERY_AUDIT.md`
- `docs/en/reports/ORDER_RECOVERY_AUDIT.md`
- `docs/ko/reports/DOCS_AUDIT.md`
- `docs/en/reports/DOCS_AUDIT.md`
- `docs/ko/reports/FINAL_IMPLEMENTATION_REPORT.md`
- `docs/en/reports/FINAL_IMPLEMENTATION_REPORT.md`

## 4. 문서 경로 수정 내역

- active 문서의 실행 entrypoint는 `uvicorn backend.app.main:app --reload --port 8000` 기준으로 맞췄습니다.
- training command는 `python scripts/train/train_pretrained_models.py --interval 1d` 기준으로 정리했습니다.
- backtest command는 `python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 5 --models random_walk,drift,flat --no-plots` 기준으로 정리했습니다.
- 모델 artifact 문서는 `artifacts/models`, metadata 문서는 `artifacts/metadata` 기준으로 맞췄습니다.
- 과거 경로 문자열은 `docs/*/reports`의 cleanup history 안에서만 의도적으로 허용했습니다.

## 5. 코드 import 수정 내역

- application import 경로는 이번 복구 작업에서 변경하지 않았습니다.
- `market_ai` -> `backend` 역방향 import는 발견되지 않았습니다.
- `app/main.py`는 새 entrypoint를 가리키는 얇은 compatibility wrapper로 유지합니다.
- 이번 코드 변경은 유지보수 스크립트 보강에 한정했습니다.

## 6. artifact 경로 확인 결과

- `.npz` 파일은 `artifacts/models`에 있습니다.
- metadata JSON은 `artifacts/metadata`에 있습니다.
- `market_ai/config.py`의 기본값은 `artifacts/models`, `artifacts/metadata`입니다.
- `market_ai/modeling/registry.py`는 `settings.model_dir`와 `settings.metadata_dir`를 사용합니다.
- 문서와 `.env.example`, `configs/default.yaml`도 동일한 위치를 설명합니다.

## 7. API compatibility 확인 결과

`scripts/maintenance/smoke_test_api.py`는 FastAPI `TestClient`로 다음 endpoint를 확인합니다.

- `GET /api/health`
- `GET /api/models`
- `GET /api/data-status?symbol=CL=F&interval=1d`
- `GET /api/forecast?symbol=CL=F&interval=1d`
- `GET /api/chart?symbol=CL=F&interval=1d`

현재 실행 결과는 모두 200입니다. 외부 데이터나 artifact unavailable로 인한 503은 스크립트에서 expected 503으로 별도 분류하도록 보강했습니다.

## 8. 테스트 실행 결과

- `python scripts/maintenance/check_docs_i18n.py`: 실패. 현재 shell에 `python` 명령이 없습니다.
- `python3 scripts/maintenance/check_docs_i18n.py`: 통과.
- `.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy`: 통과. historical report reference만 허용됨.
- `python -m compileall backend market_ai scripts`: 실패. 현재 shell에 `python` 명령이 없습니다.
- `python3 -m compileall backend market_ai scripts`: 통과.
- `.venv/bin/python -m compileall backend market_ai scripts`: 통과.
- `python scripts/maintenance/smoke_test_api.py`: 실패. 현재 shell에 `python` 명령이 없습니다.
- `python3 scripts/maintenance/smoke_test_api.py`: 실패. system `python3`에 `numpy` module이 없습니다.
- `.venv/bin/python scripts/maintenance/smoke_test_api.py`: 통과. 필수 endpoint 모두 200.
- `.venv/bin/python scripts/maintenance/audit_unused_files.py`: 통과. Markdown audit table 출력.
- `pytest`: 실패. 현재 shell PATH에 `pytest` 명령이 없습니다.
- `python3 -m pytest`: 실패. system `python3`에 `pytest` module이 없습니다.
- `.venv/bin/python -m pytest`: 통과, 39 tests.
- `npm test`: 실패. 현재 shell에 `npm` 명령이 없습니다.
- `npm run build`: 실패. 현재 shell에 `npm` 명령이 없습니다.
- `ruff check .`: 실패. 현재 shell에 `ruff` 명령이 없습니다.
- `mypy`: 실패. 현재 shell에 `mypy` 명령이 없습니다.

## 9. 남은 위험

- `_archive/legacy_20260429`에는 과거 프로젝트와 cache remnants가 보존되어 있으며, active scan 대상은 아니지만 저장소 크기와 혼동 위험이 있습니다.
- root working tree는 이전 migration 결과가 아직 commit되지 않은 큰 변경 상태입니다.
- frontend build/test와 lint는 local Node/Ruff/Mypy 도구가 없어 검증하지 못했습니다.
- `python` 명령이 없는 환경에서는 문서의 예시 명령을 `python3` 또는 `.venv/bin/python`으로 실행해야 합니다.

## 10. 사용자가 직접 확인해야 할 항목

- `_archive/legacy_20260429`를 장기 보관할지, 별도 백업 후 제거할지 결정해야 합니다.
- frontend를 실제 배포할 경우 Node.js와 npm을 설치한 환경에서 `npm test`와 `npm run build`를 다시 확인해야 합니다.
- 운영 환경에서는 `ALLOW_MOCK_DATA`를 켜지 않고 실제 provider와 artifact 경로가 준비되어 있는지 확인해야 합니다.
- 현재 uncommitted migration diff를 검토한 뒤 branch/commit 단위로 고정하는 것이 좋습니다.
