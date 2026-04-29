# 최종 구현 보고서

## 2026-04-29 작업 순서 복구 감사 업데이트

이번 작업에서는 구조 정리와 문서 이중화의 순서가 뒤바뀌면서 생길 수 있는 문서/코드 불일치를 감사하고, 검증 스크립트와 보고서를 보강했습니다.

### 복구 변경 요약

- `docs/ko/reports/ORDER_RECOVERY_AUDIT.md`와 `docs/en/reports/ORDER_RECOVERY_AUDIT.md`를 추가했습니다.
- `scripts/maintenance/check_docs_i18n.py`가 `AGENTS.md`, root generated report 잔존 여부, `docs/ko`와 `docs/en` 상대경로 쌍, optional legacy string scan을 검사하도록 보강했습니다.
- `scripts/maintenance/smoke_test_api.py`가 dependency unavailable로 인한 503을 expected 503으로 분류할 수 있도록 보강했습니다.
- `DOCS_AUDIT.md`에 `ORDER_RECOVERY_AUDIT.md` 문서 쌍을 추가했습니다.
- 모델 artifact와 metadata가 `artifacts/models`, `artifacts/metadata`에 보존되어 있고, config/docs/API 설명과 일치함을 확인했습니다.
- `market_ai`가 `backend`를 import하지 않는 것을 확인했습니다.

### 최신 검증 결과

- `python scripts/maintenance/check_docs_i18n.py`: 실패. 현재 shell에 `python` 명령이 없습니다.
- `python3 scripts/maintenance/check_docs_i18n.py`: 통과. required root docs 5개, root pair 2개, `docs/ko` 15개, `docs/en` 15개가 대응됩니다.
- `.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy`: 통과. historical report reference 22건만 허용됐습니다.
- `python -m compileall backend market_ai scripts`: 실패. 현재 shell에 `python` 명령이 없습니다.
- `python3 -m compileall backend market_ai scripts`: 통과.
- `.venv/bin/python -m compileall backend market_ai scripts`: 통과.
- `python scripts/maintenance/smoke_test_api.py`: 실패. 현재 shell에 `python` 명령이 없습니다.
- `python3 scripts/maintenance/smoke_test_api.py`: 실패. system `python3`에 `numpy` module이 없습니다.
- `.venv/bin/python scripts/maintenance/smoke_test_api.py`: 통과. 필수 endpoint가 모두 200을 반환했습니다.
- `.venv/bin/python scripts/maintenance/audit_unused_files.py`: 통과. Markdown audit table 출력.
- `pytest`: 실패. 현재 shell PATH에 `pytest` 명령이 없습니다.
- `python3 -m pytest`: 실패. system `python3`에 `pytest` module이 없습니다.
- `.venv/bin/python -m pytest`: 통과, 39 tests.
- `frontend/`에서 `npm test`: 실패. 현재 shell에 `npm` 명령이 없습니다.
- `frontend/`에서 `npm run build`: 실패. 현재 shell에 `npm` 명령이 없습니다.
- `frontend/`에서 `npm run lint`: 실패. 현재 shell에 `npm` 명령이 없습니다.
- `ruff check .`: 실패. 현재 shell에 `ruff` 명령이 없습니다.
- `mypy`: 실패. 현재 shell에 `mypy` 명령이 없습니다.

## 2026-04-29 문서 정리 업데이트

이번 작업에서는 기능 코드를 크게 옮기지 않고 Markdown 문서를 한국어 원본 + 영어 mirror 정책에 맞춰 정리했습니다.

### 문서 변경 요약

- `README.md`를 한국어 메인 문서로 확장하고 `README.en.md`를 같은 구조의 영어 mirror로 맞췄습니다.
- `AGENTS.md`에 Codex가 읽을 한국어/영어 프로젝트 지침을 함께 작성했습니다.
- `docs/ko`와 `docs/en`의 핵심 문서 구조를 동일하게 유지했습니다.
- `docs/ko/reports/DOCS_AUDIT.md`와 `docs/en/reports/DOCS_AUDIT.md`를 추가했습니다.
- `scripts/maintenance/check_docs_i18n.py`를 고정 목록 검사에서 root 필수 문서, root report clutter, `docs/ko`와 `docs/en` 전체 상대경로 비교 방식으로 개선했습니다.
- `_archive` 아래 과거 Markdown은 active 문서가 아니라 obsolete/duplicate 문서로 audit에만 기록했습니다.

### 문서 검증 결과

- `python scripts/maintenance/check_docs_i18n.py`: 실패. 현재 shell에 `python` 명령이 없습니다.
- `python3 scripts/maintenance/check_docs_i18n.py`: 통과. required root docs 5개, root pair 2개, `docs/ko` 15개, `docs/en` 15개가 대응됩니다.
- `.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy`: 통과. historical report reference만 허용됩니다.
- `python -m compileall backend market_ai scripts`: 실패. 현재 shell에 `python` 명령이 없습니다.
- `python3 -m compileall backend market_ai scripts`: 통과.
- `.venv/bin/python -m compileall backend market_ai scripts`: 통과.
- `.venv/bin/python scripts/maintenance/smoke_test_api.py`: 통과. 필수 endpoint가 모두 200 반환.
- `pytest`: 실패. 현재 shell에 `pytest` 명령이 없습니다.
- `python3 -m pytest`: 실패. 현재 system `python3` 환경에 `pytest` module이 없습니다.
- `.venv/bin/python -m pytest`: 통과, 39 tests.

## 변경 요약

저장소를 Universal Market Forecasting Platform 구조로 재배치했습니다. FastAPI 코드는 `backend`, 시장 AI 로직은 `market_ai`, frontend asset은 `frontend`, CLI entrypoint는 `scripts`, model weight는 `artifacts/models`, metadata JSON은 `artifacts/metadata`, 생성 output은 `outputs`에 둡니다.

## 이동한 파일 목록

- `oil-tv-dashboard/app/main.py` -> `backend/app/main.py` 및 `backend/app/api/routes/*`.
- `oil-tv-dashboard/app/config.py` -> `market_ai/config.py`, `backend/app/core/config.py`.
- `oil-tv-dashboard/app/services/*` -> `market_ai/forecasting`, `market_ai/data`, `market_ai/modeling`.
- `oil-tv-dashboard/app/features/*` -> `market_ai/features/*`.
- `oil-tv-dashboard/app/forecasters/*`, `app/regime/*` -> `market_ai/modeling/*`.
- `oil-tv-dashboard/app/llm/*` -> `market_ai/llm`, `market_ai/schemas/llm_context.py`.
- `oil-tv-dashboard/app/models/*.npz` -> `artifacts/models/*.npz`.
- `oil-tv-dashboard/app/models/*.json` -> `artifacts/metadata/*.json`.
- `oil-tv-dashboard/train_pretrained_models.py` -> `scripts/train/train_pretrained_models.py`.
- `oil-tv-dashboard/backtest_forecasters.py` -> `market_ai/backtesting/runner.py`, `scripts/backtest/run_backtest.py` wrapper.
- `oil-tv-dashboard/app/static/*`, `app/templates/index.html` -> `frontend/`.
- Test는 `tests/unit`, `tests/integration`으로 이동했습니다.
- Report는 `docs/en/reports`, `docs/ko/reports`로 이동했습니다.

## 삭제한 파일 목록

없음. 파괴적 삭제는 수행하지 않았습니다.

## Archive한 파일 목록

- `oil-price-baseline/` -> `_archive/legacy_20260429/oil-price-baseline`.
- 남은 `oil-tv-dashboard/` shell/cache remnants -> `_archive/legacy_20260429/oil-tv-dashboard-remnants`.
- Root `.DS_Store` -> `_archive/legacy_20260429/root.DS_Store`.

## 유지한 Legacy Compatibility

- `app/main.py`는 `backend.app.main:app`을 가리키는 얇은 wrapper로 유지했습니다.
- `GET /api/chart`는 기존 chart payload key를 유지합니다.

## 새 디렉터리 구조

활성 top-level directory는 `backend`, `frontend`, `market_ai`, `scripts`, `configs`, `docs`, `data`, `artifacts`, `outputs`, `notebooks`, `tests`, `_archive`입니다.

## 실행한 테스트 명령어

- `python -m compileall backend market_ai scripts`: local shell에 `python`이 없어 실행 불가.
- `python3 -m compileall backend market_ai scripts`: 통과.
- `.venv/bin/python -m compileall backend market_ai scripts`: 통과.
- `.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy`: 통과.
- `.venv/bin/python scripts/maintenance/audit_unused_files.py`: 통과, Markdown audit table 출력.
- `.venv/bin/python scripts/maintenance/smoke_test_api.py`: 통과. 필수 endpoint가 모두 200 반환.
- `.venv/bin/python -m pytest`: 통과, 39 tests.
- `frontend/`에서 `npm test`: 실패. local shell에 `npm`이 없습니다.
- `frontend/`에서 `npm run build`: 실패. local shell에 `npm`이 없습니다.
- `ruff check .`: 실패. local shell에 `ruff`가 없습니다.
- `mypy`: 실패. local shell에 `mypy`가 없습니다.

## 테스트 결과

Compile, docs parity, legacy string scan, audit script, API smoke, pytest는 `.venv/bin/python` 기준으로 통과했습니다. Frontend test/build와 lint command는 local tool이 없어 실패했습니다.

## 사용자가 다음에 확인해야 할 사항

- `_archive/legacy_20260429`를 장기 보관할지, 검토 후 제거할지 결정해야 합니다.
- `outputs/backtests` 생성 output을 로컬에 유지할지, 필요 시 재생성할지 결정해야 합니다.
