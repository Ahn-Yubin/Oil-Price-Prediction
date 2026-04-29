# 변경 이력

## 2026-04-29

- 문서 정책을 한국어 원본 + 영어 mirror로 정리했습니다.
- `README.md`를 한국어 메인으로 확장하고 `README.en.md`를 같은 구조의 영어 mirror로 맞췄습니다.
- `AGENTS.md`에 한국어/영어 프로젝트 지침을 함께 정리했습니다.
- `docs/ko/reports/DOCS_AUDIT.md`와 `docs/en/reports/DOCS_AUDIT.md`를 추가했습니다.
- `scripts/maintenance/check_docs_i18n.py`를 `docs/ko`와 `docs/en` 전체 상대경로 비교 방식으로 개선했습니다.
- 저장소를 `backend`, `frontend`, `market_ai`, `scripts`, `docs`, `data`, `artifacts`, `outputs`, `notebooks`, `tests` 구조로 재배치했습니다.
- FastAPI entrypoint를 `backend.app.main:app`으로 이동하고 `app.main:app`은 얇은 compatibility wrapper로 남겼습니다.
- `.npz` 모델 artifact는 `artifacts/models`, metadata JSON은 `artifacts/metadata`로 이동했습니다.
- 학습과 backtest CLI를 `scripts/train/train_pretrained_models.py`, `scripts/backtest/run_backtest.py`로 이동했습니다.
- 문서 쌍 검사, unused-file audit, API smoke test maintenance script를 추가했습니다.
