# 변경 이력

## 2026-05-10

- EIA bulk, CFTC ZIP/manual CSV, CME manual CSV 데이터 경로를 정리했습니다.
- 공개 뉴스와 LLM/event context 생성 흐름을 운영 문서에 반영했습니다.
- `/api/market-context`와 frontend context marker/news/scenario panel 문서를 추가했습니다.
- Google Gemma/Gemini OpenAI-compatible LLM 설정, `export` 확인법, `.env` 사용법을 문서화했습니다.
- 오래된 report성 Markdown을 canonical 문서로 흡수하고 삭제했습니다.
- `README`, `PROJECT_STATUS`, `DATA_PIPELINE`, `OPERATIONS`, `LLM_CONTEXT`, `API`, `FRONTEND`, `MODEL_DESIGN`, `ROADMAP`의 한국어/영어 mirror를 업데이트했습니다.

## 2026-04-29

- 문서 정책을 한국어 원본 + 영어 mirror로 정리했습니다.
- `README.md`를 한국어 메인으로 확장하고 `README.en.md`를 같은 구조의 영어 mirror로 맞췄습니다.
- `AGENTS.md`에 한국어/영어 프로젝트 지침을 함께 정리했습니다.
- `scripts/maintenance/check_docs_i18n.py`를 `docs/ko`와 `docs/en` 전체 상대경로 비교 방식으로 개선했습니다.
- 저장소를 `backend`, `frontend`, `market_ai`, `scripts`, `docs`, `data`, `artifacts`, `outputs`, `notebooks`, `tests` 구조로 재배치했습니다.
- FastAPI entrypoint를 `backend.app.main:app`으로 이동하고 `app.main:app`은 얇은 compatibility wrapper로 남겼습니다.
- `.npz` 모델 artifact는 `artifacts/models`, metadata JSON은 `artifacts/metadata`로 이동했습니다.
- 학습과 backtest CLI를 `scripts/train/train_pretrained_models.py`, `scripts/backtest/run_backtest.py`로 이동했습니다.
- 문서 쌍 검사, unused-file audit, API smoke test maintenance script를 추가했습니다.
