# 구현 계획

1. `/api/chart`, `/api/forecast`, `/api/models`, `/api/data-status`, `/api/health` compatibility를 유지합니다.
2. FastAPI 코드는 `backend/app`, 시장 AI 로직은 `market_ai`로 이동합니다.
3. Model artifact는 `artifacts/models`, metadata JSON은 `artifacts/metadata`로 이동합니다.
4. 사람이 실행하는 script는 `scripts`로 이동합니다.
5. 문서 쌍 검사, unused-file audit, API smoke test maintenance script를 추가합니다.
6. 판단이 불확실한 legacy content는 삭제하지 않고 archive합니다.
7. compile, smoke API, docs parity, audit script, pytest, frontend test, 가능한 lint로 검증합니다.
