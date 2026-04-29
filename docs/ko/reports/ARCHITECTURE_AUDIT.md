# 아키텍처 감사

기존 저장소는 FastAPI endpoint, forecasting 로직, model artifact, report, output, 과거 baseline 실험 프로젝트가 oil 전용 디렉터리에 섞여 있었습니다.

주요 발견:

- HTTP 코드와 시장 AI 로직이 구형 `app` package에 결합되어 있었습니다.
- `.npz` 모델 artifact와 JSON metadata가 application code 아래에 있었습니다.
- Backtest와 train script가 프로젝트 root에 있었습니다.
- 생성 report가 root에 있었습니다.
- `oil-price-baseline` 과거 실험 프로젝트가 dashboard와 나란히 있었습니다.

새 구조는 `backend`, `market_ai`, `frontend`, `scripts`, `artifacts`, `outputs`, 양언어 `docs`로 책임을 분리합니다.
