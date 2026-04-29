# 로드맵

이 프로젝트는 유가 예측 use case에서 시작해 범용 시장 AI 플랫폼으로 확장합니다.

## 다음 단계

- 유가 선물에서 주식, ETF, FX, crypto, rates, index, commodity로 확장합니다.
- Provider abstraction을 강화하고 data quality report를 정례화합니다.
- Calibration, conformal interval, quantile coverage 평가를 확장합니다.
- LLM event ingestion을 production-grade로 만들되 숫자 forecast는 시계열 모델에만 맡깁니다.
- Frontend가 커지면 Vite component 구조로 점진적으로 분리합니다.

## 유지해야 할 원칙

- `/api/chart` compatibility를 명시적 제거 전까지 유지합니다.
- Artifact와 metadata는 source code와 분리합니다.
- 한국어 원본과 영어 mirror 문서를 항상 함께 갱신합니다.
