# 프론트엔드

Frontend는 TradingView Lightweight Charts 스타일 overlay를 제공하는 dashboard UI입니다.

## 위치

- `frontend/index.html`: dashboard HTML shell
- `frontend/src/main.js`: 현재 chart interaction 구현
- `frontend/src/dashboard.css`: dashboard style
- `frontend/src/api`: API client stub

## API 사용

UI는 `/api/forecast`를 먼저 시도하고, 필요하면 `/api/chart` compatibility payload로 fallback합니다. `/api/chart`는 기존 overlay와 호환되어야 합니다.

## 향후 구조

UI가 커지면 `frontend/src/components/chart`, `components/controls`, `components/badges`, `components/panels`로 component를 분리합니다.
