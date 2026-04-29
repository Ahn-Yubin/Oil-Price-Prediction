# 데이터 파이프라인

데이터 파이프라인은 market data provider, symbol normalization, timeframe normalization, data status reporting으로 구성됩니다.

## Provider

기본 provider는 yfinance입니다. Provider 구현은 `market_ai/data/providers`에 둡니다. 향후 CSV, vendor API, database provider를 같은 interface로 확장할 수 있어야 합니다.

## 데이터 품질

모든 API 응답은 가능한 경우 `DataStatus`를 통해 source, resolved symbol, interval, last bar, stale 여부, warning을 노출합니다.

## Mock Data 정책

Production에서는 mock data를 조용히 사용하지 않습니다. Mock/fallback data는 `APP_ENV=development` 또는 `ALLOW_MOCK_DATA=true`일 때만 허용됩니다.

## 저장 위치

- `data/raw`: 원본 수집 데이터
- `data/interim`: 중간 처리 데이터
- `data/processed`: 정제된 데이터
- `data/features`: feature matrix
- `data/external`: 외부 보조 데이터
