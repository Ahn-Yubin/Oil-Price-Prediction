# Oil TradingView Dashboard

`oil-price-baseline` 예측 결과를 TradingView Lightweight Charts로 시각화하는 FastAPI 대시보드입니다.
현재는 심볼 입력 + 인터벌 선택 기반으로 `yfinance` 최신 OHLC를 조회해
TradingView Lightweight Charts 단일 차트 위에 캔들 + 예측선 + 95% 신뢰구간 밴드를 오버레이합니다.
예측은 **사전훈련된 전역 딥러닝 가중치 파일(`app/models/*.npz`)**만 사용하며,
런타임에서 새 학습은 수행하지 않습니다.

## Offline pretraining (필수 1회)

아래를 먼저 실행해서 가중치 파일을 생성하세요.

```bash
cd /Users/ahnyubin/Documents/Codex/2026-04-17-ai/oil-tv-dashboard
source .venv/bin/activate
python train_pretrained_models.py
```

특정 인터벌만 학습:

```bash
python train_pretrained_models.py --interval 1d
```

강제 재학습:

```bash
python train_pretrained_models.py --force
```

## Run

```bash
cd /Users/ahnyubin/Documents/Codex/2026-04-17-ai/oil-tv-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

브라우저에서 `http://127.0.0.1:8000` 접속

가중치 파일이 없으면 API는 `503`으로 실패하며,
`Pretrained model file not found ...` 메시지를 반환합니다.

## Data source

- 기본: `/Users/ahnyubin/Documents/Codex/2026-04-17-ai/oil-price-baseline/outputs/predictions.csv`
- 파일이 없거나 컬럼이 다르면 mock 데이터로 자동 대체

필수 컬럼:

- `date`
- `actual`
- `predicted`

## API

- `GET /api/chart`
  - query: `symbol` (예: `NYMEX:CL1!`, `ICEEUR:BRN1!`, `CL=F`)
  - query: `interval` (`1d`, `1h`, `30m`, `15m`)
  - `actual`: 실제 유가 시계열
  - `candles`: OHLC 캔들 데이터
  - `predicted`: AI 예측 시계열
  - `predicted_lower` / `predicted_upper`: 95% 신뢰구간
  - `metrics`: MAE/RMSE/MAPE
  - `updated_at`: API 응답 시각(UTC)

## Model

- `Global DL MLP (pretrained, multi-horizon)`
- 입력: 최근 로그수익률 window
- 출력: horizon step 로그수익률 벡터
- CI: 검증셋 누적 잔차 표준편차 기반 95% 밴드
