# Oil Price Forecast Baseline

졸업작품 시작용 AI 유가예측 베이스라인 코드입니다.

## 1) 설치

```bash
cd /Users/ahnyubin/Documents/Codex/2026-04-17-ai/oil-price-baseline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 실행

### A. 데모 데이터로 바로 실행 (인터넷 불필요)

```bash
python train_oil_baseline.py --source demo --model rf --horizon 1
```

### B. 야후 파이낸스로 실제 유가 데이터 사용

```bash
python train_oil_baseline.py --source yfinance --symbol CL=F --start-date 2015-01-01 --model rf --horizon 1
```

### C. CSV 데이터 사용

CSV는 최소한 `Date`, `Close` 컬럼이 있어야 합니다.

```bash
python train_oil_baseline.py --source csv --csv-path ./data/oil.csv --date-col Date --target-col Close --model rf --horizon 1
```

## 3) 결과물

실행 후 `outputs/` 폴더에 아래 파일이 생성됩니다.

- `metrics.json`: MAE, RMSE, MAPE
- `predictions.csv`: 날짜별 실제값/예측값
- `forecast_plot.png`: 예측 그래프
- `model.joblib`: 학습된 모델과 feature 목록

## 4) 모델 옵션

- `rf`: RandomForestRegressor (기본)
- `gbr`: GradientBoostingRegressor
- `xgb`: XGBoostRegressor (xgboost 설치 필요)

## 5) 확장 아이디어

- 입력 변수 추가: 환율, 금리, 원유재고, 뉴스 감성 점수
- 예측 구간 확대: 1일/3일/7일 horizon 비교
- 대시보드 연동: Streamlit으로 시각화
