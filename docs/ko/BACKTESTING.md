# 백테스트

Backtesting은 모델 출력의 point accuracy, quantile quality, horizon별 성능, regime별 성능을 확인하는 검증 계층입니다.

## 지원 모델

기본 비교 대상은 단일 운영 모델 `oil_context_fusion`과 내부 benchmark인 `random_walk`, `drift`, `motif`, `pattern_mlp`입니다.

`cycle`, `lstm`, `tcn`, `ensemble`은 removed/deprecated 모델이며 요청 시 명확한 error를 반환합니다.

## 실행

```bash
.venv/bin/python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 10 --models oil_context_fusion,random_walk,drift,motif,pattern_mlp --include-regime-breakdown --no-plots
```

## 코드 위치

재사용 가능한 backtest 로직은 `market_ai/backtesting/runner.py`에 있습니다. `scripts/backtest/run_backtest.py`는 사람이 실행하는 CLI wrapper입니다.

## 산출물

Backtest output은 `outputs/backtests`에 기록합니다. `model_availability.csv`에는 deep artifact가 없어서 unavailable인 모델도 기록합니다. Plot은 `outputs/backtests/plots`에 둡니다.

Leaderboard batch:

```bash
.venv/bin/python scripts/evaluate/run_model_leaderboard.py --symbols CL=F,BZ=F,NG=F --interval 1d --max-origins 50
```

이 명령은 `outputs/backtests/leaderboards/{timestamp}` 아래에 `leaderboard.csv`, `horizon_metrics.csv`, `probabilistic_metrics.csv`, `regime_metrics.csv`, `model_availability.csv`, `summary.md`를 저장하고 `latest.json`을 갱신합니다. `summary.md`는 canonical 문서가 아니라 실행 산출물이므로 오래되면 삭제해도 됩니다. `/api/backtests`는 latest leaderboard가 있으면 이를 우선 반환합니다.

Quantile calibration:

```bash
.venv/bin/python scripts/evaluate/calibrate_quantiles.py --model motif --symbol CL=F --interval 1d
```

Calibration artifact는 `artifacts/calibration/{model}_{symbol}_{interval}.json`에 저장합니다. Artifact가 `calibration_status=calibrated`이면 `/api/forecast`가 band를 conformal adjustment로 넓히고, 없으면 기존 unvalidated warning을 유지합니다.

## 원칙

Backtest는 미래 정보를 사용하지 않는 rolling/expanding origin 방식이어야 합니다. Deep model도 origin 시점까지의 과거 close와 event context만 사용합니다. Forecast band coverage와 pinball loss를 함께 기록해 probabilistic forecast 품질을 확인합니다.
