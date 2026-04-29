# 백테스트

Backtesting은 모델 출력의 point accuracy, quantile quality, horizon별 성능, regime별 성능을 확인하는 검증 계층입니다.

## 지원 모델

기본 비교 대상은 `random_walk`, `drift`, `seasonal_naive`, `volatility_scaled_naive`, `flat`, `motif`, `pattern_mlp`, `deep_lstm_tcn_fusion`, `llm_context_seq_moe`입니다.

`cycle`, `lstm`, `tcn`, `ensemble`은 removed/deprecated 모델이며 요청 시 명확한 error를 반환합니다.

## 실행

```bash
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 10 --models random_walk,drift,motif,pattern_mlp,deep_lstm_tcn_fusion,llm_context_seq_moe --include-regime-breakdown --no-plots
```

## 코드 위치

재사용 가능한 backtest 로직은 `market_ai/backtesting/runner.py`에 있습니다. `scripts/backtest/run_backtest.py`는 사람이 실행하는 CLI wrapper입니다.

## 산출물

Backtest output은 `outputs/backtests`에 기록합니다. `model_availability.csv`에는 deep artifact가 없어서 unavailable인 모델도 기록합니다. Plot은 `outputs/backtests/plots`에 둡니다.

## 원칙

Backtest는 미래 정보를 사용하지 않는 rolling/expanding origin 방식이어야 합니다. Deep model도 origin 시점까지의 과거 close와 event context만 사용합니다. Forecast band coverage와 pinball loss를 함께 기록해 probabilistic forecast 품질을 확인합니다.
