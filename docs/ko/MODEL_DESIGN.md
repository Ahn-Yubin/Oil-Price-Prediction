# 모델 설계

숫자 예측은 시계열 모델과 baseline이 담당합니다. LLM은 숫자 가격 예측기가 아닙니다.

## Forecast Target

Forecast target은 volatility-scaled cumulative log return distribution 구조를 유지합니다. 예측 가격은 다음 방식으로 복원합니다.

```text
price_t+h = current_price * exp(predicted_cumulative_log_return_h)
```

이 구조는 asset price scale이 달라도 모델 출력을 비교하고 calibration하기 쉽게 만듭니다.

## Quantile

Quantile path는 monotonic해야 합니다. Coverage가 backtest로 검증되기 전에는 probabilistic band를 검증된 confidence interval이라고 부르지 않습니다.

## Artifact와 Metadata

`.npz` model artifact는 `artifacts/models`에 둡니다. Metadata JSON은 `artifacts/metadata`에 둡니다. Source code 아래에 model weight를 두지 않습니다.
