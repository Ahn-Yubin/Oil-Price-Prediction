# 모델 설계

이 문서는 일반 사용자가 “유가 예측 모델이 어떤 자료를 보고, 어떤 방식으로 판단하고, 최종 예측값을 어떻게 만드는지” 이해할 수 있도록 설명합니다. 자세한 코드 구현은 뒤쪽 기술 부록에만 남깁니다.

LLM은 가격을 직접 찍어내는 예측기가 아닙니다. 뉴스와 이벤트를 읽어 시장 분위기를 정리하고, 최종 화면에서는 애널리스트식 해설을 만드는 역할만 합니다. 숫자 예측은 `oil_context_fusion`이라는 단일 통합 유가 예측 모델이 담당합니다.

## 한 줄 요약

`oil_context_fusion`은 가격 흐름, 관련 에너지 시장, 금리/환율/주식시장, 원유 재고와 포지션, 뉴스 분위기를 함께 보고 여러 관점의 expert 판단을 합친 뒤 미래 유가 경로와 범위를 계산합니다.

## 전체 데이터 흐름

```mermaid
flowchart LR
    A["WTI 원유 가격데이터"] --> E["통합 유가 예측 모델"]
    B["브렌트유·천연가스·휘발유·난방유 등 관련시장"] --> E
    C["금리·환율·나스닥/주식시장·변동성 지표"] --> E
    D["EIA 재고·CFTC 포지션·뉴스/이벤트"] --> E
    E --> F["여러 expert의 판단"]
    F --> G["가중치 조정"]
    G --> H["기준 예측 경로"]
    G --> I["상단/하단 예측 범위"]
    H --> J["차트 표시와 AI 시황 해설"]
    I --> J
```

데이터는 항상 “그 시점에 이미 알 수 있었던 정보”만 사용합니다. 예를 들어 EIA 재고나 CFTC 포지션은 발표일 이후에만 학습 샘플에 들어갑니다. 이 원칙은 백테스트에서 미래 정보를 몰래 보는 문제를 막기 위해 중요합니다.

## 모델을 구성하는 expert들

통합 모델은 하나의 모델이지만 내부에는 일곱 가지 관점이 있습니다. 화면에는 하나의 예측선만 나오지만, 내부에서는 아래 expert들의 판단을 상황에 맞게 섞습니다.

| Expert | 쉬운 설명 | 주로 보는 것 |
| --- | --- | --- |
| 장기 흐름 expert | 과거 가격 흐름의 순서와 누적된 방향성을 기억합니다. 급등 후 조정, 긴 횡보 후 돌파 같은 긴 맥락을 봅니다. | 긴 가격 흐름 |
| 충격/단기 패턴 expert | 최근 며칠 또는 몇 시간 동안의 빠른 변화와 변동성 충격을 잘 잡습니다. | 단기 가격 변화, 급등락 |
| 중요 구간 집중 expert | 모든 과거 구간을 똑같이 보지 않고, 현재 판단에 중요한 시점을 더 강하게 봅니다. | 중요한 봉, 전환점 |
| 뉴스/거시 맥락 expert | 뉴스, 이벤트, 재고, 금리, 환율, 위험선호 분위기를 가격 흐름과 함께 해석합니다. | 뉴스데이터, 경제지표, 수급 자료 |
| 패턴 expert | 최근 흐름의 평균, 변동성, 추세, 고점/저점 위치를 요약해 “지금 차트 모양이 어떤 상태인지” 판단합니다. | 차트 모양, 추세, 변동성 |
| 모티프 expert | 최근 움직임과 비슷했던 과거 구간을 찾아 당시 이후의 흐름을 참고합니다. 단순 복사는 아니고, 비슷한 장면들의 평균적인 힌트를 가져옵니다. | 과거 유사 구간 |
| 이벤트 충격 expert | 지정학/공급 충격 신호가 강할 때 일반적인 p50 경로가 과도하게 평평해지는 문제를 줄이기 위한 전용 관점입니다. | 이벤트 컨텍스트, 패턴 요약 |

## 패턴 모델과 모티프 모델

패턴 모델은 차트의 현재 모양을 요약하는 역할입니다. 예를 들어 “고점권에서 밀리고 있는지”, “저점권에서 반등하는지”, “상승 추세가 잠시 쉬는지”, “박스권에 갇혀 있는지” 같은 정보를 숫자로 정리합니다. 이 expert는 모델이 최근 가격의 모양을 빠르게 읽도록 돕습니다.

모티프 모델은 “지금과 비슷했던 과거 장면”을 찾습니다. 최근 일정 구간의 움직임을 과거 여러 구간과 비교하고, 가장 비슷한 과거 구간들이 이후에 어떤 방향으로 움직였는지 참고합니다. 원유는 재고 발표, 지정학 뉴스, OPEC 발언 뒤에 비슷한 반복 패턴이 나타나는 경우가 있기 때문에 이 관점이 도움이 됩니다.

두 모델은 더 이상 사용자에게 따로 노출되는 별도 예측 모델이 아닙니다. 지금은 `oil_context_fusion` 내부 expert로 들어가 최종 예측을 보조합니다.

## 최종 예측값이 만들어지는 방식

1. 가격데이터와 관련시장데이터에서 추세, 변동성, 상대 강도, 최근 충격을 계산합니다.
2. 원유 재고, 포지션, 금리, 환율, 주식시장, 뉴스데이터를 시점에 맞게 붙입니다.
3. 일곱 expert가 각자 미래 경로에 대한 판단을 만듭니다.
4. 가중치 조정 장치가 현재 국면에 더 맞는 expert에 더 큰 비중을 줍니다.
5. 여러 expert의 결과를 합쳐 p05부터 p95까지의 분위수 경로를 만듭니다.
6. 기본 표시선은 모델이 학습한 p50 경로입니다. 화면에서 보이는 중앙선은 inference-time tail/hump 후처리로 꾸미지 않습니다.
7. 현재 가격에서 미래 가격으로 복원해 차트에 표시합니다.
8. AI 시황 해설은 이 결과와 뉴스/차트 맥락을 바탕으로 “왜 그런 방향으로 보는지” 설명합니다.

## 입력과 출력

아래 표의 이름은 코드 변수명이 아니라 사용자가 이해하기 쉬운 데이터 묶음 이름입니다. 크기는 실제 모델에 들어가는 텐서 크기입니다.

| 데이터 묶음 | 크기 | 의미 |
| --- | --- | --- |
| 가격데이터 | `[batch, 128, 23]` | 원유 자체 가격, 거래량, 변동성, 모멘텀, 추세, 고점/저점 위치 |
| 관련시장데이터 | `[batch, 128, 6]` | 브렌트유, 천연가스, 휘발유, 난방유, 달러, 금리, 주식시장 등에서 만든 보조 정보 |
| 뉴스/이벤트데이터 | `[batch, 27]` | LLM이 요약한 13개 context feature와 최근 원시 뉴스 풀에서 계산한 뉴스량, 선택 coverage, 상승/하락 압력, 에너지/지정학/거시/수급 압력, 소스 다양성 14개 feature |
| 현재상태데이터 | `[batch, 4]` | 현재 가격, 최근 변동성, 모델이 보는 과거 길이, 예측 길이 |
| 예측범위 | `[batch, 30, 7]` | 최대 30스텝까지의 하단/중앙/상단 경로. 화면에서는 30일 경로 전체와 1주/2주/한달 endpoint를 표시 |
| 상승가능성 | `[batch, 30]` | 각 미래 시점에서 상승 쪽으로 기울 가능성 |
| 예상변동성 | `[batch, 30]` | 각 미래 시점에서 흔들릴 수 있는 정도 |
| 모델신뢰도 | `[batch, 1]` | 현재 입력 상태에서 모델이 스스로 평가한 안정성 점수 |

## 예측 대상

모델은 미래 가격을 그대로 외우거나 직접 맞히지 않습니다. 먼저 “현재 가격 대비 얼마나 움직일지”를 학습한 뒤, 마지막에 다시 가격으로 바꿉니다.

```text
변동성으로 조정한 미래 움직임 = 미래 누적 로그수익률 / 최근 실현변동성
미래 가격 = 현재 가격 * exp(예측 누적 로그수익률)
```

이 구조를 쓰면 가격대가 바뀌어도 모델이 “움직임의 크기”를 비교적 안정적으로 배울 수 있습니다.

2026-06-05 이후 학습은 p50 자체가 너무 평평해지는 문제를 줄이기 위해 경로 형태를 명시적으로 봅니다. 손실 함수는 분위수 pinball loss 외에 step return, detrended shape, 경로 range, step volatility, curvature, step 방향성, shock/range 보조 head를 함께 최적화합니다. 예측 target은 여전히 volatility-scaled cumulative log return이며, 가격 복원 공식은 그대로 유지합니다.

## 예측 기간 표시 방식

모델 하나로 예측 기간을 마음대로 늘렸다 줄이는 것은 구조적으로 “일부는 가능하고, 일부는 위험합니다.”

현재 운영 UI는 1D h30 artifact 하나를 기준으로 30일 경로 전체를 표시합니다. 사용자가 예측 기간을 바꾸는 selector는 제거했고, 같은 h30 출력 위에 1주, 2주, 한달 endpoint를 점과 텍스트로 표시합니다. 이렇게 해야 한 화면에서 서로 다른 모델이나 horizon artifact를 비교하는 문제가 생기지 않습니다.

이 방식의 장점은 모든 구간 표시가 같은 모델 판단에서 나온다는 점입니다. 1주/2주/한달 label은 별도 예측선이 아니라 동일한 30일 p50 경로의 endpoint marker입니다.

30보다 더 긴 기간은 같은 모델을 반복 호출해 억지로 이어 붙일 수는 있지만 오차가 빠르게 누적됩니다. 그래서 60일, 90일 같은 장기 기간이 필요하면 h60, h90 artifact를 별도로 학습하는 편이 더 안전합니다.

## 현재 작동 범위와 확장 전략

운영 UI는 현재 1D/30일 고정 화면을 제공합니다. 1H는 API와 연구용 artifact 대상으로 남기고, 15M/30M은 현재 데이터 기간이 짧고 뉴스/수급 자료의 발표 시각을 분봉에 정확히 맞추는 품질이 아직 부족하다고 판단해 운영 UI에서 제외합니다.

| 주기 | 화면 선택 기간 | 모델 artifact | 상태 |
| --- | --- | --- | --- |
| 1D | 30일 + 1주/2주/한달 endpoint | h30 | 단일 운영 artifact 전체 경로 표시 |
| 1H | API/연구용 | h30 | 별도 검증 대상 |
| 30M | 제외 | 연구 대상 | 데이터 기간과 뉴스/수급 정렬 보강 필요 |
| 15M | 제외 | 연구 대상 | 노이즈가 크고 학습 기간이 짧아 별도 검증 필요 |

확장 순서는 다음이 합리적입니다.

1. 1D h30을 기준 운영 모델로 둡니다.
2. 1H h30은 별도 검증 후 필요할 때 시간 단위 예측으로 추가합니다.
3. 각 artifact마다 SSE, MSE, RMSE, MAE, R2, MAPE, sMAPE, 방향정확도, step 방향정확도, range ratio, turn error, shape score를 기록하고 1D와 1H를 따로 비교합니다.
4. 충분한 백테스트가 쌓이면 예측 범위 보정 artifact를 별도로 만듭니다.
5. 이후 30M/15M은 발표 시각 정렬과 데이터 기간 문제가 해결된 뒤 검토합니다.

## 2026-06-05 재학습 상태

현재 우선 안정화한 artifact는 CL=F 전용 `oil_context_fusion_1d_h30`입니다. 뉴스는 Google News RSS backfill과 공개 RSS를 합쳐 가격 데이터 범위와 맞췄고, 이벤트 컨텍스트는 외부 Google Generative LLM으로 인코딩했습니다. 최종 재처리 후 `External LLM fallback`은 0건입니다. 2026-06-05에는 LLM이 직접 읽는 최신 뉴스 제한이 모델 입력 병목이 되지 않도록, 전체 point-in-time 원시 뉴스 풀에서 계산한 14개 aggregate feature를 추가해 event/context 입력을 27차원으로 확장했습니다.

| 모델 | 학습 방식 | Train/Val/Test | 주요 성능 |
| --- | --- | ---: | --- |
| `oil_context_fusion_1d_h30` | chronological holdout | 1,651 / 353 / 353 | validation MAPE 5.45%, test MAPE 6.93%. validation/test RMSE 5.42/8.02, range ratio 1.32/1.26, shape score 90.5/87.6 |

이번 artifact는 과최적화 확인을 위해 final-fit이 아니라 시간순 holdout split으로 저장했습니다. Metadata는 전체 sample 범위(`sample_start=2016-12-07`, `sample_end=2026-04-23`)와 실제 train cutoff(`train_end=2023-07-03`, `training_cutoff=2023-07-03`)를 분리해 기록합니다.

2026-06-05에는 딥 모델 p50이 origin별 입력보다 horizon별 평균 템플릿을 과하게 반복하는 문제가 확인되어, 1D 경로 표시에는 point-in-time path adapter를 추가했습니다. 이 adapter는 미래 가격을 보지 않고, origin 시점까지의 가격 상태와 event/context 벡터만 사용합니다.

- 일반 구간: `pattern_mlp` residual shape를 절반 섞어 딥 모델의 고정 horizon 템플릿을 줄입니다.
- Geopolitical supply shock: 전쟁, 공격, 제재, 호르무즈/홍해 같은 공급 차질 뉴스가 강하면, 전체 bias가 mixed/neutral이어도 공급 리스크 프리미엄을 반영한 우상방 경로를 엽니다.
- Bullish geopolitical breakout: LLM/event encoder의 방향 점수, 원시 뉴스 bullish/geopolitical 압력, 최근 momentum, RSI를 사용해 빠른 상승 shock path를 만듭니다.
- Event risk premium: LLM 방향 점수가 하루 흔들려도 원시 뉴스 풀의 bullish/geopolitical/energy 압력이 지속되면 상승 리스크 프리미엄을 연속적으로 반영합니다. 이것은 threshold 하나가 어긋나면 갑자기 횡보로 떨어지는 문제를 줄이기 위한 장치입니다. 단조 직선 경로를 피하기 위해 이벤트는 terminal 방향과 레벨을 정하고, 경로의 고점/저점 잔차는 모델/모티프/최근 과거 경로에서 가져옵니다.
- Overextended mean reversion: 최근 30/60일 급등, RSI, 20일 고점 근접 상태를 사용해 초기 급락 후 회복하는 V자형 경로를 만듭니다.

이 adapter는 LLM을 숫자 가격 예측기로 쓰지 않습니다. LLM은 context/event encoder로만 쓰이고, 숫자 경로는 여전히 volatility-scaled cumulative log return 구조에서 복원합니다. Adapter가 작동하면 `deep_model_info.oil_context_fusion.path_adapter`에 종류와 공급충격 점수, 사용한 event/context 지표를 남겨 해설 API가 “왜 이런 방향성이 나왔는지” 설명할 수 있게 합니다.

## 학습 손실과 평가 지표

MAPE는 화면에 보고하기 좋은 지표지만 학습 손실로는 부족합니다. 가격 레벨에 민감하고, 경로 모양과 꼬리 사건을 충분히 벌하지 못합니다. 현재 deep loss는 다음 항을 함께 최적화합니다.

- Quantile pinball loss: p05-p95 분포 경로를 학습합니다.
- Median Huber loss: 중앙 경로의 안정적인 오차를 줄입니다.
- Step-return loss: 누적 경로가 아니라 하루하루 변화량도 맞추게 합니다.
- Terminal loss: 예측기간 말단의 누적 방향을 확인합니다.
- Path shape/range/step-volatility/curvature loss: 고점/저점, 경로 진폭, 곡률, 변동성을 평탄하게 뭉개지 않게 합니다.
- Step direction and direction-head loss: 각 step 방향성과 상승 확률 head를 학습합니다.
- Shock/range auxiliary loss: 충격장과 큰 range 구간을 별도로 학습합니다.
- Gaussian tail path loss: 정규분포 꼬리 바깥의 큰 누적경로 오차를 quadratic 이상으로 크게 벌합니다.
- Range shortfall tail loss: 실제 경로 range가 큰데 예측이 평탄하면 exponential tail penalty를 줍니다.

따라서 운영 판단에는 MAPE만 보지 않고 RMSE/MAE, sMAPE, step directional accuracy, range ratio, turn error, shape score를 함께 봅니다. 특히 스크린샷처럼 실제 급등 경로를 평탄하게 예측하는 경우는 새 tail loss와 range-shortfall loss에서 큰 손실로 계산됩니다.

## 학습 명령

1D h30 운영 artifact 예시:

```bash
.venv/bin/python scripts/train/train_deep_fusion_models.py \
  --model oil_context_fusion \
  --interval 1d \
  --horizon 30 \
  --lookback 128 \
  --symbols CL=F \
  --use-processed-data \
  --market-panel data/processed/market_panel/1d/panel.csv \
  --oil-fundamentals data/processed/oil_fundamentals/eia_weekly.csv \
  --cot data/processed/oil_fundamentals/cftc_cot_weekly.csv \
  --macro-panel data/processed/macro_panel/fred_daily_wide.csv \
  --event-context data/processed/event_context/event_context_daily.csv \
  --max-samples 0 \
  --epochs 28 \
  --batch-size 64 \
  --learning-rate 0.0007 \
  --patience 8 \
  --device mps \
  --force \
  --llm-context \
  --progress-every-batches 20
```

운영 배포용 최종 artifact를 전체 데이터로 다시 맞출 때는 별도 holdout 평가를 기록한 뒤 `--fit-final-all-data`를 추가합니다.

다른 시간봉은 `--interval`, `--horizon`, `--lookback`, `--market-panel`만 해당 주기에 맞게 바꿔 같은 구조로 학습합니다.

## Artifact와 Metadata

- 모델 artifact: `artifacts/models`
- metadata JSON: `artifacts/metadata`
- smoke test artifact: `artifacts/smoke`

Metadata에는 모델 이름, 주기, 예측 길이, sample 범위, 실제 train cutoff, 입력 데이터 경로, expert 목록, SSE/MSE/RMSE/MAE/R2/MAPE/sMAPE/방향정확도/range ratio/shape score를 기록합니다.

## 제거되거나 내부로 들어간 모델

- `lstm`, `tcn`: 단독 모델로는 운영 재현성이 낮아 통합 모델 내부 expert로 흡수했습니다.
- `motif`, `pattern_mlp`: 사용자에게 별도 모델로 보여주지 않고 통합 모델 내부 expert로 사용합니다.
- `cycle`: 단독 예측선보다 cycle 관련 특징으로 쓰는 편이 낫습니다.
- `ensemble`: 고정 비중 혼합은 시장 국면을 학습하지 못해, 현재는 학습형 가중치 조정 방식으로 대체했습니다.
