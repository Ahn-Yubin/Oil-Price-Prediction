# 운영

이 문서는 반복 실행 명령, 환경변수, 데이터 구축, 학습, 검증, 서버 운영 절차를 정리합니다. 로컬 shell에 `python`이 없으면 모든 명령에서 `.venv/bin/python`을 사용합니다.

## 서버 실행

```bash
.venv/bin/python -m uvicorn backend.app.main:app --reload --port 8000
```

기존 `app.main:app`은 compatibility wrapper입니다. 새 운영 명령은 `backend.app.main:app`을 기준으로 합니다.

환경변수를 바꾼 뒤 이미 떠 있는 서버에 자동 반영되지 않습니다. LLM key, model, data path를 바꿨다면 서버를 재시작합니다.

## export와 .env

`export`는 현재 터미널 session과 그 session에서 실행한 child process에만 적용됩니다. 터미널을 닫으면 사라지고 다른 터미널에는 전달되지 않습니다.

현재 shell 설정 확인:

```bash
echo "$ENABLE_LLM_CONTEXT"
echo "$ENABLE_EXTERNAL_LLM_CALLS"
echo "$LLM_CONTEXT_MODE"
echo "$LLM_API_BASE"
echo "$LLM_MODEL"
test -n "$LLM_API_KEY" && echo "LLM_API_KEY is set" || echo "LLM_API_KEY is missing"
```

영구 설정 선택지는 두 가지입니다.

1. 개인 shell 기본값으로 쓰려면 `~/.zshrc`에 export를 넣습니다.
2. 프로젝트별로 쓰려면 `.env`를 만듭니다. 주요 server/script entrypoint는 프로젝트 루트 `.env`를 자동 로드합니다.

```bash
cp .env.example .env
# .env 안의 LLM_API_KEY와 LLM_MODEL을 편집
```

`.env`와 `.env.*`는 `.gitignore`에 포함되어 있습니다. 그래도 API key를 출력하거나 커밋하지 않습니다.

shell의 `echo "$LLM_MODEL"`은 `.env` 값을 자동으로 보여주지 않습니다. 프로젝트가 읽는 설정은 다음으로 확인합니다.

```bash
.venv/bin/python - <<'PY'
from market_ai.config import get_settings
s = get_settings()
print("enable_llm_context:", s.enable_llm_context)
print("enable_external_llm_calls:", s.enable_external_llm_calls)
print("llm_model:", s.llm_model)
print("llm_api_base:", s.llm_api_base)
print("llm_api_key_set:", bool(s.llm_api_key))
PY
```

## Google Gemma/Gemini LLM Context

Google OpenAI-compatible endpoint를 쓸 때 예시:

```bash
export ENABLE_LLM_CONTEXT=true
export ENABLE_EXTERNAL_LLM_CALLS=true
export LLM_CONTEXT_MODE=google_generative
export LLM_API_KEY="YOUR_GOOGLE_API_KEY"
export LLM_API_BASE="https://generativelanguage.googleapis.com/v1beta"
export LLM_MODEL="gemma-3-27b-it"
```

Dry-run:

```bash
.venv/bin/python scripts/llm/test_llm_context.py --mode openai_compatible --dry-run
```

Live call:

```bash
.venv/bin/python scripts/llm/test_llm_context.py --mode google_generative --live
```

성공 기준은 `safety_check_passed=true`이고 `warnings`에 `External LLM fallback`이 없는 것입니다.

## 데이터 구축

Market panel:

```bash
.venv/bin/python scripts/data/fetch_market_prices.py --universe research_core --interval 1d --period 10y
.venv/bin/python scripts/data/fetch_market_prices.py --universe research_core --interval 1h --period 730d
.venv/bin/python scripts/data/fetch_market_prices.py --universe research_core --interval 30m --period 60d
.venv/bin/python scripts/data/fetch_market_prices.py --universe research_core --interval 15m --period 60d
```

Oil fundamentals and positioning:

```bash
.venv/bin/python scripts/data/fetch_eia_petroleum.py
.venv/bin/python scripts/data/fetch_cftc_cot.py
.venv/bin/python scripts/data/fetch_cme_settlements.py --manual-csv data/external/fundamentals/cme_settlements.csv
```

Event context:

```bash
.venv/bin/python scripts/data/build_event_context.py \
  --news-path data/raw/news/public_market_news.csv \
  --symbols CL=F,BZ=F,NG=F,RB=F,HO=F,GC=F,SI=F,HG=F,DX-Y.NYB,EURUSD=X,USDKRW=X,JPY=X,SPY,QQQ,^GSPC,^VIX,XLE,USO \
  --mode google_generative \
  --live
```

Manifest:

```bash
.venv/bin/python scripts/data/build_data_inventory.py
```

## 학습

현재 보유한 processed data를 사용한 1일봉 학습 예시:

```bash
.venv/bin/python scripts/train/train_deep_fusion_models.py \
  --model both \
  --interval 1d \
  --horizon 8 \
  --lookback 128 \
  --universe research_core \
  --use-processed-data \
  --market-panel data/processed/market_panel/1d/panel.csv \
  --oil-fundamentals data/processed/oil_fundamentals/eia_weekly.csv \
  --cot data/processed/oil_fundamentals/cftc_cot_weekly.csv \
  --event-context data/processed/event_context/event_context_daily.csv \
  --max-samples 512 \
  --epochs 3 \
  --batch-size 64 \
  --device mps \
  --force
```

Longer horizon artifact:

```bash
.venv/bin/python scripts/train/train_deep_fusion_models.py \
  --model both \
  --interval 1d \
  --horizon 45 \
  --lookback 128 \
  --universe research_core \
  --use-processed-data \
  --market-panel data/processed/market_panel/1d/panel.csv \
  --oil-fundamentals data/processed/oil_fundamentals/eia_weekly.csv \
  --cot data/processed/oil_fundamentals/cftc_cot_weekly.csv \
  --event-context data/processed/event_context/event_context_daily.csv \
  --max-samples 512 \
  --epochs 3 \
  --batch-size 64 \
  --device mps \
  --force
```

Production training은 `--synthetic`, `--quick-test`, `--allow-synthetic-fallback` 없이 synthetic fallback을 사용하지 않습니다. Artifact는 `artifacts/models`, metadata JSON은 `artifacts/metadata`에 저장됩니다.

## 백테스트와 Calibration

```bash
.venv/bin/python scripts/backtest/run_backtest.py \
  --symbol CL=F \
  --interval 1d \
  --max-origins 10 \
  --models random_walk,drift,motif,pattern_mlp,deep_lstm_tcn_fusion,llm_context_seq_moe \
  --no-plots
```

Leaderboard와 calibration:

```bash
.venv/bin/python scripts/evaluate/run_model_leaderboard.py --symbols CL=F,BZ=F,NG=F --interval 1d --max-origins 50
.venv/bin/python scripts/evaluate/calibrate_quantiles.py --model motif --symbol CL=F --interval 1d
```

Calibration artifact가 충분히 검증되기 전까지 forecast band는 residual-volatility adapter이며 검증된 confidence interval이 아닙니다.

## API와 차트

- `/api/forecast`: 신규 forecast contract
- `/api/chart`: 기존 chart compatibility contract
- `/api/market-context`: 뉴스, context marker, 모델 시나리오 해설

예시:

```bash
curl "http://127.0.0.1:8000/api/market-context?symbol=NYMEX:CL1%21&interval=1d&models=llm_context_seq_moe"
```

## 검증

```bash
.venv/bin/python -m pytest tests/integration/test_api.py tests/unit/test_real_data_pipeline.py tests/unit/test_deep_dataset.py tests/unit/test_train_deep_fusion_cli_policy.py
.venv/bin/python -m compileall backend market_ai scripts
.venv/bin/python scripts/maintenance/check_docs_i18n.py --check-legacy
```

Frontend JS syntax:

```bash
node --check frontend/src/main.js
```

`npm`이 PATH에 없으면 Vite build는 실행할 수 없습니다. Node runtime이 별도 위치에 있으면 해당 `node` binary로 `--check`를 실행합니다.
