# 운영

운영 문서는 개발자와 운영자가 반복해서 실행하는 명령과 환경 설정을 정리합니다.

## 서버 실행

```bash
uvicorn backend.app.main:app --reload --port 8000
```

기존 `app.main:app`은 compatibility wrapper입니다. 새 문서와 운영 명령은 `backend.app.main:app`을 기준으로 작성합니다.

## 학습

Legacy `.npz` fallback:

```bash
python scripts/train/train_pretrained_models.py --interval 1d
```

Deep model:

```bash
python scripts/train/train_deep_fusion_models.py --model both --interval 1d --universe oil_core --epochs 10 --batch-size 64
python scripts/train/train_deep_fusion_models.py --model deep_lstm_tcn_fusion --interval 1d --quick-test --epochs 1 --max-samples 256
```

Artifact는 `artifacts/models`, metadata JSON은 `artifacts/metadata`에 저장됩니다.

## 백테스트

```bash
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 10 --models random_walk,drift,motif,pattern_mlp,deep_lstm_tcn_fusion,llm_context_seq_moe --no-plots
```

Deep artifact가 없으면 해당 모델은 unavailable로 기록되고 전체 backtest는 계속됩니다.

## API와 차트

`/api/forecast?symbol=CL=F&interval=1d&models=deep_lstm_tcn_fusion`처럼 model selector를 전달할 수 있습니다. Frontend model selector도 같은 query를 사용합니다. `/api/chart` schema는 backward compatibility를 유지합니다.

## 유지보수

```bash
python scripts/maintenance/check_docs_i18n.py --check-legacy
python -m compileall backend market_ai scripts
python scripts/maintenance/smoke_test_api.py
python -m pytest
```

로컬 shell에 `python`이 없으면 `.venv/bin/python`을 사용합니다.

## 기본 경로

명시적으로 override하지 않는 한 `MODEL_DIR=artifacts/models`, `METADATA_DIR=artifacts/metadata`, `DATA_DIR=data`를 사용합니다.
