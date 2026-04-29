# 운영

운영 문서는 개발자와 운영자가 반복해서 실행하는 명령과 환경 설정을 정리합니다.

## 서버 실행

```bash
uvicorn backend.app.main:app --reload --port 8000
```

기존 `app.main:app`은 compatibility wrapper입니다. 새 문서와 운영 명령은 `backend.app.main:app`을 기준으로 작성합니다.

## 학습

```bash
python scripts/train/train_pretrained_models.py --interval 1d
```

## 백테스트

```bash
python scripts/backtest/run_backtest.py --symbol CL=F --interval 1d --max-origins 5 --models random_walk,drift,flat --no-plots
```

## 유지보수

```bash
python scripts/maintenance/check_docs_i18n.py
python scripts/maintenance/audit_unused_files.py
python scripts/maintenance/smoke_test_api.py
```

## 기본 경로

명시적으로 override하지 않는 한 `MODEL_DIR=artifacts/models`, `METADATA_DIR=artifacts/metadata`, `DATA_DIR=data`를 사용합니다.
