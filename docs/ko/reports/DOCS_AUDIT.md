# 문서 감사

이 문서는 현재 저장소의 Markdown 문서를 한국어 원본 + 영어 mirror 정책에 맞춰 분류한 결과입니다. 스캔은 `.git`, `.venv`, `venv`, `node_modules`, `__pycache__`, `.pytest_cache`, `dist`, `build`를 제외했습니다. `_archive`의 과거 프로젝트 문서는 active 문서가 아니라 obsolete/duplicate로 기록했습니다.

| path | category | language | paired_with | action | notes |
|---|---|---|---|---|---|
| `README.md` | root public docs | ko | `README.en.md` | keep | 한국어 메인 README |
| `README.en.md` | root public docs | en | `README.md` | keep | 영어 mirror README |
| `CHANGELOG.md` | root public docs | ko | `CHANGELOG.en.md` | keep | 한국어 변경 이력 |
| `CHANGELOG.en.md` | root public docs | en | `CHANGELOG.md` | keep | 영어 변경 이력 |
| `AGENTS.md` | root public docs | bilingual | 자체 포함 | keep | Codex 지침이므로 한국어와 영어를 한 파일에 함께 유지 |
| `docs/ko/ARCHITECTURE.md` | architecture docs | ko | `docs/en/ARCHITECTURE.md` | keep | 책임 분리와 의존성 방향 |
| `docs/en/ARCHITECTURE.md` | architecture docs | en | `docs/ko/ARCHITECTURE.md` | keep | Korean source mirror |
| `docs/ko/API.md` | API docs | ko | `docs/en/API.md` | keep | `/api/chart` compatibility와 `/api/forecast` 설명 |
| `docs/en/API.md` | API docs | en | `docs/ko/API.md` | keep | Korean source mirror |
| `docs/ko/MODEL_DESIGN.md` | model docs | ko | `docs/en/MODEL_DESIGN.md` | keep | forecast target과 artifact 정책 |
| `docs/en/MODEL_DESIGN.md` | model docs | en | `docs/ko/MODEL_DESIGN.md` | keep | Korean source mirror |
| `docs/ko/DATA_PIPELINE.md` | architecture docs | ko | `docs/en/DATA_PIPELINE.md` | keep | provider와 data quality 정책 |
| `docs/en/DATA_PIPELINE.md` | architecture docs | en | `docs/ko/DATA_PIPELINE.md` | keep | Korean source mirror |
| `docs/ko/BACKTESTING.md` | backtesting docs | ko | `docs/en/BACKTESTING.md` | keep | CLI, output, 검증 원칙 |
| `docs/en/BACKTESTING.md` | backtesting docs | en | `docs/ko/BACKTESTING.md` | keep | Korean source mirror |
| `docs/ko/LLM_CONTEXT.md` | LLM docs | ko | `docs/en/LLM_CONTEXT.md` | keep | LLM 허용/금지 역할 |
| `docs/en/LLM_CONTEXT.md` | LLM docs | en | `docs/ko/LLM_CONTEXT.md` | keep | Korean source mirror |
| `docs/ko/FRONTEND.md` | frontend docs | ko | `docs/en/FRONTEND.md` | keep | TradingView overlay UI 설명 |
| `docs/en/FRONTEND.md` | frontend docs | en | `docs/ko/FRONTEND.md` | keep | Korean source mirror |
| `docs/ko/OPERATIONS.md` | root public docs | ko | `docs/en/OPERATIONS.md` | keep | 반복 운영 명령 |
| `docs/en/OPERATIONS.md` | root public docs | en | `docs/ko/OPERATIONS.md` | keep | Korean source mirror |
| `docs/ko/ROADMAP.md` | root public docs | ko | `docs/en/ROADMAP.md` | keep | 확장 계획 |
| `docs/en/ROADMAP.md` | root public docs | en | `docs/ko/ROADMAP.md` | keep | Korean source mirror |
| `docs/ko/reports/ARCHITECTURE_AUDIT.md` | generated reports | ko | `docs/en/reports/ARCHITECTURE_AUDIT.md` | keep | 구조 감사 report |
| `docs/en/reports/ARCHITECTURE_AUDIT.md` | generated reports | en | `docs/ko/reports/ARCHITECTURE_AUDIT.md` | keep | Korean source mirror |
| `docs/ko/reports/IMPLEMENTATION_PLAN.md` | generated reports | ko | `docs/en/reports/IMPLEMENTATION_PLAN.md` | keep | 구현 계획 report |
| `docs/en/reports/IMPLEMENTATION_PLAN.md` | generated reports | en | `docs/ko/reports/IMPLEMENTATION_PLAN.md` | keep | Korean source mirror |
| `docs/ko/reports/CLEANUP_AUDIT.md` | generated reports | ko | `docs/en/reports/CLEANUP_AUDIT.md` | keep | 파일 정리 감사 report |
| `docs/en/reports/CLEANUP_AUDIT.md` | generated reports | en | `docs/ko/reports/CLEANUP_AUDIT.md` | keep | Korean source mirror |
| `docs/ko/reports/DOCS_AUDIT.md` | generated reports | ko | `docs/en/reports/DOCS_AUDIT.md` | keep | 이번 문서 감사 report |
| `docs/en/reports/DOCS_AUDIT.md` | generated reports | en | `docs/ko/reports/DOCS_AUDIT.md` | keep | Korean source mirror |
| `docs/ko/reports/ORDER_RECOVERY_AUDIT.md` | generated reports | ko | `docs/en/reports/ORDER_RECOVERY_AUDIT.md` | keep | 작업 순서 복구 감사 report |
| `docs/en/reports/ORDER_RECOVERY_AUDIT.md` | generated reports | en | `docs/ko/reports/ORDER_RECOVERY_AUDIT.md` | keep | Korean source mirror |
| `docs/ko/reports/FINAL_IMPLEMENTATION_REPORT.md` | generated reports | ko | `docs/en/reports/FINAL_IMPLEMENTATION_REPORT.md` | keep | 최종 구현 report |
| `docs/en/reports/FINAL_IMPLEMENTATION_REPORT.md` | generated reports | en | `docs/ko/reports/FINAL_IMPLEMENTATION_REPORT.md` | keep | Korean source mirror |
| `market_ai/llm/prompts/market_context.ko.md` | LLM docs | ko | `market_ai/llm/prompts/market_context.en.md` | keep | 코드 패키지 내부 prompt asset |
| `market_ai/llm/prompts/market_context.en.md` | LLM docs | en | `market_ai/llm/prompts/market_context.ko.md` | keep | 코드 패키지 내부 prompt asset |
| `_archive/legacy_20260429/oil-price-baseline/README.md` | obsolete/duplicate docs | en | 없음 | archive | 과거 oil baseline 실험 문서, active docs로 승격하지 않음 |
| `_archive/legacy_20260429/oil-tv-dashboard-remnants/.pytest_cache/README.md` | obsolete/duplicate docs | en | 없음 | ignore | pytest cache 설명 파일, active docs 아님 |

## 결과

Active public docs와 `docs/ko`, `docs/en` 문서는 모두 쌍으로 존재합니다. `_archive` 아래 문서는 보관용이며 현재 문서 정책의 active 대상이 아닙니다.
