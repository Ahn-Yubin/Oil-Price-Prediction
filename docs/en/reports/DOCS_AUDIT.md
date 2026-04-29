# Documentation Audit

This document classifies the repository Markdown files against the Korean source + English mirror policy. The scan excludes `.git`, `.venv`, `venv`, `node_modules`, `__pycache__`, `.pytest_cache`, `dist`, and `build`. Historical project docs under `_archive` are recorded as obsolete/duplicate, not active docs.

| path | category | language | paired_with | action | notes |
|---|---|---|---|---|---|
| `README.md` | root public docs | ko | `README.en.md` | keep | Korean primary README |
| `README.en.md` | root public docs | en | `README.md` | keep | English mirror README |
| `CHANGELOG.md` | root public docs | ko | `CHANGELOG.en.md` | keep | Korean changelog |
| `CHANGELOG.en.md` | root public docs | en | `CHANGELOG.md` | keep | English changelog |
| `AGENTS.md` | root public docs | bilingual | self-contained | keep | Codex instructions keep Korean and English in one file |
| `docs/ko/ARCHITECTURE.md` | architecture docs | ko | `docs/en/ARCHITECTURE.md` | keep | Responsibilities and dependency direction |
| `docs/en/ARCHITECTURE.md` | architecture docs | en | `docs/ko/ARCHITECTURE.md` | keep | Korean source mirror |
| `docs/ko/API.md` | API docs | ko | `docs/en/API.md` | keep | `/api/chart` compatibility and `/api/forecast` |
| `docs/en/API.md` | API docs | en | `docs/ko/API.md` | keep | Korean source mirror |
| `docs/ko/MODEL_DESIGN.md` | model docs | ko | `docs/en/MODEL_DESIGN.md` | keep | Forecast target and artifact policy |
| `docs/en/MODEL_DESIGN.md` | model docs | en | `docs/ko/MODEL_DESIGN.md` | keep | Korean source mirror |
| `docs/ko/DATA_PIPELINE.md` | architecture docs | ko | `docs/en/DATA_PIPELINE.md` | keep | Provider and data quality policy |
| `docs/en/DATA_PIPELINE.md` | architecture docs | en | `docs/ko/DATA_PIPELINE.md` | keep | Korean source mirror |
| `docs/ko/BACKTESTING.md` | backtesting docs | ko | `docs/en/BACKTESTING.md` | keep | CLI, outputs, validation principles |
| `docs/en/BACKTESTING.md` | backtesting docs | en | `docs/ko/BACKTESTING.md` | keep | Korean source mirror |
| `docs/ko/LLM_CONTEXT.md` | LLM docs | ko | `docs/en/LLM_CONTEXT.md` | keep | Allowed and forbidden LLM roles |
| `docs/en/LLM_CONTEXT.md` | LLM docs | en | `docs/ko/LLM_CONTEXT.md` | keep | Korean source mirror |
| `docs/ko/FRONTEND.md` | frontend docs | ko | `docs/en/FRONTEND.md` | keep | TradingView overlay UI |
| `docs/en/FRONTEND.md` | frontend docs | en | `docs/ko/FRONTEND.md` | keep | Korean source mirror |
| `docs/ko/OPERATIONS.md` | root public docs | ko | `docs/en/OPERATIONS.md` | keep | Repeated operations commands |
| `docs/en/OPERATIONS.md` | root public docs | en | `docs/ko/OPERATIONS.md` | keep | Korean source mirror |
| `docs/ko/ROADMAP.md` | root public docs | ko | `docs/en/ROADMAP.md` | keep | Expansion plan |
| `docs/en/ROADMAP.md` | root public docs | en | `docs/ko/ROADMAP.md` | keep | Korean source mirror |
| `docs/ko/reports/ARCHITECTURE_AUDIT.md` | generated reports | ko | `docs/en/reports/ARCHITECTURE_AUDIT.md` | keep | Architecture audit report |
| `docs/en/reports/ARCHITECTURE_AUDIT.md` | generated reports | en | `docs/ko/reports/ARCHITECTURE_AUDIT.md` | keep | Korean source mirror |
| `docs/ko/reports/IMPLEMENTATION_PLAN.md` | generated reports | ko | `docs/en/reports/IMPLEMENTATION_PLAN.md` | keep | Implementation plan report |
| `docs/en/reports/IMPLEMENTATION_PLAN.md` | generated reports | en | `docs/ko/reports/IMPLEMENTATION_PLAN.md` | keep | Korean source mirror |
| `docs/ko/reports/CLEANUP_AUDIT.md` | generated reports | ko | `docs/en/reports/CLEANUP_AUDIT.md` | keep | File cleanup audit report |
| `docs/en/reports/CLEANUP_AUDIT.md` | generated reports | en | `docs/ko/reports/CLEANUP_AUDIT.md` | keep | Korean source mirror |
| `docs/ko/reports/DOCS_AUDIT.md` | generated reports | ko | `docs/en/reports/DOCS_AUDIT.md` | keep | This documentation audit report |
| `docs/en/reports/DOCS_AUDIT.md` | generated reports | en | `docs/ko/reports/DOCS_AUDIT.md` | keep | Korean source mirror |
| `docs/ko/reports/ORDER_RECOVERY_AUDIT.md` | generated reports | ko | `docs/en/reports/ORDER_RECOVERY_AUDIT.md` | keep | Order recovery audit report |
| `docs/en/reports/ORDER_RECOVERY_AUDIT.md` | generated reports | en | `docs/ko/reports/ORDER_RECOVERY_AUDIT.md` | keep | Korean source mirror |
| `docs/ko/reports/FINAL_IMPLEMENTATION_REPORT.md` | generated reports | ko | `docs/en/reports/FINAL_IMPLEMENTATION_REPORT.md` | keep | Final implementation report |
| `docs/en/reports/FINAL_IMPLEMENTATION_REPORT.md` | generated reports | en | `docs/ko/reports/FINAL_IMPLEMENTATION_REPORT.md` | keep | Korean source mirror |
| `market_ai/llm/prompts/market_context.ko.md` | LLM docs | ko | `market_ai/llm/prompts/market_context.en.md` | keep | Prompt asset inside code package |
| `market_ai/llm/prompts/market_context.en.md` | LLM docs | en | `market_ai/llm/prompts/market_context.ko.md` | keep | Prompt asset inside code package |
| `_archive/legacy_20260429/oil-price-baseline/README.md` | obsolete/duplicate docs | en | none | archive | Historical oil baseline experiment doc, not promoted to active docs |
| `_archive/legacy_20260429/oil-tv-dashboard-remnants/.pytest_cache/README.md` | obsolete/duplicate docs | en | none | ignore | pytest cache explanation file, not active docs |

## Result

Active public docs and the `docs/ko`, `docs/en` trees have paired files. Documents under `_archive` are retained for history and are not active policy targets.
