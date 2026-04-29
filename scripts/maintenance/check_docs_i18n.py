#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_ROOT_DOCS = (
    "README.md",
    "README.en.md",
    "CHANGELOG.md",
    "CHANGELOG.en.md",
    "AGENTS.md",
)

ROOT_DOC_PAIRS = (
    ("README.md", "README.en.md"),
    ("CHANGELOG.md", "CHANGELOG.en.md"),
)

FORBIDDEN_ROOT_REPORTS = (
    "ARCHITECTURE_AUDIT.md",
    "IMPLEMENTATION_PLAN.md",
    "FINAL_IMPLEMENTATION_REPORT.md",
    "CLEANUP_AUDIT.md",
    "DOCS_AUDIT.md",
)

LEGACY_PATTERNS = (
    ("old uvicorn entrypoint", re.compile(r"\buvicorn\s+app\.main:app\b")),
    ("old train command", re.compile(r"\bpython\s+train_pretrained_models\.py\b")),
    ("old backtest command", re.compile(r"\bpython\s+backtest_forecasters\.py\b")),
    ("old model artifact glob", re.compile(r"app/models/\*\.npz")),
    ("old model directory", re.compile(r"app/models/")),
    ("absolute local path", re.compile(r"/Users/|/private/")),
    ("legacy dashboard path", re.compile(r"\boil-tv-dashboard\b")),
    ("legacy baseline path", re.compile(r"\boil-price-baseline\b")),
    ("silent mock fallback phrase", re.compile(r"mock 데이터로 자동 대체")),
    ("overstated confidence interval phrase", re.compile(r"95%\s*신뢰구간")),
)

REPORT_HISTORY_PATTERNS = {
    "old uvicorn entrypoint",
    "old train command",
    "old backtest command",
    "old model artifact glob",
    "old model directory",
    "legacy dashboard path",
    "legacy baseline path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Korean/English Markdown pairs and documentation hygiene."
    )
    parser.add_argument(
        "--check-legacy",
        action="store_true",
        help="also scan active docs for obsolete path strings",
    )
    return parser.parse_args()


def relative_markdown_files(base: Path) -> set[Path]:
    if not base.exists():
        return set()
    return {path.relative_to(base) for path in base.rglob("*.md") if path.is_file()}


def active_markdown_files() -> list[Path]:
    files = [ROOT / name for name in REQUIRED_ROOT_DOCS if (ROOT / name).exists()]
    docs_root = ROOT / "docs"
    if docs_root.exists():
        files.extend(path for path in docs_root.rglob("*.md") if path.is_file())
    return sorted(set(files))


def is_report_file(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return len(rel.parts) >= 4 and rel.parts[0] == "docs" and rel.parts[2] == "reports"


def check_legacy_strings() -> tuple[list[str], int]:
    failures: list[str] = []
    allowed_historical = 0
    for path in active_markdown_files():
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(ROOT))
        for label, pattern in LEGACY_PATTERNS:
            if not pattern.search(text):
                continue
            if is_report_file(path) and label in REPORT_HISTORY_PATTERNS:
                allowed_historical += 1
                continue
            failures.append(f"{rel}: {label}")
    return failures, allowed_historical


def main() -> int:
    args = parse_args()
    missing: list[str] = []
    hygiene_failures: list[str] = []

    for name in REQUIRED_ROOT_DOCS:
        if not (ROOT / name).exists():
            missing.append(name)

    for ko_name, en_name in ROOT_DOC_PAIRS:
        if (ROOT / ko_name).exists() != (ROOT / en_name).exists():
            missing.append(f"{ko_name} / {en_name} pair")

    for name in FORBIDDEN_ROOT_REPORTS:
        if (ROOT / name).exists():
            hygiene_failures.append(f"root generated report must move into docs/*/reports: {name}")

    ko_root = ROOT / "docs" / "ko"
    en_root = ROOT / "docs" / "en"
    ko_files = relative_markdown_files(ko_root)
    en_files = relative_markdown_files(en_root)

    for rel in sorted(ko_files - en_files):
        missing.append(str((en_root / rel).relative_to(ROOT)))
    for rel in sorted(en_files - ko_files):
        missing.append(str((ko_root / rel).relative_to(ROOT)))

    legacy_failures: list[str] = []
    allowed_historical = 0
    if args.check_legacy:
        legacy_failures, allowed_historical = check_legacy_strings()

    if missing or hygiene_failures or legacy_failures:
        print("Documentation i18n check failed.")
    if missing:
        print("Missing files:")
        for path in missing:
            print(f"- {path}")
    if hygiene_failures:
        print("Repository hygiene failures:")
        for failure in hygiene_failures:
            print(f"- {failure}")
    if legacy_failures:
        print("Obsolete documentation strings:")
        for failure in legacy_failures:
            print(f"- {failure}")
    if missing or hygiene_failures or legacy_failures:
        return 1

    print("Documentation i18n check passed.")
    print(f"- required root docs: {len(REQUIRED_ROOT_DOCS)}")
    print(f"- root pairs: {len(ROOT_DOC_PAIRS)}")
    print(f"- docs/ko markdown files: {len(ko_files)}")
    print(f"- docs/en markdown files: {len(en_files)}")
    if args.check_legacy:
        print(f"- allowed historical report references: {allowed_historical}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
