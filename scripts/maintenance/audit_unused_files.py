#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
    "_archive",
}
LEGACY_RE = re.compile(r"(old|backup|legacy|temp|tmp|copy)", re.IGNORECASE)
DOC_EXTENSIONS = {".md", ".rst", ".txt"}


@dataclass
class FileAudit:
    path: str
    current_purpose: str
    imported_by: list[str]
    referenced_by_docs: list[str]
    action: str
    reason: str
    risk: str
    new_path: str


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def module_name(path: Path) -> str | None:
    if path.suffix != ".py":
        return None
    rel = path.relative_to(ROOT).with_suffix("")
    if rel.name == "__init__":
        rel = rel.parent
    return ".".join(rel.parts)


def imported_modules(path: Path) -> set[str]:
    if path.suffix != ".py":
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
            imports.update(f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*")
    return imports


def purpose_for(path: Path) -> str:
    rel = path.relative_to(ROOT)
    if rel.parts[0] in {"backend", "market_ai", "scripts", "frontend", "tests", "docs"}:
        return rel.parts[0]
    if rel.parts[0] == "artifacts":
        return "model artifact or metadata"
    if rel.parts[0] == "outputs":
        return "generated output"
    if rel.name == ".env.example":
        return "environment template"
    return "repository file"


def action_for(path: Path, imported_by: list[str], doc_refs: list[str]) -> tuple[str, str, str, str]:
    rel = path.relative_to(ROOT)
    rel_text = str(rel)
    if rel.name in {".gitignore", "AGENTS.md", "README.md", "README.en.md", "CHANGELOG.md", "CHANGELOG.en.md", "requirements.txt", "pytest.ini"}:
        return "keep", "root project metadata is part of the target structure", "low", ""
    if rel.name == ".gitkeep":
        return "keep", "directory placeholder is intentionally versioned", "low", ""
    if rel_text.startswith("artifacts/models/") or rel_text.startswith("artifacts/metadata/"):
        return "keep", "model artifact or metadata must be preserved", "high", ""
    if rel_text.startswith("docs/") or rel_text.startswith("tests/") or rel.name == ".env.example":
        return "keep", "documentation, test, or env template is protected", "low", ""
    if rel.parts[0] in {"backend", "frontend", "market_ai", "scripts", "configs", "notebooks", "app"}:
        return "keep", "file is under the target platform structure", "low", ""
    if imported_by:
        return "keep", "referenced by Python import graph", "medium", ""
    if doc_refs:
        return "keep", "referenced by documentation", "medium", ""
    if rel_text.startswith("outputs/"):
        return "keep", "generated output is already isolated under outputs", "low", ""
    if LEGACY_RE.search(rel.name):
        return "archive", "legacy/temp naming pattern without import reference", "medium", f"_archive/legacy_YYYYMMDD/{rel_text}"
    if rel.suffix in {".pyc", ".DS_Store"}:
        return "delete", "generated local file", "low", ""
    return "unknown", "no direct import or docs reference found", "medium", ""


def build_audit() -> list[FileAudit]:
    files = iter_files(ROOT)
    modules = {module_name(path): path for path in files if module_name(path)}
    imports_by_file = {path: imported_modules(path) for path in files}
    imported_by: dict[Path, list[str]] = {path: [] for path in files}
    for source, imports in imports_by_file.items():
        for mod, target in modules.items():
            if mod and (mod in imports or any(item.startswith(f"{mod}.") for item in imports)):
                imported_by[target].append(str(source.relative_to(ROOT)))

    docs = [path for path in files if path.suffix in DOC_EXTENSIONS]
    audits: list[FileAudit] = []
    for path in files:
        rel_text = str(path.relative_to(ROOT))
        doc_refs = []
        for doc in docs:
            if doc == path:
                continue
            try:
                text = doc.read_text(encoding="utf-8")
            except Exception:
                continue
            if rel_text in text or path.name in text:
                doc_refs.append(str(doc.relative_to(ROOT)))
        action, reason, risk, new_path = action_for(path, sorted(imported_by[path]), sorted(doc_refs))
        audits.append(
            FileAudit(
                path=rel_text,
                current_purpose=purpose_for(path),
                imported_by=sorted(imported_by[path]),
                referenced_by_docs=sorted(doc_refs),
                action=action,
                reason=reason,
                risk=risk,
                new_path=new_path,
            )
        )
    return audits


def markdown(audits: list[FileAudit]) -> str:
    lines = [
        "| path | current purpose | imported_by | referenced_by_docs | action | reason | risk | new_path |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in audits:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.path,
                    row.current_purpose,
                    "<br>".join(row.imported_by) or "-",
                    "<br>".join(row.referenced_by_docs) or "-",
                    row.action,
                    row.reason,
                    row.risk,
                    row.new_path or "-",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit unused and legacy files without deleting them.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    audits = build_audit()
    payload = json.dumps([asdict(row) for row in audits], ensure_ascii=False, indent=2) if args.format == "json" else markdown(audits)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
