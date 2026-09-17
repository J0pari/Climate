#!/usr/bin/env python3
"""Reject edit-history narration from durable Climate text surfaces.

Git history owns change narration. Repository documentation and source comments
should describe the current contract, rationale, assumptions, limitations, and
invariants rather than recording how a line or feature was edited.

The binding surface deliberately covers active documentation plus canonical and
reference source trees. Historical archives and generated status are excluded:
archives may preserve historical wording, while generated status has its own
machine authority and drift check. Legacy root-level source remains visible to
``source_gates.py`` and can be sanitized incrementally without weakening this
binding rule for canonical work.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".md", ".rs", ".py", ".cu", ".cuh", ".cpp", ".cc", ".cxx",
    ".h", ".hpp", ".f90", ".F90", ".jl", ".hs",
}

# These patterns target unmistakable edit-history annotations rather than
# scientific uses of words such as "additive" or "previous state".
HISTORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "edit_batch_annotation",
        re.compile(r"\b(?:batch\s*\d+\s+)?additive\s*:", re.IGNORECASE),
    ),
    (
        "edit_batch_label",
        re.compile(r"\bbatch\s*\d+\s+additive\b", re.IGNORECASE),
    ),
    (
        "replacement_history",
        re.compile(r"\breplaces?\s+previously\b", re.IGNORECASE),
    ),
    (
        "recent_change_narration",
        re.compile(
            r"\brecently\s+(?:added|removed|retired|changed|updated|"
            r"implemented|fixed|introduced)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "repository_temporal_snapshot",
        re.compile(
            r"\b(?:the\s+)?repository\s+now\s+(?:has|contains|includes|"
            r"provides|supports|uses|realizes)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "provisional_for_now",
        re.compile(r"\bfor\s+now\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    text: str

    def render(self) -> str:
        return (
            f"{self.rule}: {self.path}:{self.line}: durable text describes edit "
            f"history; move change narration to the commit message\n"
            f"    {self.text.strip()}"
        )


def findings_for_text(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        for rule, pattern in HISTORY_PATTERNS:
            if pattern.search(line):
                findings.append(Finding(path, line_number, rule, line))
    return findings


def _active_markdown(root: Path) -> Iterable[Path]:
    for relative in (Path("README.md"), Path("AGENTS.md")):
        path = root / relative
        if path.is_file():
            yield path

    for directory in (root / "docs", root / "blueprints"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.md")):
            relative = path.relative_to(root)
            if relative.parts[:2] in {("docs", "archive"), ("docs", "generated")}:
                continue
            yield path


def _active_source(root: Path) -> Iterable[Path]:
    for directory in (root / "src", root / "reference"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                yield path

    cmake = root / "CMakeLists.txt"
    if cmake.is_file():
        yield cmake


def durable_paths(root: Path = ROOT) -> list[Path]:
    root = root.resolve()
    paths = {*_active_markdown(root), *_active_source(root)}
    return sorted(paths)


def check_repository(root: Path = ROOT) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    for path in durable_paths(root):
        relative = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        findings.extend(findings_for_text(relative, text))
    return findings


def main() -> int:
    findings = check_repository()
    for finding in findings:
        print(finding.render())
    if findings:
        print(f"\n{len(findings)} durable-text finding(s)", file=sys.stderr)
        return 1
    print("durable documentation/comment semantics: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
