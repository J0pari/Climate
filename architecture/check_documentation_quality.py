#!/usr/bin/env python3
"""Reject non-factual quality self-assessment and temporal status prose.

Documentation should state mechanisms, contracts, tests, metrics, limitations,
and authority paths. It should not award the repository or a method prestige
qualities by adjective, and it should not maintain hand-authored repository
state snapshots that belong in machine authorities, exact-head execution
records, or Git history.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

QUALITY_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "quality.rigor_self_certification",
        re.compile(r"\brigor(?:ous|ously)\b", re.IGNORECASE),
        "replace claims of rigor with the concrete test, proof obligation, metric, or scope",
    ),
    (
        "quality.prestige_adjective",
        re.compile(
            r"\b(?:technically\s+serious|sophisticated|trustworthy|"
            r"state[- ]of[- ]the[- ]art|production[- ]grade|world[- ]class|"
            r"best[- ]in[- ]class|high[- ]quality)\b",
            re.IGNORECASE,
        ),
        "replace prestige language with the capability or criterion it is meant to denote",
    ),
    (
        "quality.vague_scientific_meaning",
        re.compile(
            r"\bscientifically\s+meaningful\s+"
            r"(?:axis|axes|task|tasks|view|views|experiment|experiments|"
            r"split|representation|representations)\b",
            re.IGNORECASE,
        ),
        "name the declared endpoint, task, partition, or decision rule instead",
    ),
)

TEMPORAL_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "quality.temporal_currently",
        re.compile(r"\bcurrently\b", re.IGNORECASE),
        "durable prose should state the contract directly or defer repository state to its authority",
    ),
    (
        "quality.temporal_at_present",
        re.compile(r"\bat\s+present\b", re.IGNORECASE),
        "durable prose should not encode an unversioned temporal snapshot",
    ),
    (
        "quality.temporal_already_contains",
        re.compile(
            r"\b(?:the\s+)?repository\s+already\s+"
            r"(?:contains|has|includes|provides|supports|implements)\b|"
            r"\balready\s+contains\b",
            re.IGNORECASE,
        ),
        "name the authoritative path directly instead of narrating repository progress",
    ),
    (
        "quality.temporal_now_capability",
        re.compile(
            r"\b(?:the\s+)?repository\s+now\s+"
            r"(?:contains|has|includes|provides|supports|implements|realizes)\b",
            re.IGNORECASE,
        ),
        "repository change narration belongs in Git history or generated state",
    ),
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str
    text: str

    def render(self) -> str:
        return (
            f"{self.rule}: {self.path}:{self.line}: {self.message}\n"
            f"    {self.text.strip()}"
        )


def documentation_paths(root: Path = ROOT) -> list[Path]:
    paths: set[Path] = set()
    readme = root / "README.md"
    if readme.is_file():
        paths.add(readme)
    docs = root / "docs"
    if docs.is_dir():
        for path in docs.rglob("*.md"):
            relative = path.relative_to(root)
            if relative.parts[:2] == ("docs", "archive"):
                continue
            paths.add(path)
    return sorted(paths)


def findings_for_text(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    patterns = QUALITY_PATTERNS + TEMPORAL_PATTERNS
    for line_number, line in enumerate(text.splitlines(), 1):
        for rule, pattern, message in patterns:
            if pattern.search(line):
                findings.append(Finding(path, line_number, rule, message, line))
    return findings


def check_repository(root: Path = ROOT) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    for path in documentation_paths(root):
        relative = path.relative_to(root).as_posix()
        findings.extend(
            findings_for_text(
                relative,
                path.read_text(encoding="utf-8", errors="replace"),
            )
        )
    return findings


def main() -> int:
    findings = check_repository()
    for finding in findings:
        print(finding.render())
    if findings:
        print(f"\n{len(findings)} documentation-quality finding(s)", file=sys.stderr)
        return 1
    print("documentation quality semantics: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
