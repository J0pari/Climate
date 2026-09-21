#!/usr/bin/env python3
"""Validate repository-local Markdown link targets without network access."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
REFERENCE_DEFINITION = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*(\S+)")
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
INLINE_CODE = re.compile(r"`[^`]*`")


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    line: int
    target: str
    message: str

    def render(self) -> str:
        return f"{self.code}: {self.path}:{self.line}: {self.message}: {self.target}"


def _target(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<"):
        end = value.find(">")
        if end >= 0:
            return value[1:end]
    # Markdown titles follow the destination after whitespace. Paths with spaces
    # should use angle brackets or percent encoding, both handled above/below.
    return value.split(None, 1)[0] if value else ""


def _local_target(target: str) -> str | None:
    target = target.strip()
    if not target or target.startswith("#") or target.startswith("//"):
        return None
    if SCHEME.match(target):
        return None
    path_part = target.split("#", 1)[0].split("?", 1)[0]
    if not path_part:
        return None
    return unquote(path_part)


def _resolve(root: Path, source: Path, target: str) -> Path:
    if target.startswith("/"):
        return (root / target.lstrip("/")).resolve()
    return (source.parent / target).resolve()


def markdown_paths(root: Path = ROOT) -> list[Path]:
    ignored_parts = {".git", "target", "build", ".snapshot-transfer"}
    result = []
    for path in root.rglob("*.md"):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part in ignored_parts for part in relative.parts):
            continue
        if path.is_file():
            result.append(path)
    return sorted(result)


def check(root: Path = ROOT) -> list[Finding]:
    findings: list[Finding] = []
    resolved_root = root.resolve()
    for source in markdown_paths(root):
        relative = source.relative_to(root).as_posix()
        fenced = False
        for line_no, original in enumerate(
            source.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            stripped = original.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fenced = not fenced
                continue
            if fenced:
                continue
            line = INLINE_CODE.sub("", original)
            raw_targets = [match.group(1) for match in INLINE_LINK.finditer(line)]
            definition = REFERENCE_DEFINITION.match(line)
            if definition:
                raw_targets.append(definition.group(1))
            for raw in raw_targets:
                target = _target(raw)
                local = _local_target(target)
                if local is None:
                    continue
                resolved = _resolve(root, source, local)
                try:
                    resolved.relative_to(resolved_root)
                except ValueError:
                    findings.append(
                        Finding(
                            "markdown_link.outside_repository",
                            relative,
                            line_no,
                            target,
                            "local Markdown target escapes repository",
                        )
                    )
                    continue
                if not resolved.exists():
                    findings.append(
                        Finding(
                            "markdown_link.missing_target",
                            relative,
                            line_no,
                            target,
                            "local Markdown target does not exist",
                        )
                    )
    return findings


def main() -> int:
    findings = check()
    for finding in findings:
        print(finding.render())
    if findings:
        print(f"\n{len(findings)} Markdown link finding(s)", file=sys.stderr)
        return 1
    print("Markdown local-link integrity: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
