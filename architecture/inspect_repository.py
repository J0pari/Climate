#!/usr/bin/env python3
"""Static repository inspector for Climate.

This tool never executes Climate scientific code. It inspects repository shape,
build metadata, and architecture audit findings so humans and Commons can learn
whether the checkout is structurally coherent before granting execution rights.

Output codes are intended to be stable machine interfaces. Messages may improve
without changing the code when the underlying condition is unchanged.
"""
from __future__ import annotations

import argparse
import json
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from architecture import source_gates

ROOT = Path(__file__).resolve().parents[1]

LANGUAGES = {
    ".rs": "rust",
    ".py": "python",
    ".cu": "cuda",
    ".cuh": "cuda-header",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".f90": "fortran",
    ".F90": "fortran",
    ".jl": "julia",
    ".hs": "haskell",
    ".cue": "cue",
}


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str
    path: str | None = None
    detail: dict | None = None


@dataclass(frozen=True)
class SourceEntry:
    path: str
    language: str
    bytes: int


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def source_inventory(root: Path) -> list[SourceEntry]:
    entries: list[SourceEntry] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in LANGUAGES:
            continue
        rel = path.relative_to(root)
        if any(part in {".git", "build", "target", ".venv", "venv", "__pycache__"}
               for part in rel.parts):
            continue
        entries.append(SourceEntry(
            path=rel.as_posix(),
            language=LANGUAGES[path.suffix],
            bytes=path.stat().st_size,
        ))
    return entries


def inspect_required_architecture_files(root: Path) -> list[Finding]:
    required = (
        "AGENTS.md",
        "docs/ARCHITECTURE.md",
        "docs/GPU-ENGINEERING.md",
        "docs/VALIDATION-AND-EVIDENCE.md",
        "docs/META-EXPERIMENTATION.md",
        "docs/COMMONS-INTEGRATION.md",
        "docs/ROADMAP.md",
        "contracts/climate.cue",
    )
    findings: list[Finding] = []
    for rel in required:
        if not (root / rel).is_file():
            findings.append(Finding(
                code="architecture.required_file_missing",
                severity="error",
                message=f"required architecture/control file is missing: {rel}",
                path=rel,
            ))
    return findings


def inspect_cargo(root: Path) -> list[Finding]:
    manifest = root / "Cargo.toml"
    if not manifest.is_file():
        return [Finding(
            code="cargo.manifest_missing",
            severity="warning",
            message="Cargo.toml is absent; Rust sources cannot be structurally resolved as a workspace",
            path="Cargo.toml",
        )]

    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [Finding(
            code="cargo.manifest_invalid",
            severity="error",
            message=f"Cargo.toml cannot be parsed: {exc}",
            path="Cargo.toml",
        )]

    findings: list[Finding] = []
    for member in data.get("workspace", {}).get("members", []):
        if not isinstance(member, str):
            findings.append(Finding(
                code="cargo.workspace_member_invalid",
                severity="error",
                message="Cargo workspace member must be a string",
                path="Cargo.toml",
                detail={"member": repr(member)},
            ))
            continue
        matches = list(root.glob(member))
        if not matches:
            findings.append(Finding(
                code="cargo.workspace_member_missing",
                severity="error",
                message=f"Cargo workspace member does not exist: {member}",
                path=member,
            ))
    return findings


# These are intentionally simple structural extractors, not a CMake parser.
# The goal is to detect literal repository paths asserted by the current file.
CMAKE_PATH_PATTERNS = (
    re.compile(r"\b(CORE/[A-Za-z0-9_.+/-]+)"),
    re.compile(r"add_subdirectory\(\s*([A-Za-z0-9_.+/-]+)\s*\)"),
    re.compile(r'configure_file\(\s*"?\$\{PROJECT_SOURCE_DIR\}/([^"\s)]+)'),
)


def _cmake_references(text: str) -> Iterable[str]:
    seen: set[str] = set()
    for pattern in CMAKE_PATH_PATTERNS:
        for match in pattern.finditer(text):
            rel = match.group(1).rstrip(";,)")
            if rel.startswith("${") or rel in seen:
                continue
            seen.add(rel)
            yield rel


def inspect_cmake(root: Path) -> list[Finding]:
    cmake = root / "CMakeLists.txt"
    if not cmake.is_file():
        return []
    try:
        text = cmake.read_text(encoding="utf-8")
    except OSError as exc:
        return [Finding(
            code="cmake.read_failed",
            severity="error",
            message=f"cannot read CMakeLists.txt: {exc}",
            path="CMakeLists.txt",
        )]

    findings: list[Finding] = []
    for rel in _cmake_references(text):
        # Generated output paths are not repository requirements. All current
        # extractors target source-side literal references.
        if not (root / rel).exists():
            findings.append(Finding(
                code="cmake.referenced_path_missing",
                severity="error",
                message=f"CMake references a repository path that does not exist: {rel}",
                path=rel,
            ))
    return findings


def inspect_source_risks(root: Path) -> list[Finding]:
    files = source_gates.read_sources(root)
    findings: list[Finding] = []
    for audit in source_gates.run(files):
        findings.append(Finding(
            code=f"source_audit.{audit.gate}",
            severity="warning",
            message=audit.message,
            path=f"{audit.path}:{audit.line}",
            detail={"source": audit.text.strip()},
        ))
    return findings


def summarize_inventory(entries: list[SourceEntry]) -> dict:
    by_language: dict[str, dict[str, int]] = {}
    for entry in entries:
        bucket = by_language.setdefault(entry.language, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += entry.bytes
    return {
        "source_files": len(entries),
        "source_bytes": sum(entry.bytes for entry in entries),
        "by_language": dict(sorted(by_language.items())),
    }


def inspect(root: Path = ROOT) -> dict:
    root = root.resolve()
    inventory = source_inventory(root)
    findings = [
        *inspect_required_architecture_files(root),
        *inspect_cargo(root),
        *inspect_cmake(root),
        *inspect_source_risks(root),
    ]
    findings.sort(key=lambda item: (item.severity, item.code, item.path or ""))
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1

    return {
        "schema_version": 1,
        "repository": "J0pari/Climate",
        "mode": "static-observe",
        "ready_for_execution": not any(f.severity == "error" for f in findings),
        "inventory": summarize_inventory(inventory),
        "finding_counts": counts,
        "findings": [asdict(f) for f in findings],
    }


def _print_human(report: dict) -> None:
    print(f"repository: {report['repository']}")
    print(f"mode: {report['mode']}")
    print(f"ready_for_execution: {str(report['ready_for_execution']).lower()}")
    inv = report["inventory"]
    print(f"source_files: {inv['source_files']} ({inv['source_bytes']} bytes)")
    for language, stats in inv["by_language"].items():
        print(f"  {language}: {stats['files']} file(s), {stats['bytes']} bytes")
    print("findings:")
    for finding in report["findings"]:
        path = f" [{finding['path']}]" if finding.get("path") else ""
        print(f"  {finding['severity']}: {finding['code']}: {finding['message']}{path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="statically inspect Climate repository readiness")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="return non-zero when structural errors are present")
    args = parser.parse_args()

    report = inspect(args.root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 1 if args.strict and not report["ready_for_execution"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
