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
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

# Support both `python -m architecture.inspect_repository` and the documented
# direct-script entrypoint `python architecture/inspect_repository.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from architecture import source_gates, source_surface


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
    role: str
    bytes: int


@dataclass(frozen=True)
class CMakeSubdirectory:
    path: str
    guards: tuple[str, ...]


def source_inventory(root: Path) -> list[SourceEntry]:
    entries: list[SourceEntry] = []
    for item in source_surface.iter_source_files(root):
        entries.append(SourceEntry(
            path=item.relative_path,
            language=item.language,
            role=item.role,
            bytes=item.path.stat().st_size,
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
        "contracts/gpu-scheduler-pin.json",
        "contracts/work-scheduler-pin.json",
        "architecture/commons_interface.json",
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


def _cargo_existing_targets(root: Path, data: dict) -> set[str]:
    """Find existing conventional or explicitly named Cargo lib/bin targets."""
    candidates: set[Path] = {root / "src/lib.rs", root / "src/main.rs"}
    bin_dir = root / "src/bin"
    if bin_dir.is_dir():
        candidates.update(bin_dir.glob("*.rs"))

    lib = data.get("lib")
    if isinstance(lib, dict) and isinstance(lib.get("path"), str):
        candidates.add(root / lib["path"])

    bins = data.get("bin", [])
    if isinstance(bins, list):
        for entry in bins:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("path"), str):
                candidates.add(root / entry["path"])
            elif isinstance(entry.get("name"), str):
                candidates.add(root / "src/bin" / f"{entry['name']}.rs")

    return {
        path.relative_to(root).as_posix()
        for path in candidates
        if path.is_file()
    }


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
        if not list(root.glob(member)):
            findings.append(Finding(
                code="cargo.workspace_member_missing",
                severity="error",
                message=f"Cargo workspace member does not exist: {member}",
                path=member,
            ))

    if isinstance(data.get("package"), dict) and not _cargo_existing_targets(root, data):
        findings.append(Finding(
            code="cargo.package_target_missing",
            severity="error",
            message="Cargo.toml declares a root package but no existing library or binary target is discoverable",
            path="Cargo.toml",
        ))
    return findings


# These are intentionally simple structural extractors, not a CMake parser.
CMAKE_PATH_PATTERNS = (
    re.compile(r"\b(CORE/[A-Za-z0-9_.+/-]+)"),
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


def _cmake_option_defaults(text: str) -> dict[str, bool]:
    result: dict[str, bool] = {}
    pattern = re.compile(
        r'option\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+"[^"]*"\s+(ON|OFF)\s*\)',
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        result[match.group(1)] = match.group(2).upper() == "ON"
    return result


def _cmake_subdirectories(text: str) -> list[CMakeSubdirectory]:
    """Track only simple if(VAR) guards; unknown conditions stay conservative."""
    stack: list[str | None] = []
    result: list[CMakeSubdirectory] = []
    if_pattern = re.compile(r"^if\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$", re.IGNORECASE)
    endif_pattern = re.compile(r"^endif(?:\([^)]*\))?\s*$", re.IGNORECASE)
    add_pattern = re.compile(r"add_subdirectory\(\s*([A-Za-z0-9_.+/-]+)", re.IGNORECASE)

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        match = if_pattern.match(line)
        if match:
            stack.append(match.group(1))
            continue
        if endif_pattern.match(line):
            if stack:
                stack.pop()
            continue
        match = add_pattern.search(line)
        if match:
            result.append(CMakeSubdirectory(
                path=match.group(1).rstrip(";,)") ,
                guards=tuple(value for value in stack if value is not None),
            ))
    return result


def _cmake_project_languages(text: str) -> set[str]:
    match = re.search(r"\bproject\s*\((.*?)\)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return set()
    languages = re.search(r"\bLANGUAGES\b(.*)$", match.group(1), re.IGNORECASE | re.DOTALL)
    if not languages:
        return set()
    return {
        token.upper()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_+.-]*", languages.group(1))
    }


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
    if "CUDA" in _cmake_project_languages(text):
        findings.append(Finding(
            code="cmake.cuda_language_unconditional",
            severity="error",
            message="CUDA is enabled by project(... LANGUAGES ...) before ENABLE_CUDA can make accelerator support optional",
            path="CMakeLists.txt",
        ))

    for rel in _cmake_references(text):
        if not (root / rel).exists():
            findings.append(Finding(
                code="cmake.referenced_path_missing",
                severity="error",
                message=f"CMake references a repository path that does not exist: {rel}",
                path=rel,
            ))

    option_defaults = _cmake_option_defaults(text)
    for subdir in _cmake_subdirectories(text):
        target = root / subdir.path
        disabled_by_default = any(option_defaults.get(guard) is False for guard in subdir.guards)
        severity = "warning" if disabled_by_default else "error"
        suffix = " (guarded by a default-OFF option)" if disabled_by_default else ""
        detail = {"guards": list(subdir.guards)}

        if not target.exists():
            findings.append(Finding(
                code="cmake.optional_subdirectory_missing" if disabled_by_default else "cmake.subdirectory_missing",
                severity=severity,
                message=f"CMake add_subdirectory target does not exist: {subdir.path}{suffix}",
                path=subdir.path,
                detail=detail,
            ))
        elif target.is_dir() and not (target / "CMakeLists.txt").is_file():
            findings.append(Finding(
                code="cmake.optional_subdirectory_manifest_missing" if disabled_by_default else "cmake.subdirectory_manifest_missing",
                severity=severity,
                message=f"CMake add_subdirectory target has no CMakeLists.txt: {subdir.path}{suffix}",
                path=f"{subdir.path}/CMakeLists.txt",
                detail=detail,
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
    by_role: dict[str, dict[str, int]] = {}
    for entry in entries:
        language_bucket = by_language.setdefault(entry.language, {"files": 0, "bytes": 0})
        language_bucket["files"] += 1
        language_bucket["bytes"] += entry.bytes
        role_bucket = by_role.setdefault(entry.role, {"files": 0, "bytes": 0})
        role_bucket["files"] += 1
        role_bucket["bytes"] += entry.bytes
    return {
        "source_files": len(entries),
        "source_bytes": sum(entry.bytes for entry in entries),
        "by_language": dict(sorted(by_language.items())),
        "by_role": dict(sorted(by_role.items())),
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
    print("by_language:")
    for language, stats in inv["by_language"].items():
        print(f"  {language}: {stats['files']} file(s), {stats['bytes']} bytes")
    print("by_role:")
    for role, stats in inv["by_role"].items():
        print(f"  {role}: {stats['files']} file(s), {stats['bytes']} bytes")
    print("findings:")
    for finding in report["findings"]:
        path = f" [{finding['path']}]" if finding.get("path") else ""
        print(f"  {finding['severity']}: {finding['code']}: {finding['message']}{path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="statically inspect Climate repository readiness")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="return non-zero when structural errors are present")
    args = parser.parse_args()
    report = inspect(args.root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 1 if args.strict and not report["ready_for_execution"] else 0


if __name__ == "__main__":
    raise SystemExit(main())