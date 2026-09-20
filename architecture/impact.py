#!/usr/bin/env python3
"""Derived reverse-reference view for estimating repository change impact.

This tool creates no new authority. It answers a narrower question: which
existing repository surfaces mention a path exactly? The result is useful
before edits because it exposes module registrations, planning evidence,
realization/evidence authorities, tests, generated projections, and
durable contracts that may need to be re-read when the target changes.

Absence from this view is not proof of independence: dynamic imports, semantic
coupling without a literal path reference, external systems, and runtime data
dependencies require their own witnesses.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

_TEXT_SUFFIXES = {
    ".c", ".cc", ".cmake", ".cpp", ".cuh", ".cu", ".cue", ".cxx", ".f90",
    ".F90", ".h", ".hpp", ".hs", ".jl", ".json", ".md", ".py", ".rs",
    ".toml", ".txt", ".yaml", ".yml",
}
_EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", "build", "build-reference",
    "target",
}
_MAX_TEXT_BYTES = 2_000_000


@dataclass(frozen=True)
class ImpactReference:
    kind: str
    path: str
    line: int
    text: str


def _normalize_target(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or not path.parts:
        raise ValueError("target must be a non-empty repository-relative path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("target must not contain '.', '..', or empty path components")
    return path.as_posix()


def _reference_kind(relative: Path) -> str:
    parts = relative.parts
    if len(parts) >= 3 and parts[:2] == ("architecture", "modules"):
        return "module_authority"
    if relative == Path("architecture/planning_graph.json"):
        return "planning_authority"
    if parts and parts[0] == "architecture" and relative.suffix == ".json":
        return "architecture_authority"
    if parts and parts[0] in {
        "claims", "methods", "experiments", "evaluations", "configurations"
    } and relative.suffix == ".json":
        return "scientific_authority"
    if parts and parts[0] == "contracts":
        return "contract"
    if parts and parts[0] == "tests":
        return "test"
    if relative == Path("docs/ROADMAP.md") or parts[:2] == ("docs", "generated"):
        return "generated_projection"
    if relative == Path("AGENTS.md") or (parts and parts[0] == "docs"):
        return "durable_contract"
    if parts and parts[0] in {"src", "data", "reference"}:
        return "source"
    return "repository_metadata"


def _candidate_text_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _EXCLUDED_PARTS or part.startswith(".pytest") for part in relative.parts):
            continue
        if path.name in {"AGENTS.md", "CMakeLists.txt", "Cargo.toml"}:
            yield path
            continue
        if path.suffix not in _TEXT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > _MAX_TEXT_BYTES:
                continue
        except OSError:
            continue
        yield path


def references_for_target(root: Path, target: str) -> list[ImpactReference]:
    root = root.resolve()
    normalized = _normalize_target(target)
    references: list[ImpactReference] = []
    for path in _candidate_text_files(root):
        relative = path.relative_to(root)
        if relative.as_posix() == normalized:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(lines, start=1):
            if normalized not in line:
                continue
            references.append(
                ImpactReference(
                    kind=_reference_kind(relative),
                    path=relative.as_posix(),
                    line=line_number,
                    text=line.strip(),
                )
            )
    return sorted(
        references,
        key=lambda item: (item.kind, item.path, item.line, item.text),
    )


def inspect_impact(root: Path, targets: list[str]) -> dict:
    root = root.resolve()
    rows = []
    for value in targets:
        target = _normalize_target(value)
        refs = references_for_target(root, target)
        counts = Counter(item.kind for item in refs)
        rows.append({
            "path": target,
            "exists": (root / target).exists(),
            "reference_count": len(refs),
            "references_by_kind": dict(sorted(counts.items())),
            "references": [asdict(item) for item in refs],
        })
    return {
        "schema_version": 1,
        "mode": "derived-reverse-reference",
        "authority": "none",
        "targets": rows,
        "limitations": [
            "literal path references only; semantic coupling without the exact path string is not inferred",
            "runtime, external-system, dynamic-import, and data dependencies require independent inspection",
            "a reference indicates possible impact, not a requirement to edit the referencing surface",
        ],
    }


def _print_human(report: dict) -> None:
    for index, target in enumerate(report["targets"]):
        if index:
            print()
        print(f"target: {target['path']}")
        print(f"exists: {str(target['exists']).lower()}")
        print(f"references: {target['reference_count']}")
        for kind, count in target["references_by_kind"].items():
            print(f"  {kind}: {count}")
        for item in target["references"]:
            print(
                f"  - {item['kind']}: {item['path']}:{item['line']}: "
                f"{item['text']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="derive exact-path reverse references for change-impact inspection"
    )
    parser.add_argument("paths", nargs="+", help="repository-relative path(s) to inspect")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        report = inspect_impact(args.root, args.paths)
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
