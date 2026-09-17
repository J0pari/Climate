#!/usr/bin/env python3
"""Enforce Climate's repository-root ownership boundary.

Top-level source is reserved for repository manifests, build entrypoints, and
contracts. Scientific implementation belongs in an owned source/package tree;
Git history, not a root-level compatibility attic, preserves depleted code.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from architecture import source_surface


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str


def check(root: Path = ROOT) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.name.startswith("climate_"):
            findings.append(Finding(
                "root_layout.climate_prefixed_file",
                relative,
                "root-level climate_* files are prohibited; place live content under an owned architecture tree",
            ))
            continue
        if path.suffix in source_surface.SOURCE_LANGUAGES:
            findings.append(Finding(
                "root_layout.scientific_source",
                relative,
                "scientific source files must not live at repository root",
            ))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="check Climate repository-root layout")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    findings = check(args.root)
    if args.json:
        print(json.dumps({
            "ok": not findings,
            "finding_count": len(findings),
            "findings": [asdict(item) for item in findings],
        }, indent=2, sort_keys=True))
    elif findings:
        for finding in findings:
            print(f"{finding.code}: {finding.path}: {finding.message}")
    else:
        print("root layout: clean")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
