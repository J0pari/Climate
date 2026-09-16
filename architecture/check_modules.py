#!/usr/bin/env python3
"""Integrity checks for Climate's module/maturity inventory.

The module inventory is the bridge between legacy source and the scientific
claim/evidence system. It must stay complete enough that adding or moving a
scientific source file cannot silently escape maturity and known-gap tracking.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODULES = ROOT / "architecture" / "modules.json"
DEFAULT_CLAIMS = ROOT / "claims" / "registry.json"

SOURCE_SUFFIXES = {
    ".rs": "rust",
    ".py": "python",
    ".cu": "cuda",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".f90": "fortran",
    ".F90": "fortran",
    ".jl": "julia",
    ".hs": "haskell",
}

MATURITY_RANK = {
    "concept": 0,
    "prototype": 1,
    "runnable": 2,
    "verified": 3,
    "validated": 4,
    "replicated": 5,
    "decision-eligible": 6,
}

# The current legacy scientific implementation is flat at repository root.
# New architecture-control/test code is intentionally excluded. When the
# migration introduces package directories, this discovery rule should change
# in the same commit as the layout contract.
def discover_legacy_sources(root: Path) -> set[str]:
    return {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    }


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    path: str | None = None
    reference: str | None = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _claim_ids(claim_registry: dict[str, Any]) -> set[str]:
    claims = claim_registry.get("claims", [])
    if not isinstance(claims, list):
        return set()
    return {
        item.get("claim_id")
        for item in claims
        if isinstance(item, dict) and isinstance(item.get("claim_id"), str)
    }


def check(
    root: Path,
    module_registry: dict[str, Any],
    claim_registry: dict[str, Any],
) -> list[Finding]:
    findings: list[Finding] = []
    modules = module_registry.get("modules", [])
    if not isinstance(modules, list):
        return [Finding("modules.registry_not_list", "modules must be a list")]

    claims = _claim_ids(claim_registry)
    registered: dict[str, dict[str, Any]] = {}

    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            findings.append(Finding(
                "modules.entry_not_object",
                f"module at index {index} is not an object",
            ))
            continue
        path = module.get("path")
        if not isinstance(path, str) or not path:
            findings.append(Finding(
                "modules.path_missing",
                f"module at index {index} has no usable path",
            ))
            continue
        if path in registered:
            findings.append(Finding(
                "modules.path_duplicate",
                "module path appears more than once",
                path=path,
            ))
            continue
        registered[path] = module

        file_path = root / path
        if not file_path.is_file():
            findings.append(Finding(
                "modules.path_missing_on_disk",
                "registered module path does not exist",
                path=path,
            ))
            continue

        expected_language = SOURCE_SUFFIXES.get(file_path.suffix)
        actual_language = module.get("language")
        if expected_language and actual_language != expected_language:
            findings.append(Finding(
                "modules.language_mismatch",
                f"declared language {actual_language!r} does not match extension language {expected_language!r}",
                path=path,
            ))

        maturity = module.get("maturity")
        if maturity not in MATURITY_RANK:
            findings.append(Finding(
                "modules.maturity_unknown",
                f"unknown maturity {maturity!r}",
                path=path,
            ))

        claim_ids = module.get("claim_ids", [])
        if not isinstance(claim_ids, list):
            findings.append(Finding(
                "modules.claim_ids_not_list",
                "claim_ids must be a list when present",
                path=path,
            ))
            claim_ids = []
        for claim_id in claim_ids:
            if claim_id not in claims:
                findings.append(Finding(
                    "modules.claim_missing",
                    "module claim reference does not resolve",
                    path=path,
                    reference=str(claim_id),
                ))

        gaps = module.get("known_gaps", [])
        if maturity in {"concept", "prototype"} and (not isinstance(gaps, list) or not gaps):
            findings.append(Finding(
                "modules.known_gaps_missing",
                "concept/prototype module must carry at least one explicit known gap",
                path=path,
            ))

        eligible = module.get("evidence_eligible")
        if eligible is True:
            if MATURITY_RANK.get(maturity, -1) < MATURITY_RANK["verified"]:
                findings.append(Finding(
                    "modules.evidence_eligibility_too_early",
                    "evidence_eligible=true requires verified or higher maturity",
                    path=path,
                ))
            if not claim_ids:
                findings.append(Finding(
                    "modules.evidence_eligibility_without_claim",
                    "evidence-eligible module must resolve to at least one scientific/software claim",
                    path=path,
                ))
        elif eligible is not False:
            findings.append(Finding(
                "modules.evidence_eligibility_missing",
                "evidence_eligible must be an explicit boolean",
                path=path,
            ))

    discovered = discover_legacy_sources(root)
    registered_paths = set(registered)
    for path in sorted(discovered - registered_paths):
        findings.append(Finding(
            "modules.source_unregistered",
            "legacy scientific source exists without a module/maturity record",
            path=path,
        ))
    for path in sorted(registered_paths - discovered):
        # Nested paths are allowed for future package migration; if they exist
        # they were already accepted above. Only missing paths are errors here.
        if not (root / path).is_file():
            continue

    return sorted(findings, key=lambda item: (item.code, item.path or "", item.reference or ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="check Climate module inventory integrity")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--modules", type=Path, default=DEFAULT_MODULES)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    modules = load_json(args.modules)
    claims = load_json(args.claims)
    findings = check(args.root.resolve(), modules, claims)

    if args.json:
        print(json.dumps({
            "ok": not findings,
            "finding_count": len(findings),
            "findings": [asdict(item) for item in findings],
        }, indent=2, sort_keys=True))
    elif findings:
        for finding in findings:
            suffix = f" [{finding.path}]" if finding.path else ""
            ref = f" -> {finding.reference}" if finding.reference else ""
            print(f"{finding.code}: {finding.message}{suffix}{ref}")
    else:
        print(f"modules: {len(modules.get('modules', []))} records; integrity ok")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
