#!/usr/bin/env python3
"""Cross-record integrity checks for Climate's scientific claim registry.

CUE validates record shape. This checker validates graph properties and maturity
rules that are awkward or undesirable to encode as local schema constraints.
It does not decide whether a scientific claim is true.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLAIMS = ROOT / "claims" / "registry.json"
DEFAULT_EVIDENCE = ROOT / "evidence" / "registry.json"

MATURITY_RANK = {
    "concept": 0,
    "prototype": 1,
    "runnable": 2,
    "verified": 3,
    "validated": 4,
    "replicated": 5,
    "decision-eligible": 6,
}


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    claim_id: str | None = None
    reference: str | None = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _claim_map(registry: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[Finding]]:
    findings: list[Finding] = []
    out: dict[str, dict[str, Any]] = {}
    claims = registry.get("claims", [])
    if not isinstance(claims, list):
        return out, [Finding("claims.registry_not_list", "claims must be a list")]
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            findings.append(Finding(
                "claims.claim_not_object",
                f"claim at index {index} is not an object",
            ))
            continue
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id:
            findings.append(Finding(
                "claims.id_missing",
                f"claim at index {index} has no usable claim_id",
            ))
            continue
        if claim_id in out:
            findings.append(Finding(
                "claims.id_duplicate",
                "claim_id appears more than once",
                claim_id=claim_id,
            ))
            continue
        out[claim_id] = claim
    return out, findings


def _evidence_ids(registry: dict[str, Any] | None) -> set[str]:
    if registry is None:
        return set()
    records = registry.get("evidence", [])
    if not isinstance(records, list):
        return set()
    return {
        item.get("evidence_id")
        for item in records
        if isinstance(item, dict) and isinstance(item.get("evidence_id"), str)
    }


def _find_cycle(claims: dict[str, dict[str, Any]]) -> list[str] | None:
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        if node in visiting:
            start = stack.index(node)
            return stack[start:] + [node]
        if node in visited:
            return None
        visiting.add(node)
        stack.append(node)
        for dep in claims[node].get("depends_on", []):
            if dep in claims:
                cycle = visit(dep)
                if cycle:
                    return cycle
        stack.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for claim_id in claims:
        cycle = visit(claim_id)
        if cycle:
            return cycle
    return None


def check(
    claim_registry: dict[str, Any],
    evidence_registry: dict[str, Any] | None = None,
) -> list[Finding]:
    claims, findings = _claim_map(claim_registry)
    if evidence_registry is None:
        findings.append(Finding(
            "claims.evidence_registry_missing",
            "evidence registry is required; absence must not disable evidence-reference validation",
        ))
    evidence_ids = _evidence_ids(evidence_registry)

    for claim_id, claim in claims.items():
        maturity = claim.get("maturity")
        if maturity not in MATURITY_RANK:
            findings.append(Finding(
                "claims.maturity_unknown",
                f"unknown maturity {maturity!r}",
                claim_id=claim_id,
            ))

        dependencies = claim.get("depends_on", [])
        if isinstance(dependencies, list):
            for dep in dependencies:
                if dep == claim_id:
                    findings.append(Finding(
                        "claims.self_dependency",
                        "claim depends on itself",
                        claim_id=claim_id,
                        reference=dep,
                    ))
                elif dep not in claims:
                    findings.append(Finding(
                        "claims.dependency_missing",
                        "claim dependency does not resolve",
                        claim_id=claim_id,
                        reference=str(dep),
                    ))

        supporting = claim.get("supporting_evidence", [])
        attacking = claim.get("attacking_evidence", [])
        for relation, refs in (("supporting", supporting), ("attacking", attacking)):
            if not isinstance(refs, list):
                continue
            for ref in refs:
                if evidence_registry is not None and ref not in evidence_ids:
                    findings.append(Finding(
                        "claims.evidence_missing",
                        f"{relation} evidence reference does not resolve",
                        claim_id=claim_id,
                        reference=str(ref),
                    ))

        if maturity in {"validated", "replicated", "decision-eligible"} and not supporting:
            findings.append(Finding(
                "claims.maturity_without_support",
                f"{maturity} claim has no supporting evidence references",
                claim_id=claim_id,
            ))

        if maturity == "decision-eligible" and not claim.get("decision_policy"):
            findings.append(Finding(
                "claims.decision_policy_missing",
                "decision-eligible claim requires an explicit decision policy",
                claim_id=claim_id,
            ))

        if maturity not in {"concept", "prototype"} and not claim.get("required_evidence"):
            findings.append(Finding(
                "claims.required_evidence_missing",
                "claim above prototype maturity must declare required evidence",
                claim_id=claim_id,
            ))

    cycle = _find_cycle(claims)
    if cycle:
        findings.append(Finding(
            "claims.dependency_cycle",
            "claim dependency graph contains a cycle: " + " -> ".join(cycle),
            claim_id=cycle[0],
        ))

    return sorted(findings, key=lambda item: (item.code, item.claim_id or "", item.reference or ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="check Climate claim registry integrity")
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    claims = load_json(args.claims)
    evidence = load_json(args.evidence) if args.evidence.is_file() else None
    findings = check(claims, evidence)

    if args.json:
        print(json.dumps({
            "ok": not findings,
            "finding_count": len(findings),
            "findings": [asdict(item) for item in findings],
        }, indent=2, sort_keys=True))
    elif findings:
        for finding in findings:
            suffix = f" [{finding.claim_id}]" if finding.claim_id else ""
            ref = f" -> {finding.reference}" if finding.reference else ""
            print(f"{finding.code}: {finding.message}{suffix}{ref}")
    else:
        print(f"claims: {len(claims.get('claims', []))} records; integrity ok")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
