#!/usr/bin/env python3
"""Validate staged realization and semantic authority for sheaf/cohomology work.

The checker does not decide whether the scientific claim is true. It prevents
two cheaper substitutes for progress:

* weakening the pinned claim statement;
* marking mathematical obligations realized without concrete witnesses.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "methods" / "sheaf-realization.v1.json"
CLAIMS = ROOT / "claims" / "registry.json"
ALLOWED_STATUSES = {"open", "reference_realized", "validated"}
STRUCTURAL_OBLIGATIONS = {
    "station_cover_nerve",
    "climate_stalks_and_restrictions",
    "restriction_functoriality",
    "data_sheaf_coboundary",
    "global_section_and_gluing",
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_ledger(
    ledger: dict[str, Any], claims: dict[str, Any], root: Path = ROOT
) -> list[str]:
    errors: list[str] = []
    method_id = ledger.get("method_id")
    claim = next(
        (item for item in claims.get("claims", []) if item.get("claim_id") == method_id),
        None,
    )
    if claim is None:
        errors.append(f"missing claim registry entry for {method_id!r}")
        return errors

    if ledger.get("claim_statement_pin") != claim.get("statement"):
        errors.append("claim statement differs from realization-ledger pin")

    obligations = ledger.get("obligations")
    if not isinstance(obligations, list) or not obligations:
        errors.append("realization ledger must contain obligations")
        return errors

    seen: set[str] = set()
    structural_open = False
    for item in obligations:
        obligation_id = item.get("obligation_id")
        if not isinstance(obligation_id, str) or not obligation_id:
            errors.append("obligation missing obligation_id")
            continue
        if obligation_id in seen:
            errors.append(f"duplicate obligation_id {obligation_id}")
        seen.add(obligation_id)

        status = item.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{obligation_id}: unknown status {status!r}")
            continue

        witnesses = item.get("witnesses")
        if not isinstance(witnesses, list):
            errors.append(f"{obligation_id}: witnesses must be a list")
            continue

        if status != "open" and not witnesses:
            errors.append(f"{obligation_id}: realized obligation has no witnesses")
        if status == "open" and witnesses:
            errors.append(f"{obligation_id}: open obligation must not imply witness completion")

        for witness in witnesses:
            if not isinstance(witness, str) or not witness:
                errors.append(f"{obligation_id}: invalid witness path {witness!r}")
                continue
            candidate = (root / witness).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                errors.append(f"{obligation_id}: witness escapes repository: {witness}")
                continue
            if not candidate.is_file():
                errors.append(f"{obligation_id}: witness file does not exist: {witness}")

        if obligation_id in STRUCTURAL_OBLIGATIONS and status == "open":
            structural_open = True

    missing_structural = STRUCTURAL_OBLIGATIONS - seen
    if missing_structural:
        errors.append(
            "ledger omits required structural obligations: "
            + ", ".join(sorted(missing_structural))
        )

    if structural_open and claim.get("maturity") in {
        "verified", "validated", "replicated", "decision-eligible"
    }:
        errors.append(
            "claim maturity exceeds open sheaf-structure obligations; keep the claim below verified"
        )

    return errors


def main() -> int:
    ledger = _load(LEDGER)
    claims = _load(CLAIMS)
    errors = validate_ledger(ledger, claims)
    obligations = ledger.get("obligations", [])
    realized = sum(1 for item in obligations if item.get("status") != "open")
    total = len(obligations)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"sheaf realization ledger: {realized}/{total} obligations realized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
