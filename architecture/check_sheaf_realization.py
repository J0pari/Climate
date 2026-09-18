#!/usr/bin/env python3
"""Validate sheaf realization authority and production/reference separation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "methods" / "sheaf-realization.v1.json"
CLAIMS = ROOT / "claims" / "registry.json"
ALLOWED_STATUSES = {"open", "reference_realized", "canonical_realized", "validated"}
PRODUCTION_OBLIGATIONS = {
    "station_locality_complex",
    "climate_stalks_and_restrictions",
    "restriction_functoriality",
    "sparse_degree0_coboundary",
    "higher_cochain_d_squared_zero",
    "global_section_and_gluing",
    "synthetic_fault_falsification",
}

def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def validate_ledger(ledger: dict[str, Any], claims: dict[str, Any], root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    method_id = ledger.get("method_id")
    claim = next((item for item in claims.get("claims", []) if item.get("claim_id") == method_id), None)
    if claim is None:
        return [f"missing claim registry entry for {method_id!r}"]
    if ledger.get("claim_statement_pin") != claim.get("statement"):
        errors.append("claim statement differs from realization-ledger pin")
    obligations = ledger.get("obligations")
    if not isinstance(obligations, list) or not obligations:
        return errors + ["realization ledger must contain obligations"]

    seen: set[str] = set()
    production_open = False
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

        if obligation_id in PRODUCTION_OBLIGATIONS:
            if status == "reference_realized":
                errors.append(f"{obligation_id}: reference-only witnesses cannot discharge a production obligation")
            if status in {"canonical_realized", "validated"} and not any(
                isinstance(path, str) and path.startswith("src/") for path in witnesses
            ):
                errors.append(f"{obligation_id}: canonical production realization requires a src/ witness")
            if status == "open":
                production_open = True

    missing = PRODUCTION_OBLIGATIONS - seen
    if missing:
        errors.append("ledger omits required production obligations: " + ", ".join(sorted(missing)))
    if production_open and claim.get("maturity") in {"verified", "validated", "replicated", "decision-eligible"}:
        errors.append("claim maturity exceeds open production sheaf obligations; keep the claim below verified")
    return errors

def main() -> int:
    ledger = _load(LEDGER)
    claims = _load(CLAIMS)
    errors = validate_ledger(ledger, claims)
    obligations = ledger.get("obligations", [])
    counts = {status: sum(1 for item in obligations if item.get("status") == status) for status in ALLOWED_STATUSES}
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "sheaf realization ledger: "
        f"reference={counts['reference_realized']}, canonical={counts['canonical_realized']}, "
        f"validated={counts['validated']}, open={counts['open']}, total={len(obligations)}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
