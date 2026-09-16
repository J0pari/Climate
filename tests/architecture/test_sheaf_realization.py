"""Negative witnesses for the sheaf realization promotion guard."""
from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "architecture" / "check_sheaf_realization.py"
spec = importlib.util.spec_from_file_location("check_sheaf_realization", CHECKER)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

with (ROOT / "methods" / "sheaf-realization.v1.json").open("r", encoding="utf-8") as handle:
    BASE_LEDGER = json.load(handle)
with (ROOT / "claims" / "registry.json").open("r", encoding="utf-8") as handle:
    BASE_CLAIMS = json.load(handle)


class SheafRealizationGuardTests(unittest.TestCase):
    def test_current_ledger_is_valid(self):
        self.assertEqual(module.validate_ledger(BASE_LEDGER, BASE_CLAIMS, ROOT), [])

    def test_current_legacy_surface_has_no_forbidden_authority_symbols(self):
        source = (ROOT / "climate_multiscale_sheaf.hs").read_text(encoding="utf-8")
        self.assertEqual(module.validate_legacy_semantics(source), [])

    def test_legacy_authority_symbol_reintroduction_is_rejected(self):
        source = "computeBettiNumbers = undefined\n"
        errors = module.validate_legacy_semantics(source)
        self.assertTrue(any("computeBettiNumbers" in error for error in errors), errors)

    def test_statement_weakening_or_drift_is_rejected(self):
        claims = copy.deepcopy(BASE_CLAIMS)
        claim = next(
            item
            for item in claims["claims"]
            if item["claim_id"] == BASE_LEDGER["method_id"]
        )
        claim["statement"] = "Sheaf methods identify climate-data inconsistencies."
        errors = module.validate_ledger(BASE_LEDGER, claims, ROOT)
        self.assertTrue(any("statement differs" in error for error in errors), errors)

    def test_realized_obligation_without_witness_is_rejected(self):
        ledger = copy.deepcopy(BASE_LEDGER)
        obligation = next(
            item for item in ledger["obligations"] if item["status"] == "open"
        )
        obligation["status"] = "reference_realized"
        errors = module.validate_ledger(ledger, BASE_CLAIMS, ROOT)
        self.assertTrue(any("has no witnesses" in error for error in errors), errors)

    def test_nonexistent_witness_is_rejected(self):
        ledger = copy.deepcopy(BASE_LEDGER)
        obligation = ledger["obligations"][0]
        obligation["witnesses"] = ["reference/definitely-not-present.py"]
        errors = module.validate_ledger(ledger, BASE_CLAIMS, ROOT)
        self.assertTrue(any("does not exist" in error for error in errors), errors)

    def test_open_structural_obligation_blocks_verified_claim_maturity(self):
        claims = copy.deepcopy(BASE_CLAIMS)
        claim = next(
            item
            for item in claims["claims"]
            if item["claim_id"] == BASE_LEDGER["method_id"]
        )
        claim["maturity"] = "verified"
        errors = module.validate_ledger(BASE_LEDGER, claims, ROOT)
        self.assertTrue(any("maturity exceeds" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
