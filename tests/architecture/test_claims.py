"""Negative witnesses for scientific claim-registry integrity."""
from __future__ import annotations

import unittest

from architecture import check_claims


def claim(claim_id: str, **overrides):
    base = {
        "claim_id": claim_id,
        "statement": "fixture",
        "claim_type": "software",
        "maturity": "concept",
        "scope": "fixture",
        "depends_on": [],
        "required_evidence": [],
        "supporting_evidence": [],
        "attacking_evidence": [],
        "unresolved_falsifiers": [],
    }
    base.update(overrides)
    return base


class ClaimIntegrityTests(unittest.TestCase):
    def test_duplicate_claim_id_is_rejected(self):
        findings = check_claims.check({
            "claims": [claim("x"), claim("x")],
        })
        self.assertIn("claims.id_duplicate", {f.code for f in findings})

    def test_missing_dependency_is_rejected(self):
        findings = check_claims.check({
            "claims": [claim("x", depends_on=["missing"])],
        })
        self.assertIn("claims.dependency_missing", {f.code for f in findings})

    def test_dependency_cycle_is_rejected(self):
        findings = check_claims.check({
            "claims": [
                claim("a", depends_on=["b"]),
                claim("b", depends_on=["a"]),
            ],
        })
        self.assertIn("claims.dependency_cycle", {f.code for f in findings})

    def test_validated_claim_requires_supporting_evidence(self):
        findings = check_claims.check({
            "claims": [
                claim(
                    "x",
                    maturity="validated",
                    required_evidence=["independent validation"],
                )
            ],
        })
        self.assertIn("claims.maturity_without_support", {f.code for f in findings})

    def test_decision_eligible_claim_requires_policy(self):
        findings = check_claims.check({
            "claims": [
                claim(
                    "x",
                    maturity="decision-eligible",
                    required_evidence=["decision validation"],
                    supporting_evidence=["e1"],
                )
            ],
        })
        self.assertIn("claims.decision_policy_missing", {f.code for f in findings})

    def test_evidence_reference_must_resolve_when_registry_is_supplied(self):
        findings = check_claims.check(
            {
                "claims": [claim("x", supporting_evidence=["missing-evidence"])],
            },
            {"evidence": []},
        )
        self.assertIn("claims.evidence_missing", {f.code for f in findings})

    def test_valid_concept_registry_passes(self):
        findings = check_claims.check({
            "claims": [
                claim("a"),
                claim("b", depends_on=["a"]),
            ],
        })
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
