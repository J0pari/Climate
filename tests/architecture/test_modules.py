"""Negative witnesses for Climate module inventory integrity."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture import check_modules


def claims(*ids: str):
    return {"claims": [{"claim_id": value} for value in ids]}


def module(path: str, **overrides):
    base = {
        "path": path,
        "family": "fixture",
        "language": check_modules.SOURCE_SUFFIXES.get(Path(path).suffix, "unknown"),
        "maturity": "prototype",
        "evidence_eligible": False,
        "intended_role": "fixture",
        "known_gaps": ["not verified"],
    }
    base.update(overrides)
    return base


class ModuleIntegrityTests(unittest.TestCase):
    def test_unregistered_top_level_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "new_method.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(root, {"modules": []}, claims())
            self.assertIn("modules.source_unregistered", {f.code for f in findings})

    def test_duplicate_module_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module("x.rs"), module("x.rs")]},
                claims(),
            )
            self.assertIn("modules.path_duplicate", {f.code for f in findings})

    def test_language_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module("x.rs", language="python")]},
                claims(),
            )
            self.assertIn("modules.language_mismatch", {f.code for f in findings})

    def test_missing_claim_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module("x.rs", claim_ids=["missing"])]},
                claims("known"),
            )
            self.assertIn("modules.claim_missing", {f.code for f in findings})

    def test_prototype_requires_known_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module("x.rs", known_gaps=[])]},
                claims(),
            )
            self.assertIn("modules.known_gaps_missing", {f.code for f in findings})

    def test_unverified_module_cannot_be_evidence_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module(
                    "x.rs",
                    evidence_eligible=True,
                    claim_ids=["c"],
                )]},
                claims("c"),
            )
            self.assertIn("modules.evidence_eligibility_too_early", {f.code for f in findings})

    def test_evidence_eligible_module_requires_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module(
                    "x.rs",
                    maturity="verified",
                    evidence_eligible=True,
                    known_gaps=[],
                )]},
                claims(),
            )
            self.assertIn("modules.evidence_eligibility_without_claim", {f.code for f in findings})

    def test_registered_prototype_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x.rs").write_text("fn main() {}\n", encoding="utf-8")
            findings = check_modules.check(
                root,
                {"modules": [module("x.rs", claim_ids=["c"])]},
                claims("c"),
            )
            self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
